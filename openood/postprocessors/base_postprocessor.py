from typing import Any, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm


class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self._params = {}

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list

    def _update_params(self):
        for k in self._params.keys():
            if not hasattr(self, k):
                raise AttributeError(f'{k} in params not set in setup or setup not run yet')
            self._params[k] = getattr(self, k)

    @property
    def params(self):
        self._update_params()
        return self._params

    @params.setter
    def params(self, value):
        if self._params.keys() != value.keys():
            raise ValueError("Invalid value for params: Keys do not match.")
        for k, v in value.items():
            setattr(self, k, v)
        self._update_params()

    def extract_features(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        feature_list, label_list = [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            feats = self.extract_features_batch(net, data)

            feature_list.append(feats.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        # feature_list = torch.cat(feature_list).numpy().astype(float)
        label_list = torch.cat(label_list).numpy().astype(int)

        return feature_list, label_list

    @torch.no_grad()
    def extract_features_batch(self, net, data):
        _, features = net(data, return_feature=True)
        return features

    def gibbs_inference(self,
                  feats: List[torch.FloatTensor],
                  progress: bool = True):
        pred_list, conf_list = [], []
        for batch in tqdm(feats,
                          disable=not progress or not comm.is_main_process()):
            data = batch.cuda()
            pred, conf = self.postprocess(None, data, from_feats=True) # type: ignore

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()

        return pred_list, conf_list
