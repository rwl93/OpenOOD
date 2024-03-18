from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm
import os

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

COV_ESTIMATOR={
    'emp': sklearn.covariance.EmpiricalCovariance,
    'oas': sklearn.covariance.OAS,
    'lw': sklearn.covariance.LedoitWolf,
}

class MDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.cov_estimator = config.postprocessor.postprocessor_args.cov_estimator

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            fname = 'vit-b-16-img1k-feats.pkl'
            fname = os.path.join(os.getcwd(), fname)
            if os.path.isfile(fname):
                dat = torch.load(os.path.join(os.getcwd(), 'vit-b-16-img1k-feats.pkl'))
                all_feats = dat['feats']
                all_labels = dat['labels']
                all_preds = dat['preds']
            else:
                all_feats = []
                all_labels = []
                all_preds = []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Setup: ',
                                      position=0,
                                      leave=True):
                        data, labels = batch['data'].cuda(), batch['label']
                        logits, features = net(data, return_feature=True)
                        all_feats.append(features.cpu())
                        all_labels.append(deepcopy(labels))
                        all_preds.append(logits.argmax(1).cpu())
                all_feats = torch.cat(all_feats)
                all_labels = torch.cat(all_labels)
                all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # compute class-conditional statistics
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            # group_lasso = sklearn.covariance.EmpiricalCovariance(
            #     assume_centered=False)
            covest = COV_ESTIMATOR[self.cov_estimator](
                assume_centered=False)
            covest.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(covest.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)

        class_scores = torch.zeros((logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()

        conf = torch.max(class_scores, dim=1)[0]
        return pred, conf
