import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dists
from torch.utils.data import DataLoader

import openood.utils.comm as comm
from openood.utils import Config


class ProtoDPMMTrainer:
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        config: Config):

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.optimmizer = torch.optim.Adam(
            net.parameters()
            config.optimizer.lr,
            weight_decay = config.optimizer.weight_decay,
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            loss = self.net.loss(data, target,
                                 nosample=self.config.loss.nosample,
                                 nodpmmupdate=False,
                                 beta=self.config.loss.beta,
                                ):

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
