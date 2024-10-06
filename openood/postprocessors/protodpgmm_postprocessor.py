import os
from typing import Any
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm,trange

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

import scipy
from sklearn.decomposition import PCA


MAX_SAMPLES = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultivariateT(dists.Distribution):
    def __init__(self, df, loc, scale):
        super().__init__(validate_args=False)
        df = torch.as_tensor(df)
        if loc.ndim == 1:
            if scale.ndim != 2:
                raise ValueError("Invalid scale for single component MultivariateT")
            if df.ndim != 0:
                raise ValueError("Invalid df for single component MultivariateT")
            loc = loc[None, :]
            scale = scale[None, :]
            df = df[None]
        self.df = df
        self.loc = loc
        # self.scale = scale
        try:
            self.prec_chol = torch.linalg.cholesky(torch.linalg.inv(scale))
        except:
            self.prec_chol = torch.linalg.cholesky(
                dists.transforms.PositiveDefiniteTransform()(
                    torch.linalg.inv(scale)
                    ))
        self.logdet = torch.linalg.slogdet(scale)[1]
        self.dim = loc.shape[-1]

    def log_prob(self, value, keepdims=False):
        if value.ndim == 1:
            value = value[None, None, 1]
        elif value.ndim == 2:
            value = value[:, None, :]
        dev = value - self.loc[None, :, :] #  -> (N, K, D)
        maha = torch.square(torch.einsum("nki,kij->nkj", dev, self.prec_chol)).sum(-1) # (N, K)

        t = 0.5 * (self.df + self.dim) # (K)
        A = torch.special.gammaln(t) # (K)
        B = torch.special.gammaln(0.5 * self.df) # (K)
        C = self.dim/2. * torch.log(self.df * torch.pi) # (K)
        D = 0.5 * self.logdet # (K)
        E = -t * torch.log(1 + (1./self.df) * maha) # (K) * ((K)*(N,K)) -> N,K
        if keepdims:
            return A - B - C - D + E
        return (A - B - C - D + E).squeeze()


class MLP(nn.Module):
    """
    Define a simple fully connected MLP with ReLU activations.
    """
    def __init__(self,
                 in_features: int,
                 features: Sequence[int],
                 kernel_init=nn.init.kaiming_normal_,
                 bias_init=nn.init.zeros_,
                 activation=nn.ReLU,
                ):
        super().__init__()
        layers = []
        curr_feat = in_features
        for feat in features[:-1]:
            layers.extend([
                nn.Linear(curr_feat, feat),
                activation()
            ])
            curr_feat = feat
        layers.append(nn.Linear(curr_feat, features[-1]))
        # Initialization
        for layer in layers:
            if isinstance(layer, nn.Linear):
                kernel_init(layer.weight)
                bias_init(layer.bias)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ProtoDPMMNet(nn.Module):
    def __init__(
            self,
            data_dim: int,
            latent_dim: int,
            num_classes: int,
            recog_scale: float,
            hidden_layers: Sequence[int] = [128],
            margin : float = 0.,
            alpha: float = 1.,
            activation: nn.Module = nn.Tanh,
            cov_type: str = "full",
        ):
        super(ProtoDPMMNet, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.recog_logscale = nn.Parameter(
            torch.log(torch.tensor(recog_scale)),
            requires_grad=False
        )
        self.register_buffer("margin", torch.tensor(margin))
        # Prior
        self.register_buffer("alpha", torch.tensor(alpha))
        self.logkappa0 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.register_buffer("Sigmabar", 1. * torch.eye(latent_dim))
        self.register_buffer("mu0", torch.zeros(latent_dim))
        self.lognu0 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        # Components
        self.register_buffer("N", torch.zeros(self.num_classes))
        self.register_buffer("prior_logits", prior_logits)
        self.register_buffer("prior_means", prior_means)
        self.register_buffer("prior_covs", prior_covs)
        # Construct a recognition network to produce a Gaussian potential
        self.encoder = MLP(
            data_dim,
            hidden_layers + [latent_dim],
            activation=activation
        )

    @property
    def device(self):
        return self.recog_logscale.device

    @property
    def nu0(self):
        return torch.exp(self.lognu0)

    @property
    def kappa0(self):
        return torch.exp(self.logkappa0)

    @property
    def Sigma0(self):
        Sigma0 = self.Sigmabar * (self.nu0 + self.latent_dim + 1)
        return Sigma0

    def p_z(self):
        alpha = self.alpha
        return dists.Categorical(
            logits=torch.cat((self.prior_logits, torch.log(alpha[None])), 0)
        )

    def p_x_given_z(self):
        # Unwrap hierarchical prior params
        nus = torch.cat((self.N + self.nu0, self.nu0[None]), 0)
        kappas = torch.cat((self.N + self.kappa0, self.kappa0[None]), 0)
        factors = (kappas + 1) / (kappas * (nus - self.latent_dim + 1))
        mus = torch.cat((self.prior_means, self.mu0[None]), 0)
        covs = torch.cat((self.prior_covs, self.Sigma0[None]), 0)
        return MultivariateT(nus, mus, factors[:, None, None] * covs)

    def encode(self,
               data : Float[Tensor, "batch_size data_dim"],
               nosample = False,
              ):
        N = data.shape[0]
        D = self.latent_dim

        # Pass the data through the recognition network to get potential means
        surrogate_means, surrogate_logscales = self.encoder(data)
        if nosample:
            return surrogate_means
        # Scale means and cov
        if surrogate_logscales is None:
            surrogate_scales = torch.exp(self.recog_logscale) * torch.ones(N, D, device=self.device)      # (N, D)
        else:
            surrogate_scales = F.softplus(surrogate_logscales) + .1
        surr_cov = torch.diag_embed(surrogate_scales)                                  # (N, D, D)
        surr_dist = dists.MultivariateNormal(surrogate_means, covariance_matrix=surr_cov)
        samples = surr_dist.rsample()
        return samples

    def loglik(self, encodings : Float[Tensor, "batch_size latent_dim"],):
        xs = encodings
        p_z = self.p_z()
        p_x_given_z = self.p_x_given_z()
        lls = p_z.logits[None, :] + p_x_given_z.log_prob(xs[:, None, :])
        return lls

    def update_dpmm(self, encodings, labels):
        # Sufficient stats from encodings
        N = torch.stack([(labels == k).sum() for k in range(self.num_classes)])
        sumx = torch.stack([encodings[labels == k].sum(0) for k in range(self.num_classes)])
        sumxxT = torch.stack([
            torch.einsum("ni,nj->ij", encodings[labels==k], encodings[labels==k])
            for k in range(self.num_classes)])
        # TODO: Update hierarchical prior too
        nu_post = self.nu0 + N
        kappa_post = self.kappa0 + N
        mu_post = (self.kappa0 * self.mu0 + sumx) / kappa_post[:, None] # (D,) + (N, D) / (N, 1) -> (N,D)
        Sigma_post = sumxxT - kappa_post[:, None, None] * torch.einsum("ki,kj->kij", mu_post, mu_post) # (K,D,D) + (K,D,D) -> (K,D,D)
        Sigma_post += self.Sigma0[None, :, :]
        Sigma_post += self.kappa0 * torch.outer(self.mu0, self.mu0)[None, :, :]
        # Update hierarchical prior
        self.N = N
        self.prior_logits = torch.log(N)
        self.prior_means = mu_post
        self.prior_covs = Sigma_post

    def loss(self,
             data : Float[Tensor, "batch_size data_dim"],
             labels : Float[Tensor, "batch_size"] = None,
             nosample : bool = False,
             nodpmmupdate : bool = False,
             beta : float = 1.,
            ):
        """
        Compute the ELBO using reparameterization trick
        """
        encodings = self.encode(data, nosample=nosample)
        # DPMM Update
        if not nodpmmupdate:
            self.update_dpmm(encodings, labels)
        # Calculate loglikelihoods
        lls = self.loglik(encodings)
        # Add margin
        lls[torch.arange(lls.shape[0]), labels] -= self.margin

        ce = F.cross_entropy(lls, labels, reduction="mean")
        # Prior Sigma prob log IW(Sigma | nu0, Sigma0)
        # Regularization loss to make the covs more like the prior
        covs = self.prior_covs / (self.nu0 + self.N + self.latent_dim + 1)[:, None, None]
        covs_logdet = torch.linalg.slogdet(covs)[1]
        Sigma0_logdet = torch.linalg.slogdet(self.Sigma0)[1]
        Js = torch.linalg.inv(covs)
        logIW = - 0.5 * (self.nu0 + self.latent_dim + 1) * covs_logdet # (K)
        logIW += 0.5 * self.nu0 * Sigma0_logdet
        logIW -= 0.5 * torch.einsum("kij,jl->k", Js, self.Sigma0)
        logIW -= 0.5 * self.nu0 * self.latent_dim * torch.log(torch.tensor(2))
        logIW -= torch.special.multigammaln(0.5 * self.nu0, self.latent_dim)
        logIW = logIW.sum()#.mean()
        loss = ce + beta * (F.elu(-logIW)+1)
        return loss


class ProtoDPGMMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.latent_dim = int(self.args.latent_dim)
        self.alpha = float(self.args.alpha)
        self.recog_scale = float(self.args.recog_scale)
        self.margin = float(self.args.margin)
        self.hidden_layers = float(self.args.hidden_layers)
        self.protonet = None
        self.train_epochs = int(self.args.train_epochs)
        self.nosample = self.args.nosample
        self.beta = float(self.args.beta)
        self.lr = float(self.args.lr)
        self.weight_decay = float(self.args.weight_decay)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, from_feats: bool = False):
        if from_feats:
            features = data
        else:
            _, features = net(data, return_feature=True)
        encodings = self.protonet(features, nosample=True)
        lls = self.loglik(encodings)
        py_x = torch.softmax(lls, -1)
        id_probs = py_x[:,:-1]
        pred = id_probs.argmax(1)
        conf = - py_x[:, -1]
        return pred, conf

    def get_trainset_feats(self, net: nn.Module, loader):
        fname = 'vit-b-16-img1k-feats.pkl'
        fname = os.path.join(os.getcwd(), fname)
        if os.path.isfile(fname):
            dat = torch.load(fname)
            all_feats = dat['feats']
            all_labels = dat['labels']
        else:
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(loader,
                                  desc='Extract Train Features: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    logits, features = net(data, return_feature=True)
                    # TODO: Only store sufficient stats to speed this up
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            print(f'Saving ViT-B/16 feats and labels to: {fname}')
            torch.save({'feats': all_feats, 'labels': all_labels, 'preds': all_preds}, fname)
        all_feats = all_feats.numpy()
        if self.use_pca:
            self.pca = PCA(
                n_components=self.pca_dim, random_state=123).fit(all_feats)
            all_feats = self.pca.transform(all_feats)
        self.dim = all_feats.shape[1]
        return all_feats, all_labels

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            print('\nTraining encoder network...')
            print(f'alpha: {self.alpha}')
            feats, labels = self.get_trainset_feats(net, id_loader_dict['train'])
            self.protonet = ProtoDPMMNet(
                feats.shape[-1],
                self.latent_dim,
                self.num_classes,
                self.recog_scale,
                hidden_layers=self.hidden_layers,
                margin=self.margin,
                alpha=self.alpha,
                activation=nn.Tanh,
            )
            self.protonet = self.protonet.to("cuda")
            optimizer = torch.optim.Adam(
                self.protonet.parameters()
                self.lr,
                weight_decay=self.weight_decay)

            self.protonet.train()
            # Train loop
            for epoch in trange(self.train_epochs):
                loss = self.protonet.loss(
                    feats, labels, nosample=self.nosample, beta=self.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"{epoch}: {float(loss)}")

            self.protonet.eval()
            encs = self.protonet.encode(feats, nosample=True)
            self.protonet.update_dpmm(encs, labels)
            self.setup_flag = True
