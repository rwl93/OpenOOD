from typing import Sequence

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F

from jaxtyping import Float


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
            backbone: nn.Module,
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
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        try:
            self.data_dim = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

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

    def forward(self, data):
        feats = self.backbone(data)
        encodings = self.encode(data, nosample=True)
        return self.loglik(encodings)
