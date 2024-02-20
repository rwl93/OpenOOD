from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

import scipy


MAX_SAMPLES = 10000


def compute_cov_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the covariances.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends on the `covariance_type`.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    cov_chol : array-like
        The cholesky decomposition of sample covariances of the current
        components. The shape depends on `covariance_type`.

    This is adapted from sklearn.mixture._gaussian_mixture:
        https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/mixture/_gaussian_mixture.py#L299
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )
    if covariance_type == 'full':
        if covariances.ndim == 2:
            if covariances.shape[0] != covariances.shape[1]:
                raise ValueError("Invalid covariance dimensions")
            cov_chol = scipy.linalg.cholesky(covariances, lower=True)
        elif covariances.ndim == 3:
            n_components, n_features, _ = covariances.shape
            cov_chol = np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                cov_chol[k] = scipy.linalg.cholesky(covariance, lower=True)
        else:
            raise ValueError("Invalid covariance dimension")
    elif covariance_type == 'tied':
        cov_chol = scipy.linalg.cholesky(covariances, lower=True)
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        cov_chol = np.sqrt(covariances)
    return cov_chol


def multivariate_normal_pdf(x, mean, shape, covariance_type):
    return np.exp(multivariate_normal_logpdf(x, mean, shape, covariance_type))

def multivariate_normal_logpdf(x, mean, chol, covariance_type):
    if covariance_type == 'full':
        return multivariate_normal_logpdf_full(x, mean, chol)
    elif covariance_type == 'tied':
        raise NotImplementedError
    else:
        return multivariate_normal_logpdf_diag(x, mean, chol, covariance_type)

def multivariate_normal_logpdf_diag(x, mean, chol, covariance_type):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # chol =
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if isinstance(chol, float) or chol.ndim == 0:
        chol = np.array([chol,])
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0) # (K,D)
        chol = np.expand_dims(chol, axis=0) # (K,D) or (K,1)
    if chol.ndim == 1:
        chol = np.expand_dims(chol, axis=-1)
    K = mean.shape[0]
    D = mean.shape[-1]
    # (K,D) or (K,1) => (K,)
    halflogdet = np.log(chol).sum(-1)
    if covariance_type == 'spherical' and chol.shape[1] == 1:
        halflogdet *= D

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            maha = 0.5 * (((x - mean[k]) / chol[k]) ** 2).sum(-1)
            # float - (K,) - (N,K) => (N,K)
            outk = -(D/2.) * np.log(2 * np.pi) - halflogdet[k] - maha
            out.append(outk)
        out = np.array(out).T
    else:
        x_KND = np.broadcast_to(x, (K, N, D))
        x_NKD = np.transpose(x_KND, (1, 0, 2))
        # float * (((N,K,D) - (K,D)) / (K, D or 1) ** 2) sumlast => (N,K)
        maha = 0.5 * (((x_NKD - mean) / chol) ** 2).sum(-1)
        # float - (K,) - (N,K) => (N,K)
        out = -(D/2.) * np.log(2 * np.pi) - halflogdet - maha
    return out.squeeze()

def multivariate_normal_logpdf_full(x, mean, scale_tril):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0)
        scale_tril = np.expand_dims(scale_tril, axis=0)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = np.log(np.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1) # (K,)

    scale_tril_torch = torch.tensor(scale_tril)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            #  (N,D)
            dev = x - mean[k]
            dev = torch.tensor(dev).T # (D,N)
            maha = torch.linalg.solve_triangular(scale_tril_torch[k], dev,  upper=False)
            maha = maha.numpy() # (D,N)
            maha = np.square(maha).sum(0) # (N,)
            maha *= 0.5
            outk = -(D/2.) * np.log(2 * np.pi) - halflogdet[k] - maha # (N,)
            out.append(outk)
        out = np.array(out).T
    else:
        x_KND = np.broadcast_to(x, (K, N, D))
        x_NKD = np.transpose(x_KND, (1, 0, 2))
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN_torch = torch.tensor(dev.transpose(1,2,0))
        maha = torch.linalg.solve_triangular(scale_tril_torch, dev_KDN_torch,  upper=False)
        maha = maha.numpy() # (K,D,N)
        # maha = scipy.linalg.solve_triangular(scale_tril, dev.T, lower=True).T
        if maha.ndim == 3:
            maha = np.square(maha).sum(1).T # (N,K)
        else:
            maha = np.square(maha).sum(0) # (N,)
        maha *= 0.5
        out = -(D/2.) * np.log(2 * np.pi) - halflogdet - maha # (N, K)
    return out.squeeze()


def multivariate_t_pdf(x, df, mean, chol, covariance_type):
    return np.exp(multivariate_t_logpdf(x, df, mean, chol, covariance_type))

def multivariate_t_logpdf(x, df, mean, chol, covariance_type):
    if covariance_type == 'full':
        return multivariate_t_logpdf_full(x, df,  mean, chol)
    elif covariance_type == 'tied':
        raise NotImplementedError
    else:
        return multivariate_t_logpdf_diag(x, df, mean, chol, covariance_type)

def multivariate_t_logpdf_diag(x, df, mean, chol, covariance_type):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # df = (K,)
    # mean = (K,D) or (D,)
    # chol =
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if isinstance(chol, float) or chol.ndim == 0:
        chol = np.array([chol,])
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0) # (K,D)
        chol = np.expand_dims(chol, axis=0) # (K,D) or (K,1)
    if chol.ndim == 1:
        chol = np.expand_dims(chol, axis=-1)
    K = mean.shape[0]
    D = mean.shape[-1]
    # (K,D) or (K,1) => (K,)
    halflogdet = np.log(chol).sum(-1)
    if covariance_type == 'spherical' and chol.shape[1] == 1:
        halflogdet *= D

    t = 0.5 * (df + D)
    A = scipy.special.gammaln(t)
    B = scipy.special.gammaln(0.5 * df)
    C = D/2. * np.log(df * np.pi)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            maha = (((x - mean[k]) / chol[k]) ** 2).sum(-1)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * np.log(maha)
            out.append(outk)
        out = np.array(out).T
    else:
        x_KND = np.broadcast_to(x, (K, N, D))
        x_NKD = np.transpose(x_KND, (1, 0, 2))
        # float * (((N,K,D) - (K,D)) / (K, D or 1) ** 2) sumlast => (N,K)
        maha = (((x_NKD - mean) / chol) ** 2).sum(-1)
        maha = 1. + (1. / df) * maha

        t = 0.5 * (df + D)
        A = scipy.special.gammaln(t)
        B = scipy.special.gammaln(0.5 * df)
        C = D/2. * np.log(df * np.pi)
        out = A - B - C - halflogdet - t * np.log(maha)
    return out.squeeze()

def multivariate_t_logpdf_full(x, df, mean, scale_tril):
    """Compute the logpdf of a multivariate t distribution.

    Parameters
    ----------
    x : array-like
        Batch of samples
    mean : array-like
    scale_tril : array-like
        Lower Cholesky of the covariance.
    """
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0)
        scale_tril = np.expand_dims(scale_tril, axis=0)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = np.log(np.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1) # (K,)
    scale_tril_torch = torch.tensor(scale_tril)
    t = 0.5 * (df + D)
    A = scipy.special.gammaln(t)
    B = scipy.special.gammaln(0.5 * df)
    C = D/2. * np.log(df * np.pi)
    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            dev = x - mean[k]
            dev = torch.tensor(dev).T # (D, N)
            maha = torch.linalg.solve_triangular(scale_tril_torch[k], dev, upper=False)
            maha = maha.numpy()
            maha = np.square(maha).sum(0) # (N,)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * np.log(maha)
            out.append(outk)
        out = np.array(out).T
    else:
        x_KND = np.broadcast_to(x, (K, N, D))
        x_NKD = np.transpose(x_KND, (1, 0, 2))
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN_torch = torch.tensor(dev.transpose(1,2,0))
        maha = torch.linalg.solve_triangular(scale_tril_torch, dev_KDN_torch,  upper=False)
        maha = maha.numpy() # (K,D,N)
        # maha = scipy.linalg.solve_triangular(scale_tril, dev.T, lower=True).T
        if maha.ndim == 3:
            maha = np.square(maha).sum(1).T # (N,K)
        else:
            maha = np.square(maha).sum(0) # (N,)
        maha = 1. + (1. / df) * maha
        out = A - B - C - halflogdet - t * np.log(maha)
    return out.squeeze()


class DPGMMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.covariance_type = self.config.postprocessor.postprocessor_args.covariance_type
        self.alpha = self.config.postprocessor.postprocessor_args.alpha

    def logpostpred(self, x):
        nu_N = self.nu0 + self.N
        dim = x.shape[1]
        df = nu_N - dim + 1
        kappa_N = self.kappa0 + self.N
        factor = (kappa_N + 1) / (kappa_N * df)
        if self.covariance_type == 'full':
            st = self.chol * np.sqrt(factor[:, np.newaxis, np.newaxis])
        elif self.covariance_type in ['diag', 'spherical']:
            if self.covariance_type=='diag':
                factor = factor[:, np.newaxis]
            st = self.chol * np.sqrt(factor)
        else:
            raise NotImplementedError
        return multivariate_t_logpdf(x, df, self.meanN, st, self.covariance_type)

    def logpriorpred(self, x):
        dim = x.shape[1]
        df = self.nu0 - dim + 1
        factor = (self.kappa0 + 1) / (self.kappa0 * df)
        st = np.sqrt(factor) * self.priorchol
        return multivariate_t_logpdf(x, df, self.mu0, st, self.covariance_type)

    def urn_coeff(self):
        coeffs = np.concatenate((self.N, [self.alpha]), axis=0)
        coeffs /= coeffs.sum()
        return coeffs

    def py_x(self, x):
        py = self.urn_coeff()
        logpost = self.logpostpred(x)
        logprior = self.logpriorpred(x)
        logprior = np.expand_dims(logprior, -1)
        logpx_y = np.concatenate((logpost, logprior), axis=-1)
        logpy_x = logpx_y + np.log(py)
        py_x = scipy.special.softmax(logpy_x, axis=-1)
        return py_x

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating cluster statistics from training set...')
            all_feats = []
            all_labels = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    logits, features = net(data, return_feature=True)
                    # TODO: Only store sufficient stats to speed this up
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)

            dim = all_feats.shape[1]
            self.Sigma0 = np.eye(dim)
            self.kappa0 = dim + 1
            self.nu0 = dim + 1
            self.mu0 = all_feats.numpy().mean(0)

            # compute class-conditional statistics
            self.meanN = []
            self.chol = []
            self.N = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data.numpy()
                self.N.append(class_samples.shape[0])
                kappaN = self.kappa0 + self.N[-1]
                sample_mean = class_samples.mean(0)
                mN = (1. / kappaN) * (self.kappa0 * self.mu0
                                      + self.N[-1] * sample_mean)
                self.meanN.append(mN)
                if self.covariance_type == 'full':
                    sumxx = class_samples.T.dot(class_samples)
                    # NOTE: Store cholesky of Sigma_N not factor * Sigma_N
                    # because sqrt(factor) * chol(Sigma) = chol(factor * Sigma)
                    mu0outer = np.outer(self.mu0, self.mu0)
                    SigmaN = self.Sigma0 + self.kappa0 * mu0outer
                    SigmaN += sumxx
                    SigmaN -= kappaN * np.outer(mN,  mN)
                elif self.covariance_type in ['diag', 'spherical']:
                    sumxx = (class_samples ** 2).sum(0)
                    if self.covariance_type == 'spherical':
                        sumxx = sumxx.mean()
                    SigmaN = self.Sigma0 + self.kappa0 * (self.mu0 ** 2)
                    SigmaN += sumxx
                    SigmaN -= kappaN * (mN ** 2)
                    if self.covariance_type == 'spherical':
                        SigmaN.mean(-1)
                else:
                    raise NotImplementedError
                temp = compute_cov_cholesky(SigmaN, self.covariance_type)
                self.chol.append(temp)
            self.meanN = np.vstack(self.meanN)  # shape [#classes, feature dim]
            self.chol = np.vstack(self.chol)  # shape [#classes, feature dim]
            self.N = np.concatenate(self.N, 0)

            # Sigma0 cholesky
            self.priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)

            # sanity check on train acc
            all_preds = self.py_x(all_feats)[:, :-1]
            all_preds = all_preds.argmax(1)
            train_acc = (all_preds == all_labels.numpy()).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, features = net(data, return_feature=True)
        py_x = self.py_x(features.numpy())
        id_probs = py_x[:,:-1]
        pred = id_probs.argmax(1)
        conf = py_x[:, -1]
        return pred, conf
