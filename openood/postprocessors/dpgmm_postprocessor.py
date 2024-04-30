import os
from typing import Any
from copy import deepcopy

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

def ispsd(x):
    return (np.linalg.eigvals(x) >= 0.).all()

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
    if covariance_type in ['full', 'tied']:
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
        return multivariate_normal_logpdf_tied(x, mean, chol)
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

def multivariate_normal_logpdf_tied(x, mean, scale_tril):
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
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = np.log(np.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1) # ()/float
    scale_tril_torch = torch.tensor(scale_tril)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            #  (N,D)
            dev = x - mean[k]
            dev = torch.tensor(dev).T # (D,N)
            maha = torch.linalg.solve_triangular(scale_tril_torch, dev,  upper=False)
            maha = maha.numpy() # (D,N)
            maha = np.square(maha).sum(0) # (N,)
            maha *= 0.5
            outk = -(D/2.) * np.log(2 * np.pi) - halflogdet - maha # (N,)
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
        if maha.ndim == 3:
            maha = np.square(maha).sum(1).T # (N,K)
        else:
            maha = np.square(maha).sum(0) # (N,)
        maha *= 0.5
        out = -(D/2.) * np.log(2 * np.pi) - halflogdet - maha # (N, K)
    return out.squeeze()


def multivariate_normal_logpdf_torch(x, mean, chol, covariance_type):
    if covariance_type == 'full':
        return multivariate_normal_logpdf_full_torch(x, mean, chol)
    elif covariance_type == 'tied':
        return multivariate_normal_logpdf_tied_torch(x, mean, chol)
    else:
        raise NotImplementedError

def multivariate_normal_logpdf_full_torch(x, mean, scale_tril):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
        scale_tril = scale_tril.unsqueeze(0)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1) # (K,)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            #  (N,D)
            dev = x - mean[k]
            dev = dev.T # (D,N)
            maha = torch.linalg.solve_triangular(scale_tril[k], dev,  upper=False)
            maha = torch.square(maha).sum(0) # (N,)
            maha *= 0.5
            outk = -(D/2.) * torch.log(torch.tensor(2 * np.pi)) - halflogdet[k] - maha # (N,)
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D) dev_KDN = dev.permute(1,2,0)
        maha = torch.linalg.solve_triangular(scale_tril, dev_KDN,  upper=False)
        if maha.ndim == 3:
            maha = torch.square(maha).sum(1).T # (N,K)
        else:
            maha = torch.square(maha).sum(0) # (N,)
        maha *= 0.5
        out = -(D/2.) * torch.log(torch.tensor(2 * np.pi)) - halflogdet - maha # (N, K)
    return out.squeeze()

def multivariate_normal_logpdf_tied_torch(x, mean, scale_tril):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1) # ()/float

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            #  (N,D)
            dev = x - mean[k]
            dev = dev.T # (D,N)
            maha = torch.linalg.solve_triangular(scale_tril, dev,  upper=False)
            maha = torch.square(maha).sum(0) # (N,)
            maha *= 0.5
            outk = -(D/2.) * torch.log(torch.tensor(2 * np.pi)) - halflogdet - maha # (N,)
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N,1,D).repeat(1,K,1)
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN = dev.permute(1,2,0)
        maha = torch.linalg.solve_triangular(scale_tril, dev_KDN,  upper=False)
        if maha.ndim == 3:
            maha = torch.square(maha).sum(1).T # (N,K)
        else:
            maha = torch.square(maha).sum(0) # (N,)
        maha *= 0.5
        out = -(D/2.) * torch.log(torch.tensor(2 * np.pi)) - halflogdet - maha # (N, K)
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


def multivariate_t_logpdf_torch(x, df, mean, chol, covariance_type):
    if covariance_type == 'full':
        return multivariate_t_logpdf_full_torch(x, df,  mean, chol)
    elif covariance_type == 'tied':
        raise NotImplementedError
    else:
        return multivariate_t_logpdf_diag_torch(x, df, mean, chol, covariance_type)

def multivariate_t_logpdf_diag_torch(x, df, mean, chol, covariance_type):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # df = (K,)
    # mean = (K,D) or (D,)
    # chol =
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    df = torch.as_tensor(df)
    if isinstance(chol, float) or chol.ndim == 0:
        chol = torch.tensor([chol,], device=DEVICE)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0) # (K,D)
        chol = chol.unsqueeze(0) # (K,D) or (K,1)
    if chol.ndim == 1:
        chol = chol.unsqueeze(-1)
    K = mean.shape[0]
    D = mean.shape[-1]
    # (K,D) or (K,1) => (K,)
    halflogdet = torch.log(chol).sum(-1)
    if covariance_type == 'spherical' and chol.shape[1] == 1:
        halflogdet *= D

    t = 0.5 * (df + D)
    A = torch.special.gammaln(t)
    B = torch.special.gammaln(0.5 * df)
    C = D/2. * torch.log(df * torch.pi)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            maha = (((x - mean[k]) / chol[k]) ** 2).sum(-1)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * torch.log(maha)
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # float * (((N,K,D) - (K,D)) / (K, D or 1) ** 2) sumlast => (N,K)
        maha = (((x_NKD - mean) / chol) ** 2).sum(-1)
        maha = 1. + (1. / df) * maha

        t = 0.5 * (df + D)
        A = torch.special.gammaln(t)
        B = torch.special.gammaln(0.5 * df)
        C = D/2. * torch.log(df * torch.pi)
        out = A - B - C - halflogdet - t * torch.log(maha)
    return out.squeeze()

def multivariate_t_logpdf_full_torch(x, df, mean, scale_tril):
    """Compute the logpdf of a multivariate t distribution.

    Parameters
    ----------
    x : FloatTensor
        Batch of samples
    mean : FloatTensor
    scale_tril : FloatTensor
        Lower Cholesky of the covariance.
    df : int or Tensor
    """
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
        scale_tril = scale_tril.unsqueeze(0)
    df = torch.as_tensor(df)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1) # (K,)
    t = 0.5 * (df + D)
    A = torch.special.gammaln(t)
    B = torch.special.gammaln(0.5 * df)
    C = D/2. * torch.log(df * torch.pi)
    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            dev = x - mean[k]
            dev = dev.T # (D, N)
            maha = torch.linalg.solve_triangular(scale_tril[k], dev, upper=False)
            maha = torch.square(maha).sum(0) # (N,)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * maha.log()
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN = dev.permute(1,2,0)
        maha = torch.linalg.solve_triangular(scale_tril, dev_KDN,  upper=False)
        if maha.ndim == 3:
            maha = torch.square(maha).sum(1).T # (N,K)
        else:
            maha = torch.square(maha).sum(0) # (N,)
        maha = 1. + (1. / df) * maha
        out = A - B - C - halflogdet - t * torch.log(maha)
    return out.squeeze()


def calc_ssd(N, sumx, sumxx, mu, dim):
    """Calculate sum of squared deviations"""
    ssd = []
    for c in range(N.shape[0]):
        Sk = sumxx[c]
        mk = sumx[c]
        muk = mu[c]
        mkmuk = np.outer(mk, muk)
        temp = N[c] * (np.outer(muk, muk) + np.eye(dim) * 1e-5) # Avoid precision errors
        ssd.append(Sk - (mkmuk + mkmuk.T) + temp)
    return np.stack(ssd, 0)


class DPGMM(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.covariance_type = None
        self.alpha = float(self.args.alpha)
        self.max_cls_conf = self.args.max_cls_conf
        self.sigma0_scale = self.args.sigma0_scale
        self.sigma0_sample_cov = self.args.sigma0_sample_cov
        self.kappa0 = self.args.kappa0
        self.use_pca = self.args.use_pca
        if self.use_pca:
            print(f'PCA Dimension: {self.args.pca_dim}')
        self.pca_dim = self.args.pca_dim
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self._params = {"_chol": None, "_priorchol": None, "mu0": None, "meanN": None}

    def set_hyperparam(self, hyperparam: list):
        self.sigma0_scale = hyperparam[1]
        self.kappa0 = hyperparam[2]

    def get_hyperparam(self):
        return [self.sigma0_scale, self.kappa0]

    def logpostpred(self, x, use_torch=False):
        if self.covariance_type == 'tied':
            if use_torch:
                meanN  = torch.tensor(self.meanN, device=DEVICE)
                chol = torch.tensor(self.chol, device=DEVICE)
                return multivariate_normal_logpdf_torch(x, meanN, chol, self.covariance_type)
            return multivariate_normal_logpdf(x, self.meanN, self.chol, self.covariance_type)
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
        if use_torch:
            df = torch.tensor(df, device=DEVICE)
            meanN  = torch.tensor(self.meanN, device=DEVICE)
            st = torch.tensor(st, device=DEVICE)
            return multivariate_t_logpdf_torch(
                x, df, meanN, st, self.covariance_type)
        return multivariate_t_logpdf(x, df, self.meanN, st, self.covariance_type)

    def logpriorpred(self, x, use_torch=False):
        dim = x.shape[1]
        if self.covariance_type == 'tied':
            if use_torch:
                mu0  = torch.tensor(self.mu0, device=DEVICE)
                chol = torch.tensor(self.priorchol, device=DEVICE)
                return multivariate_normal_logpdf_torch(x, mu0, chol,
                                                        self.covariance_type)
            return multivariate_normal_logpdf(x, self.mu0, self.priorchol,
                                              self.covariance_type)

        df = self.nu0 - dim + 1
        factor = (self.kappa0 + 1) / (self.kappa0 * df)
        st = np.sqrt(factor) * self.priorchol
        if use_torch:
            df = torch.tensor(df, device=DEVICE)
            mu0  = torch.tensor(self.mu0, device=DEVICE)
            st = torch.tensor(st, device=DEVICE)
            return multivariate_t_logpdf_torch(x, df, mu0, st, self.covariance_type)
        return multivariate_t_logpdf(x, df, self.mu0, st, self.covariance_type)

    def urn_coeff(self):
        coeffs = np.concatenate((self.N, [self.alpha]), axis=0)
        coeffs /= coeffs.sum()
        return coeffs

    def py_x(self, x, use_torch=False):
        py = self.urn_coeff()
        logpost = self.logpostpred(x, use_torch=use_torch)
        logprior = self.logpriorpred(x, use_torch=use_torch)
        if use_torch:
            logprior = logprior.unsqueeze(-1)
            logpx_y = torch.cat((logpost, logprior), -1)
            logpy_x = logpx_y + torch.log(torch.tensor(py, device=DEVICE))
            py_x = torch.log_softmax(logpy_x, -1)
        else:
            logprior = np.expand_dims(logprior, -1)
            logpx_y = np.concatenate((logpost, logprior), axis=-1)
            logpy_x = logpx_y + np.log(py)
            py_x = scipy.special.log_softmax(logpy_x, axis=-1)
        return py_x

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, from_feats: bool = False):
        if from_feats:
            features = data
        else:
            _, features = net(data, return_feature=True)
            if self.use_pca:
                features = self.pca.transform(features.cpu().numpy())
                features = torch.tensor(features).to(DEVICE)
        py_x = self.py_x(features, use_torch=True)
        id_probs = py_x[:,:-1]
        pred = id_probs.argmax(1)
        if self.max_cls_conf:
            conf = id_probs.max(1)[0]
        else:
            conf = - py_x[:, -1]
        return pred, conf

    @torch.no_grad()
    def extract_features_batch(self, net, data):
        _, features = net(data, return_feature=True)
        if self.use_pca:
            features = self.pca.transform(features.cpu().numpy())
            features = torch.tensor(features).to(DEVICE)
        return features

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
            self.pca = PCA(n_components=self.pca_dim).fit(all_feats)
            all_feats = self.pca.transform(all_feats)
        self.dim = all_feats.shape[1]
        return all_feats, all_labels

    def set_priors(self, all_feats):
        # Setup priors
        self.nu0 = self.dim + 1
        self.mu0 = all_feats.mean(0)
        self.epsilon = self.dim + 0.001
        self.kappa = 0.001
        self.sigmasq = 0.001
        if self.covariance_type in ['full', 'tied']:
            if self.sigma0_sample_cov:
                # compute class-conditional statistics
                centered_data = all_feats - self.mu0
                empcov = sklearn.covariance.EmpiricalCovariance(
                    assume_centered=False)
                empcov.fit(centered_data)
                self.Sigma0 = empcov.covariance_
            else:
                self.Sigma0 = np.eye(self.dim) * self.sigma0_scale
        elif self.covariance_type in ['diag', 'spherical']:
            if self.sigma0_sample_cov:
                # compute class-conditional statistics
                centered_data = all_feats - self.mu0
                empcov = sklearn.covariance.EmpiricalCovariance(
                    assume_centered=False)
                empcov.fit(centered_data)
                # inverse of covariance
                self.Sigma0 = np.diag(empcov.covariance_) # precision_
                if self.covariance_type == 'spherical':
                    self.Sigma0 = self.Sigma0.mean() * np.ones((self.dim,))
            else:
                self.Sigma0 = np.ones((self.dim,)) * self.sigma0_scale
        self.Psi0 = np.copy(self.Sigma0)

    def get_trainset_stats(self, all_feats, all_labels):
        # compute class-conditional statistics
        Sk = []
        mk = []
        sample_means = []
        N = []
        # Collect all class sufficient stats
        for c in range(self.num_classes):
            class_samples = all_feats[all_labels.eq(c)]
            N.append(class_samples.shape[0])
            mk.append(class_samples.sum(0))
            sample_means.append(class_samples.mean(0))
            if self.covariance_type in ['full', 'tied']:
                Sk.append(class_samples.T.dot(class_samples))
            elif self.covariance_type == 'diag':
                Sk.append((class_samples ** 2).sum(0))
            elif self.covariance_type == 'spherical':
                Sk.append((class_samples ** 2).sum(0).mean())
        self.N = np.array(N)
        return mk, Sk, sample_means

    def initial_setup(self, net: nn.Module, loader):
        all_feats, all_labels = self.get_trainset_feats(net, loader)
        self.set_priors(all_feats)
        mk, Sk, sample_means = self.get_trainset_stats(all_feats, all_labels)
        return mk, Sk, sample_means

    @property
    def priorchol(self):
        if not hasattr(self, '_priorchol'):
            raise AttributeError("Must run setup before accessing priorchol")
        return self._priorchol

    @property
    def chol(self):
        if not hasattr(self, '_chol'):
            raise AttributeError("Must run setup before accessing chol")
        return self._chol
    def gibbs_warmup(self, iters: int, fname: str = 'gibbs_warmup.pkl'):
        if not hasattr(self, 'gibbs'):
            return
        logjoint = np.array([self.logjoint()]) # type: ignore
        print(f'Gibbs Warmup Starting Log Joint: {logjoint[-1]}')
        for itr in range(iters):
            # Calculate log joint likelihood
            # Gibbs step
            self.gibbs() # type: ignore
            logjoint = np.append(logjoint, self.logjoint(), axis=0) # type: ignore
            print(f'Gibbs Warmup Iter {itr} Log Joint: {logjoint[-1]}')
        # FIXME: Should I save out params every iteration?
        np.save(fname, logjoint)


class TiedDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'tied'

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])

            # Calculate ssd
            ssd = []
            for c in range(self.num_classes):
                Sk = sumxx[c]
                mk = sumx[c]
                muk = sample_means[c]
                mkmuk = np.outer(mk, muk)
                temp = self.N[c] * (np.outer(muk, muk) + np.eye(self.dim) * 1e-5) # Avoid precision errors
                ssd.append(Sk - (mkmuk + mkmuk.T) + temp)
            ssd = np.stack(ssd, 0)

            # Calculate Sigma
            factor = 1 / (self.nu0 + self.N.sum() - self.dim - 1)
            Psi0 = np.copy(self.Sigma0) # FIXME: What is a good setting for this?
            Sigma = ssd.sum(0) + Psi0 # (K, D, D) -> (D,D)
            Sigma_mean = Sigma * factor

            # Calculate Sk, meanN, SigmaN
            Sigma0_inv = np.linalg.inv(self.Sigma0)
            Sigma_inv = np.linalg.inv(Sigma_mean)

            self._chol = []
            self.meanN = []
            for k in range(self.num_classes):
                Sk = np.linalg.inv(Sigma0_inv + self.N[k] * Sigma_inv)
                muk = Sigma0_inv @ self.mu0 + self.N[k] * Sigma_inv @ sample_means[k]
                self.meanN.append(Sk @ muk)
                SigmaN = Sk + Sigma_mean
                self._chol.append(
                    compute_cov_cholesky(SigmaN, self.covariance_type))
            # shape [#classes, feature dim, feature dim]
            self._chol = np.stack(self._chol, 0)
            self.meanN = np.vstack(self.meanN)  # shape [#classes, feature dim]

            # Set self.Sigma0 for calculating logpriorpred
            Sigma0 = self.Sigma0 + Sigma_mean
            self._priorchol = compute_cov_cholesky(Sigma0, self.covariance_type)
            print("Sigma0 for debugging")
            print(Sigma0)
            print("Final SigmaN for debugging")
            print(SigmaN)
            self.setup_flag = True
        else:
            pass


class FullyBayesianTiedDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'tied'
        self._params.update({"Sigma": None, "mu0": None, "mu": None, "Sigma0": None})

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])
            self.sumx = sumx
            self.sumxx = sumxx
            self.sample_means = sample_means

            ssd = calc_ssd(self.N, sumx, sumxx, sample_means, self.dim)

            # Calculate Sigma
            factor = 1 / (self.nu0 + self.N.sum() - self.dim - 1)
            Sigma = ssd.sum(0) + self.Psi0 # (K, D, D) -> (D,D)
            self.Sigma = Sigma * factor

            # Calculate Sigma0 and mu0
            # NOTE: Set this to dim + epsilon to ensure nu > dim - 1
            nu_tick = self.epsilon + self.num_classes
            kappa_tick = self.kappa + self.num_classes
            mu_tick = (1 / kappa_tick) * np.stack(sample_means).sum(0)
            self.mu0 = mu_tick
            muk_outers = np.stack([np.outer(s, s) for s in sample_means])
            Psi_tick = self.sigmasq * np.eye(self.dim) + muk_outers.sum(0)
            Psi_tick = Psi_tick - kappa_tick * np.outer(mu_tick, mu_tick)
            self.Sigma0 = Psi_tick / (nu_tick - self.dim - 1)

            # Calculate Sk, meanN, SigmaN
            self.Sigma0_inv = np.linalg.inv(self.Sigma0)
            self.Sigma_inv = np.linalg.inv(self.Sigma)

            self._chol = []
            self.meanN = []
            self.mu = []
            for k in range(self.num_classes):
                Sk = np.linalg.inv(self.Sigma0_inv + self.N[k] * self.Sigma_inv)
                muk_hat = self.Sigma0_inv @ self.mu0 + self.N[k] * self.Sigma_inv @ sample_means[k]
                muk_hat = Sk @ muk_hat
                self.meanN.append(muk_hat)
                muk = np.random.multivariate_normal(muk_hat, Sk)
                self.mu.append(muk)
                SigmaN = Sk + self.Sigma
                self._chol.append(compute_cov_cholesky(SigmaN, self.covariance_type))
            # shape [#classes, feature dim, feature dim]
            self._chol = np.stack(self._chol, 0)
            self.meanN = np.vstack(self.meanN)  # shape [#classes, feature dim]
            self.mu = np.vstack(self.mu)  # shape [#classes, feature dim]

            # Set _priorchol for calculating logpriorpred
            print("Sigma0 for debugging")
            print(self.Sigma0 + self.Sigma)
            self._priorchol = compute_cov_cholesky(self.Sigma0 + self.Sigma,
                                                   self.covariance_type)

            print("Final SigmaN for debugging")
            print(SigmaN)
            self.setup_flag = True
        else:
            pass

    def sample_mu(self):
        """Sample mu for each class

        Sample from :math:`\mathcal{N}(\hat{\mu}_k, \hat{S}_k)`
        where:
        .. math::
            \hat{S}_k &= (\Sigma_0^{-1} + N_k \Sigma^{-1})^{-1} \\
            \hat{\mu}_k &= \hat{S}_k (\Sigma_0^{-1} \mu_0 + N_k \Sigma^{-1} \bar{x})^{-1}

        Note
        ----
        We need to do this despite it being integrated out in the
        posterior predictive because the other Gibbs samplers use this
        quantity.
        """
        # NOTE: This mu_k is a sample from the normal dist. The meanN used for
        # calculating the posterior predictive is the mean of this normal dist.
        mu = []
        meanN = []
        for c in trange(self.num_classes,
                        desc="Sampling Class Means",
                        position=1,
                        leave=False,
                        colour='red',
                        ):
            Sk = np.linalg.inv(self.Sigma0_inv + self.N[c] * self.Sigma_inv)
            muk_hat = self.Sigma0_inv @ self.mu0 + self.N[c] * self.Sigma_inv @ self.sample_means[c]
            muk_hat = Sk @ muk_hat
            meanN.append(muk_hat)
            muk = np.random.multivariate_normal(muk_hat, Sk)
            mu.append(muk)
        self.meanN = np.vstack(meanN) # This is used for calculating logpostpred
        self.mu = np.vstack(mu) # This is used for sampling Sigma, Sigma0, mu0

    def sample_Sigma(self):
        """Sample Sigma

        Sample from :math:`\mathrm{IW}\left(\nu_0 + N, \Psi_0 + ssd \right)`
        where:
        .. math::
            ssd &= \sum_{n=1}^N (x_n - \mu_{z_n})(x_n - \mu_{z_n})^\top \\
                &= \sum_{k=1}^K S_k - m_k \mu_k^\top - \mu_k m_k^\top + N_k \mu_k\mu_k^\top
        """
        nuN = self.nu0 + self.N.sum()
        ssd = calc_ssd(self.N, self.sumx, self.sumxx, self.mu, self.dim)
        ssd = ssd.sum(0)
        shape = self.Psi0 + ssd
        self.Sigma = scipy.stats.invwishart(nuN, shape).rvs()
        self.Sigma_inv = np.linalg.inv(self.Sigma)

        # Update class-wise choleskys
        chol = []
        # for k in range(self.num_classes):
        for k in trange(self.num_classes,
                        desc="Sampling Class Sigmas",
                        position=1,
                        leave=False,
                        colour='blue',
                        ):
            Sk = np.linalg.inv(self.Sigma0_inv + self.N[k] * self.Sigma_inv)
            Sigmak = Sk + self.Sigma
            chol.append(compute_cov_cholesky(Sigmak, self.covariance_type))
        self._chol = np.stack(chol, 0)

    def sample_hyperpriors(self):
        with tqdm(total=5, desc="Sampling hyperpriors", position=1,
                  leave=False, colour='cyan') as pbar:
            nutick = self.epsilon + self.num_classes
            kappatick = self.kappa + self.num_classes
            mutick = self.mu.sum(0) / kappatick
            mukmuk = np.stack([np.outer(m, m) for m in self.mu], 0)
            Psitick = self.sigmasq * np.eye(self.dim) + mukmuk.sum(0) \
                - kappatick * np.outer(mutick, mutick)
            pbar.update(1)
            self.Sigma0 = scipy.stats.invwishart(nutick, Psitick).rvs()
            pbar.update(1)
            self.mu0 = np.random.multivariate_normal(mutick, self.Sigma0 / kappatick)
            pbar.update(1)
            self.Sigma0_inv = np.linalg.inv(self.Sigma0)
            pbar.update(1)
            # Update stored chol(Sigma0+Sigma) for fast inference
            self._priorchol = compute_cov_cholesky(
                self.Sigma0 + self.Sigma, self.covariance_type)
            pbar.update(1)

    def gibbs(self):
        self.sample_mu()
        self.sample_Sigma()
        self.sample_hyperpriors()


class FullDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'full'

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])

            # Calculate class means
            kappaN = self.kappa0 + self.N
            self.meanN = (self.kappa0 * self.mu0 + sumx) / kappaN[:,np.newaxis]

            # Calculate class Sigmas
            eps = 1e-5 * np.eye(self.dim)
            self._chol = []
            for c in range(self.num_classes):
                # NOTE: Store cholesky of Sigma_N not factor * Sigma_N
                # because sqrt(factor) * chol(Sigma) = chol(factor * Sigma)
                mu0outer = np.outer(self.mu0, self.mu0) + eps
                mNouter = np.outer(self.meanN[c], self.meanN[c])
                SigmaN = self.Sigma0 + self.kappa0 * mu0outer
                SigmaN += sumxx[c]
                SigmaN -= kappaN[c] * mNouter
                SigmaN += 4*eps
                if not ispsd(SigmaN):
                    import pdb; pdb.set_trace()
                chol = compute_cov_cholesky(SigmaN, self.covariance_type)
                self._chol.append(chol)
            # shape [#classes, feature dim, feature dim]
            self._chol = np.stack(self._chol, 0)
            self._priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)
            self.setup_flag = True
        else:
            pass


class HierarchicalDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'full'
        # NOTE: mu and Sigmak are used as solely for updating the hyperprior
        # quantities. They are not used in log prior or posterior predictive
        # calculations because they are integrated over.
        self._params.update({"Sigma0": None, "kappa0": None, "nu0": None,
            "mu": None, "Sigmak": None})
        self.num_mh_steps = 20
        self.nu0_prop_scale = 10
        self.nu0_fixed = self.args.nu0_fixed
        if self.nu0_fixed:
            self.nu0 = float(self.args.nu0)
        self.gibbs_warmup = self.args.gibbs_warmup
        self.gibbs_warmup_iters = self.args.gibbs_warmup_iters

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])
            # self.nu0 = 40. # Hard set nu0 prior
            self.sumx = sumx
            self.sumxx = sumxx
            self.sample_means = sample_means

            # STEP 1: Estimate class-wise means and covs from sample stats and
            # uninformative priors as their respective means
            kappaN = self.kappa0 + self.N
            self.mu = (self.kappa0 * self.mu0 + sumx) / kappaN[:, None]
            nuN = self.nu0 + self.N
            eps = 1e-5 * np.eye(self.dim)
            mu0outer = np.outer(self.mu0, self.mu0) + eps
            Sigma_post = self.Sigma0 + self.kappa0 * mu0outer
            Sigma_post = Sigma_post[None, :, :] + self.sumxx
            Sigma_post -= kappaN[:, None, None] * np.einsum(
                'ki,kj->kij', self.mu, self.mu)
            # FIXME: Is there a better way to do this?
            Sigma_post += 1e-4 * np.eye(self.dim)
            if (nuN <= self.dim + 1.).any():
                raise ValueError("Invalid nuN value. Must be > dim + 1")
            self.Sigmak = Sigma_post / (nuN - self.dim - 1.)[:, None, None]

            # STEP 2: Update priors: Sigma0/_priorchol, mu0, kappa0, nu0 as
            # their respective means and the MLE for nu0
            # Sigma0 := mean( W(K*nu0 + D + 1, inv(sum(prec_k)) ) )
            #         = (K*nu0 + D + 1) * inv(sum(prec_k))
            K = self.N.shape[0]
            Js = np.linalg.inv(self.Sigmak)
            Js_sum = Js.sum(0)
            Js_sum_inv = np.linalg.inv(Js_sum)
            factor = K * self.nu0 + self.dim + 1.
            # NOTE: This factor is going to be very large:
            # 1000 * 768 + 768 + 1 = 768769
            print(f'Sigma0 Wishart df = {factor}')
            self.Sigma0 = factor * Js_sum_inv
            self._priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)
            # mu0 := mean(N(inv(J)h, inv(J))) = inv(J)h
            # J := kappa0 * sum_k(prec_k)
            # h := kappa0 * sum_k(prec_k @ mu_k)
            J_inv_mu0 = Js_sum_inv / self.kappa0 # (D,D)
            h_mu0 = self.kappa0 * np.einsum('kij,kj->i', Js, self.mu) # (D,)
            self.mu0 = J_inv_mu0 @ h_mu0
            # kappa0 := mean(Ga(DK/2 + 1, 1/2 sum((muk-mu0)^T inv(Sigmak) (muk-mu0)) ) )
            #         = (DK/2 + 1) /  (1/2 sum((muk-mu0)^T inv(Sigmak) (muk-mu0)))
            alpha = (self.dim * K * 0.5) + 1.
            diffs = self.mu - self.mu0
            beta = 0.5 * np.einsum('kij,ki,kj->', Js, diffs, diffs)
            self.kappa0 = alpha / beta
            print(f'kappa0: {self.kappa0}')
            # nu0: MLE of IW (Sigma_k | nu0, Sigma0)
            # Output plot data of Loglikelihood vs MLE nu0
            if not self.nu0_fixed:
                # MLE Sigma0 @ nu0 = inverse((1 / nu0) * mean(cluster precision))
                print("Running MLE for nu0 and Sigma0 estimates")
                Jbar_inv = Js_sum_inv * K
                def calc_loglik(nu):
                    return -scipy.stats.invwishart.logpdf(
                        self.Sigmak.transpose(1,2,0),
                        df=nu, scale=nu * Jbar_inv).sum()
                nu0s = np.linspace(self.dim, 1500, 100)
                logliks = []
                for nu in tqdm(nu0s,
                               desc='Calculate Loglikelihoods of IW(Sigma_k | nu, Sigma)',
                               position=0,
                               leave=False,
                              ):
                    logliks.append(calc_loglik(nu))
                fname = 'DPGMMHierarchical-nu0-Sigma0-likelihoods.pkl'
                fname = os.path.join(os.getcwd(), fname)
                if os.path.isfile(fname):
                    print(f'Moving old files to backup: {fname+".bkp"}')
                    os.rename(fname, fname+'.bkp')
                torch.save({
                    'nu0': nu0s,
                    'loglik': logliks,
                }, fname)
                res = scipy.optimize.minimize_scalar(calc_loglik,
                    bounds=(float(self.dim), 1200.))
                self.nu0 = res.x
                print(f'nu0 optimization result: {res}')

            # STEP 3: Update prior/posterior predictive stats
            kappa_post = self.kappa0 + self.N
            self.meanN = (self.kappa0 * self.mu0 + sumx) / kappa_post[:, None]
            eps = 1e-5 * np.eye(self.dim)
            mu0outer = np.outer(self.mu0, self.mu0) + eps
            Sigma_post = self.Sigma0 + self.kappa0 * mu0outer
            Sigma_post = Sigma_post[None, :, :] + self.sumxx
            Sigma_post -= kappa_post[:, None, None] * np.einsum(
                'ki,kj->kij', self.mu, self.mu)
            Sigma_post += 1e-4 * np.eye(self.dim)
            self._chol = compute_cov_cholesky(Sigma_post, self.covariance_type)
            # Perform
            self.gibbs_warmup(self.gibbs_warmup_iters,
                              fname='hierarchical_gibbs_warmup.pkl')
            self.setup_flag = True
        else:
            pass

    def sample_cluster_params(self):
        """Sample mu and Sigma for each class

        Note
        ----
        We need to do this despite it being integrated out in the
        posterior predictive because the other Gibbs samplers use this
        quantity.
        """
        # NOTE: This mu_k is a sample from the normal dist. The meanN used for
        # calculating the posterior predictive is the mean of this normal dist.
        K = self.N.shape[0]
        nu_post = self.nu0 + self.N
        kappa_post = self.kappa0 + self.N
        mu_post =  (self.kappa0 * self.mu0 + self.sumx) / kappa_post[:, None]
        Sigma_post = self.Sigma0 + self.kappa0 * np.outer(self.mu0, self.mu0)
        Sigma_post = Sigma_post[np.newaxis, :, :] + self.sumxx
        Sigma_post -= kappa_post[:, None, None] * np.einsum('ki,kj->kij', mu_post, mu_post)
        # FIXME: Is there a better way to do this?
        Sigma_post += 1e-4 * np.eye(self.dim)
        covs = np.array([scipy.stats.invwishart(nu_post[i], Sigma_post[i]).rvs()
                         for i in range(K)])
        self.Sigmak = covs
        self.mu = np.vstack([
            np.random.multivariate_normal(mu_post[i], covs[i] / kappa_post[i])
            for i in range(K)])

    def sample_Sigma0(self, precisions):
        df = self.N.shape[0] * self.nu0 + self.dim + 1
        # Calculate sum of covs
        sum_prec = precisions.sum(0)
        loc = np.linalg.inv(sum_prec)
        self.Sigma0 = scipy.stats.wishart(df, loc).rvs()
        self._priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)

    def sample_mh_nu0(self, logdet_Sigma0, logdet_Sigmak):
        if self.nu0_fixed:
            return
        num_clusters = self.N.shape[0]
        # logdet_Sigma0 = 2 * np.log(np.diagonal(self._priorchol)).sum()
        # BUG! Must use Sigmak samples NOT Sigma post
        # logdet_Sigmak = 2 * np.log(np.diagonal(self._chol, axis1=1, axis2=2)).sum(-1)
        # FIXME: TODO: Move this to gibbs method and calculate precisions and logdet
        # from the cholesky to avoid duplicate computation
        # logdet_Sigmak = np.linalg.det(self.Sigmak)

        def _target(nu0):
            if nu0 <= 0.0: return - np.inf
            lp = -0.5 * nu0 * num_clusters * self.dim * np.log(2.)
            lp -= num_clusters * scipy.special.multigammaln(0.5 * nu0, self.dim)
            lp += 0.5 * nu0 * np.sum(logdet_Sigma0 - logdet_Sigmak)
            return lp

        nu0 = self.nu0
        curr_lp = _target(nu0)
        for _ in range(self.num_mh_steps):
            prop_nu0 = np.random.normal(nu0, self.nu0_prop_scale)
            prop_lp = _target(prop_nu0)
            if np.log(np.random.uniform()) < prop_lp - curr_lp:
                nu0 = prop_nu0
                curr_lp = prop_lp
        self.nu0 = nu0

    def sample_mu0(self, precisions):
        J_inv = np.linalg.inv(precisions.sum(0)) / self.kappa0
        h = self.kappa0 * np.einsum('kij,kj->i', precisions, self.mu)
        J_inv_h = J_inv @ h
        self.mu0 = np.random.multivariate_normal(J_inv_h, J_inv)

    def sample_kappa0(self, precisions):
        num_clusters = self.N.shape[0]
        Js = precisions
        diffs = self.mu - self.mu0
        alpha_post = 0.5 * self.dim * num_clusters + 1.0
        beta_post = 0.5 * np.einsum('kij,ki,kj->', Js, diffs, diffs)
        self.kappa0 = np.random.gamma(alpha_post, 1./beta_post)

    def update_cluster_stats(self):
        """Update the cluster statistics for calculating prior/post predictive.
        """
        K = self.N.shape[0]
        nu_post = self.nu0 + self.N
        kappa_post = self.kappa0 + self.N
        mu_post =  (self.kappa0 * self.mu0 + self.sumx) / kappa_post[:, None]
        self.meanN = mu_post
        Sigma_post = self.Sigma0 + self.kappa0 * np.outer(self.mu0, self.mu0)
        Sigma_post = Sigma_post[np.newaxis, :, :] + self.sumxx
        Sigma_post -= kappa_post[:, None, None] * np.einsum('ki,kj->kij', mu_post, mu_post)
        # FIXME: Is there a better way to do this?
        Sigma_post += 1e-4 * np.eye(self.dim)
        self._chol = compute_cov_cholesky(Sigma_post,
                                          covariance_type=self.covariance_type)

    def gibbs(self):
        self.sample_cluster_params()
        # Slow:
        # precisions = np.linalg.inv(self.Sigmak)
        Sigmak_chol = compute_cov_cholesky(self.Sigmak,
            covariance_type=self.covariance_type)
        inv_chol = np.array([
            scipy.linalg.solve_triangular(L, np.eye(self.dim), lower=True)
            for L in Sigmak_chol])
        precisions = np.einsum('ijk,ijl->ikl', inv_chol, inv_chol)
        self.sample_Sigma0(precisions)
        # Calculate log determinants for nu0 update
        logdet_Sigmak = 2 * np.log(np.diagonal(
            Sigmak_chol, axis1=1, axis2=2)).sum(-1)
        logdet_Sigma0 = 2 * np.log(np.diagonal(self._priorchol)).sum()
        self.sample_mh_nu0(logdet_Sigma0, logdet_Sigmak)
        self.sample_mu0(precisions)
        self.sample_kappa0(precisions)
        self.update_cluster_stats()

    def logjoint(self) -> float:
        """Compute the log joint probability of the data, latent variables, and params.
        """
        lp = 0.0

        K = self.N.shape[0]
        # Compute the prior probability of the cluster params
        lp += scipy.stats.invwishart.logpdf(
            self.Sigmak.transpose(1,2,0), df=self.nu0, scale=self.Sigma0).sum()
        for k in range(K):
            lp += scipy.stats.multivariate_normal.logpdf(
                self.mu[k],
                mean=self.mu0, cov=(1. / self.kappa0) * self.Sigmak[k])

        # Compute probability of cluster assignments
        # NOTE: Not updating probs. Just using urn coeffs
        # lp += Dirichlet(self.alpha / self.max_clusters * torch.ones(self.max_clusters)).log_prob(probs)
        probs = self.urn_coeff()
        lp += scipy.stats.dirichlet.logpdf(
            probs[:-1], (self.alpha / K) * np.ones((K,)))
        # uniq, cnts = np.unique(clusters, return_counts=True)
        # cluster_counts = np.zeros((self.N.shape[0],))
        # cluster_counts[uniq] = cnts
        lp += (np.log(probs[:-1]) * self.N).sum()
        # Compute likelihood for the data
        lls = - (0.5 * self.N.sum() * self.dim) * np.log(2. * np.pi)
        lls -= 0.5 * (self.N * np.linalg.slogdet(self.Sigmak)[1]).sum()
        sumsqdevs = self.sumxx - np.einsum('ki,kj->kij', self.sumx, self.mu) \
            - np.einsum('ki,kj->kij', self.mu, self.sumx) \
            + self.N[:, None, None] * np.einsum('ki,kj->kij', self.mu, self.mu)
        lls -= 0.5 * np.trace(
            np.einsum('kij,kjl->kil', sumsqdevs, np.linalg.inv(self.Sigmak)),
            axis1=-2, axis2=-1).sum()
        lp += lls
        return lp


class DiagDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'diag'

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])

            # Calculate class means
            kappaN = self.kappa0 + self.N
            self.meanN = (self.kappa0 * self.mu0 + sumx) / kappaN[:,np.newaxis]

            # Set class Sigmas
            self._chol = []
            for c in range(self.num_classes):
                SigmaN = self.Sigma0 + self.kappa0 * (self.mu0 ** 2)
                SigmaN += sumxx[c]
                SigmaN -= kappaN[c] * (self.meanN[c] ** 2)
                self._chol.append(compute_cov_cholesky(
                    SigmaN, self.covariance_type))
            self._chol = np.stack(self._chol, 0)
            self._priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)
            self.setup_flag = True
        else:
            pass


class SphericalDPGMMPostprocessor(DPGMM):
    def __init__(self, config):
        super().__init__(config)
        self.covariance_type = 'spherical'

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            sumx, sumxx, sample_means = self.initial_setup(net, id_loader_dict['train'])

            # Calculate class means
            kappaN = self.kappa0 + self.N
            self.meanN = (self.kappa0 * self.mu0 + sumx) / kappaN[:,np.newaxis]

            # Set class Sigmas
            self._chol = []
            for c in range(self.num_classes):
                SigmaN = self.Sigma0 + self.kappa0 * (self.mu0 ** 2)
                SigmaN += sumxx[c]
                SigmaN -= kappaN[c] * (self.meanN[c] ** 2)
                SigmaN = SigmaN.mean(-1)
                self._chol.append(compute_cov_cholesky(
                    SigmaN, self.covariance_type))
            self._chol = np.stack(self._chol, 0)
            self._priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)
            self.setup_flag = True
        else:
            pass
