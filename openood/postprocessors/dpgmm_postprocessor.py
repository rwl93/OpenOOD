import os
from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

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


class DPGMMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.covariance_type = self.args.covariance_type
        self.alpha = float(self.args.alpha)
        self.max_cls_conf = self.args.max_cls_conf
        self.sigma0_scale = self.args.sigma0_scale
        self.sigma0_sample_cov = self.args.sigma0_sample_cov
        self.kappa0 = self.args.kappa0
        self.use_pca = self.args.use_pca
        self.pca_dim = self.args.pca_dim
        self.hierarchical = self.args.hierarchical
        self.args_dict = self.config.postprocessor.postprocessor_sweep

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
            return multivariate_normal_logpdf(x, self.meanN, self.priorchol,
                                              self.covariance_type)
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

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\nEstimating cluster statistics from training set...')
            print(f'alpha: {self.alpha}')
            fname = 'vit-b-16-img1k-feats.pkl'
            fname = os.path.join(os.getcwd(), fname)
            if os.path.isfile(fname):
                dat = torch.load(os.path.join(os.getcwd(), 'vit-b-16-img1k-feats.pkl'))
                all_feats = dat['feats']
                all_labels = dat['labels']
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

            dim = all_feats.shape[1]
            self.nu0 = dim + 1
            # self.mu0 = np.zeros(all_feats.shape[1])
            self.mu0 = all_feats.mean(0)
            if self.covariance_type in ['full', 'tied'] and not self.hierarchical:
                if self.sigma0_sample_cov:
                    # compute class-conditional statistics
                    centered_data = all_feats - self.mu0
                    empcov = sklearn.covariance.EmpiricalCovariance(
                        assume_centered=False)
                    empcov.fit(centered_data)
                    # inverse of covariance
                    self.Sigma0 = empcov.covariance_# precision_
                else:
                    self.Sigma0 = np.eye(dim) * self.sigma0_scale
            elif self.covariance_type in ['diag', 'spherical']:
                # unitS0: self.Sigma0 = np.ones((dim,))
                if self.sigma0_sample_cov:
                    # compute class-conditional statistics
                    centered_data = all_feats - self.mu0
                    empcov = sklearn.covariance.EmpiricalCovariance(
                        assume_centered=False)
                    empcov.fit(centered_data)
                    # inverse of covariance
                    self.Sigma0 = np.diag(empcov.covariance_) # precision_
                    if self.covariance_type == 'spherical':
                        self.Sigma0 = self.Sigma0.mean() * np.ones((dim,))
                else:
                    self.Sigma0 = np.ones((dim,)) * self.sigma0_scale

            # compute class-conditional statistics
            self.meanN = []
            self.chol = []
            self.N = []
            all_SigmaN = []
            sample_means = []
            # Collect all class sufficient stats
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)]
                self.N.append(class_samples.shape[0])
                kappaN = self.kappa0 + self.N[-1]
                mk = class_samples.sum(0)
                sample_mean = class_samples.mean(0)
                sample_means.append(sample_mean)
                mN = (1. / kappaN) * (self.kappa0 * self.mu0 + mk)
                self.meanN.append(mN)
                if self.covariance_type == 'full':
                    eps = 1e-5 * np.eye(dim)
                    sumxx = class_samples.T.dot(class_samples)
                    # NOTE: Store cholesky of Sigma_N not factor * Sigma_N
                    # because sqrt(factor) * chol(Sigma) = chol(factor * Sigma)
                    mu0outer = np.outer(self.mu0, self.mu0) + eps
                    mNouter = np.outer(mN, mN)
                    SigmaN = self.Sigma0 + self.kappa0 * mu0outer
                    SigmaN += sumxx
                    SigmaN -= kappaN * mNouter
                    SigmaN += 4*eps
                    if not ispsd(SigmaN):
                        import pdb; pdb.set_trace()
                    all_SigmaN.append(SigmaN)
                elif self.covariance_type == 'tied':
                    # Store stats
                    sumxx = class_samples.T.dot(class_samples)
                    # FIXME(rwl93): This seems like a weird setting for muk
                    SigmaN = sumxx - np.outer(mk, sample_mean) - np.outer(sample_mean, mk)
                    temp = self.N[c] * (
                        np.outer(sample_mean, sample_mean) +
                        np.eye(dim) * 1e-5 # Avoid precision errors
                    )
                    SigmaN = SigmaN + temp
                    all_SigmaN.append(SigmaN)
                elif self.covariance_type in ['diag', 'spherical']:
                    sumxx = (class_samples ** 2).sum(0)
                    if self.covariance_type == 'spherical':
                        sumxx = sumxx.mean()
                    SigmaN = self.Sigma0 + self.kappa0 * (self.mu0 ** 2)
                    SigmaN += sumxx
                    SigmaN -= kappaN * (mN ** 2)
                    if self.covariance_type == 'spherical':
                        SigmaN = SigmaN.mean(-1)
                    all_SigmaN.append(SigmaN)
                else:
                    raise NotImplementedError
                if self.covariance_type != 'tied':
                    temp = compute_cov_cholesky(SigmaN, self.covariance_type) # pyright: ignore
                    self.chol.append(temp)
            # Compute tied chol mean N etc
            self.N = np.array(self.N) # shape[#classes]
            if self.covariance_type == 'tied':
                # Sigma = (1/nu_n - dim - 1) * (Psi0 + sum_K (
                    # sumxx_k - sumx_k @ mu_k - mu_k @ sumx_k + mu_k @ mu_k))
                factor = 1 / (self.nu0 + self.N.sum() - dim - 1)
                Psi0 = np.copy(self.Sigma0) # FIXME: What is a good setting for this?
                # Class-wise portion of Sigma calculated above
                # Sigma = list(Sigma_portion_k of shape D,D)
                Sigma = np.stack(all_SigmaN, 0).sum(0) # (K, D, D) -> (D,D)
                Sigma += Psi0
                Sigma_mean = Sigma * factor
                if self.hierarchical:
                    epsilon = 0.001
                    kappa = 0.001
                    sigmasq = 0.001
                    # Calculate Sigma0 given samples
                    nu_tick = epsilon + self.num_classes
                    kappa_tick = kappa + self.num_classes
                    mu_tick = (1 / kappa_tick) * np.stack(sample_means).sum(0)
                    muk_outers = np.stack([np.outer(s, s) for s in sample_means])
                    Psi_tick = sigmasq * np.eye(dim) + muk_outers.sum(0)
                    Psi_tick = Psi_tick - kappa_tick * np.outer(mu_tick, mu_tick)
                    # FIXME: This is problematic because nu_tick is not > dim+1
                    # Sigma0_mean = Psi_tick / (nu_tick - dim - 1)
                    Sigma0_mean = Psi_tick / (nu_tick - 1)
                else:
                    Sigma0_mean = self.Sigma0
                self.Sigma0 = Sigma0_mean + Sigma_mean
                print("Sigma0 for debugging")
                print(self.Sigma0)
                Sigma0_inv = scipy.linalg.inv(Sigma0_mean)
                Sigma_inv = scipy.linalg.inv(Sigma_mean)
                self.meanN = []
                self.chol = []
                for k in range(self.num_classes):
                    Sk = scipy.linalg.inv(Sigma0_inv + self.N[k] * Sigma_inv)
                    muk = Sigma0_inv @ self.mu0 + self.N[k] * Sigma_inv @ sample_means[k]
                    self.meanN.append(Sk @ muk)
                    SigmaN = Sk + Sigma_mean
                    self.chol.append(compute_cov_cholesky(SigmaN, self.covariance_type))
                print("final SigmaN for debugging")
                print(SigmaN)
            self.meanN = np.vstack(self.meanN)  # shape [#classes, feature dim]
            self.chol = np.stack(self.chol, 0)  # shape [#classes, feature dim, feature dim]

            # Sigma0 cholesky
            self.priorchol = compute_cov_cholesky(self.Sigma0, self.covariance_type)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
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
