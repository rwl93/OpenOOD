import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.evaluation_api import Evaluator, GibbsEvaluator

from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--outfile')
parser.add_argument('--arch',
                    default='resnet50',
                    choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet'])
parser.add_argument('--tvs-version', default=1, choices=[1, 2])
parser.add_argument('--ckpt-path', default=None)
parser.add_argument('--tvs-pretrained', action='store_true')
parser.add_argument('--postprocessor', default='msp')
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--noshuffle', default=False, action='store_true')
parser.add_argument('--gibbs', default=False, action='store_true')
parser.add_argument('--num-gibbs-samples', default=1000, type=int,
                    help="Number of times to Gibbs sample the parameters.")
args = parser.parse_args()

if not args.tvs_pretrained:
    assert args.ckpt_path is not None
    root = '/'.join(args.ckpt_path.split('/')[:-1])
else:
    root = os.path.join(
        ROOT_DIR, 'results',
        f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
    if not os.path.exists(root):
        os.makedirs(root)

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor
# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl')):
    with open(
            os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl'),
            'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

cov_type = postprocessor.covariance_type
if cov_type == 'full':
    chol = postprocessor.chol
    Sigma_post = np.einsum('kij,klj->kil', chol, chol)
    np.testing.assert_allclose(Sigma_post[0], chol[0] @ chol[0].T)
    Sigma = postprocessor.Sigmak
    Sigma0 = postprocessor.Sigma0
    priorchol = postprocessor.priorchol
    Sigma0_recon = priorchol @ priorchol.T
    np.testing.assert_allclose(Sigma0, Sigma0_recon)
    mu0 = postprocessor.mu0
    mu_post = postprocessor.meanN
    mu = postprocessor.mu
    sumx = postprocessor.sumx
    sumxx = postprocessor.sumxx
    N = postprocessor.N
    kappa0 = postprocessor.kappa0
    nu0 = postprocessor.nu0
    np.savez_compressed(args.outfile,
        Sigma_post=Sigma_post,
        Sigma=Sigma,
        Sigma0=Sigma0,
        mu0=mu0,
        mu_post=mu_post,
        mu=mu,
        sumx=sumx,
        sumxx=sumxx,
        N=N,
        kappa0=kappa0,
        nu0=nu0,
    )
elif cov_type == 'tied':
    chol = postprocessor.chol
    Sigma_post = np.einsum('kij,klj->kil', chol, chol)
    np.testing.assert_allclose(Sigma_post[0], chol[0] @ chol[0].T)
    Sigma = postprocessor.Sigma
    Sigma0 = postprocessor.Sigma0
    priorchol = postprocessor.priorchol
    Sigma0_post = priorchol @ priorchol.T
    mu0 = postprocessor.mu0
    mu_post = postprocessor.meanN
    mu = postprocessor.mu
    sumx = postprocessor.sumx
    sumxx = postprocessor.sumxx
    N = postprocessor.N
    nu0 = postprocessor.nu0
    print(mu.shape)
    np.savez_compressed(args.outfile,
        Sigma_post=Sigma_post,
        Sigma=Sigma,
        Sigma0=Sigma0,
        Sigma0_post=Sigma0_post,
        mu0=mu0,
        mu_post=mu_post,
        mu=mu,
        sumx=sumx,
        sumxx=sumxx,
        N=N,
        nu0=nu0,
    )
else:
    chol = postprocessor.chol
    Sigma_post = chol ** 2
    Sigma = postprocessor.Sigmak
    Sigma0 = postprocessor.Sigma0
    priorchol = postprocessor.priorchol
    Sigma0_recon = priorchol ** 2
    np.testing.assert_allclose(Sigma0, Sigma0_recon)
    mu0 = postprocessor.mu0
    mu_post = postprocessor.meanN
    mu = postprocessor.mu
    sumx = postprocessor.sumx
    sumxx = postprocessor.sumxx
    N = postprocessor.N
    kappa0 = postprocessor.kappa0
    nu0 = postprocessor.nu0
    np.savez_compressed(args.outfile,
        Sigma_post=Sigma_post,
        Sigma=Sigma,
        Sigma0=Sigma0,
        mu0=mu0,
        mu_post=mu_post,
        mu=mu,
        sumx=sumx,
        sumxx=sumxx,
        N=N,
        kappa0=kappa0,
        nu0=nu0,
    )
