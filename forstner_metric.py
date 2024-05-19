import numpy as np
import torch
from tqdm import tqdm, trange
import gdown

# fname = 'imgnet1k-class-stats.npz'
fname = 'imgnet1k-class-stats1.npz'

# url = 'https://drive.google.com/file/d/1LvgtkWeS6ArdCoF3dh58ZtwDYbhWy6Et/view?usp=sharing'
# gdown.download(url=url, output=fname, fuzzy=True)

def ispsd_np(x):
    return not (np.linalg.eigvalsh(x) < 0.).any()

def ispsd(x):
    return not (torch.linalg.eigvalsh(x) < 0.).any()

DEVICE = 'cuda'

# Load data
with np.load(fname) as data:
    N = data['N']
    sumx = data['sumx']
    sumxxT = data['sumxxT']

K,D = sumx.shape
covs_np = (1. / (N - 1.))[:, None, None] * (
    sumxxT - (1. / N)[:, None, None] * np.einsum('ki,kj->kij', sumx, sumx))

eps = np.eye(D) * 1e-6
for i in trange(K, desc='Forcing covs to be PSD'):
    while not ispsd_np(covs_np[i]):
        covs_np[i] += eps
covs_inv = np.linalg.inv(covs_np)

abinv = covs_np[0] @ covs_inv[1]
lambdas = np.linalg.eigvals(abinv)
diff = np.sqrt(np.sum(np.log(lambdas) ** 2))
print(diff)
# Convert to torch
covs = torch.tensor(
        covs_np, dtype=torch.float32, requires_grad=False, device=DEVICE)

eps = torch.eye(D).to(DEVICE) * 1e-4
for i in trange(K, desc='Forcing covs to be PSD'):
    while not ispsd(covs[i]):
        covs[i] += eps
covs_inv = torch.linalg.inv(covs[1:]) # Don't need first cov NOTE: This is offset now!

abinv = covs[0] @ covs_inv[0]
lambdas = torch.linalg.eigvals(abinv)
diff = torch.sqrt(torch.sum(torch.log(lambdas) ** 2))
print(diff)
evals, evecs = torch.linalg.eigh(covs_inv[0])
sqrt_cov_inv = evecs * torch.sqrt(evals) @ torch.linalg.inv(evecs)
sqbasqb = sqrt_cov_inv @ covs[0] @ sqrt_cov_inv
lambdas = torch.linalg.eigvalsh(sqbasqb)
diff = torch.sqrt(torch.sum(torch.log(lambdas) ** 2 ))
print(diff)
# evals, evecs = torch.linalg.eigh(covs_inv)
# diag = torch.zeros_like(covs_inv)
# diag = torch.diagonal_scatter(diag, evals, dim1=-2, dim2=-1)
# diag = torch.sqrt(diag)
# sqrt_covs_inv = torch.einsum('kij,kjl,klm->kim', evecs, diag, torch.linalg.inv(evecs))
# # I might be able to do this faster
# # Get sqrt of the inverse of each cov (eigh)
# # Calculate with eigvalsh(B^-1/2 A B^-1/2)
# diffs = torch.zeros((K,K), device=DEVICE, requires_grad=False)
# with torch.no_grad():
#     for i in trange(K - 1, desc='Calculating Forstner diffs'):
#         abinv = torch.einsum('kij,jl,klm->kim', sqrt_covs_inv[i:], covs[i], sqrt_covs_inv[i:])
#         lambdas = torch.linalg.eigvalsh(abinv)
#         diff = torch.sqrt(torch.sum(torch.log(lambdas) ** 2, 1))
#         diffs[i, i+1:] = torch.real(diff)

diffs = torch.zeros((K,K), device=DEVICE, requires_grad=False)
with torch.no_grad():
    with tqdm(total=(K*(K-1))/2, desc='Calculating Forstner diffs') as pbar:
        try:
            for i in range(K - 1):
                if covs_inv[i:].ndim == 2:
                    abinv = covs[i] @ covs_inv[i]
                    num_k = 1
                else:
                    abinv = torch.einsum('ij,kjl->kil', covs[i], covs_inv[i:])
                    num_k = abinv.shape[0]
                lambdas = torch.linalg.eigvals(abinv)
                diff = torch.sqrt(torch.sum(torch.log(lambdas) ** 2, -1))
                diffs[i, i+1:] = torch.real(diff)
                pbar.update(num_k)
        except:
            import pdb; pdb.set_trace()
diffs = diffs.cpu().numpy()
np.save('forstner_diffs', diffs)
