import tensorly as tl
import torch
from torch.optim import SGD, Adam
from typing import List, Dict, Any

tl.set_backend("pytorch")

data = tl.datasets.load_covid19_serology()['tensor']

def R2_score(X_original, X_predicted):
    """Returns the R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because prediction can be
    arbitrarily worse).

    Parameters
    ----------
    X_original: array
        The original array
    X_predicted: array
        Thre predicted array.

    Returns
    -------
    float
    """
    return 1 - torch.norm(X_predicted - X_original) ** 2.0 / torch.norm(X_original) ** 2.0

finite_mask = torch.isfinite(data).float()
print(finite_mask.sum(), "finite values in data"
      f" ({100 * finite_mask.sum() / data.numel():.2f}%)")

print(f"Data shape: {data.shape}, dtype: {data.dtype}, device: {data.device}")

ranks = [8, 6, 8]


# HOOI Tucker

# create 70% missing values
mask = torch.rand(data.shape) < 0.3
finite_mask = finite_mask * mask.float()

import numpy as np
tucker_init = tl.decomposition.tucker(data, rank=ranks, init='random', n_iter_max=0, tol=1e-5, mask=finite_mask,
                                      random_state=np.random.RandomState(seed=42))

import torch.nn as nn

class TuckerParams(nn.Module):
    def __init__(self, tucker_init):
        super().__init__()
        self.core = nn.Parameter(tucker_init.core)
        self.factors = nn.ParameterList([nn.Parameter(f) for f in tucker_init.factors])

    def get_params(self):
        return [self.core] + list(self.factors)

tucker_params = TuckerParams(tucker_init)
params = list(tucker_params.parameters())

# print(params)

optimizer = SGD(params=params, lr=2e-6)

data = data
finite_mask = finite_mask
Q_b = []
for i in range(1000000):
    optimizer.zero_grad()
    reconstructed = tl.tucker_to_tensor((params[0], params[1:]))
    loss = torch.sum(((reconstructed - data) * finite_mask) ** 2)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        r2 = R2_score(data * finite_mask, reconstructed * finite_mask).item()
        print(f"Iter {i}: Loss={loss.item():.4f}, R2={r2:.4f}")
    param_norms = [torch.norm(p).item() ** 2 for p in params]
    Q_b.append(torch.tensor(param_norms).var().item())

import matplotlib.pyplot as plt

plt.plot(Q_b)
plt.xlabel('Iteration')
plt.savefig('a.png')