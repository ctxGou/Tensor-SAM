import tensorly as tl
import torch
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

tucker_hooi = tl.decomposition.tucker(data, rank=ranks, init='random', n_iter_max=50000, tol=1e-5, mask=finite_mask)
print(f"Tucker HOOI R2 score: {R2_score(data * finite_mask, tl.tucker_to_tensor(tucker_hooi) * finite_mask):.4f}")