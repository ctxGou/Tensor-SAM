import tensorly as tl
import torch
from torch.optim import SGD, Adam
from typing import List, Dict, Any
import numpy as np
import argparse
from sam import SAM, gBAR
import wandb

import torch.nn as nn
tl.set_backend("pytorch")

def get_args():
    parser = argparse.ArgumentParser(description="Tucker decomposition with SGD optimization")
    parser.add_argument('--data', type=str, default=None, help='Path to input data file (optional, uses default if not provided)')
    parser.add_argument('--num_iter', type=int, default=100000, help='Number of optimization iterations')
    parser.add_argument('--optimization', type=str, default='Adam', help='Optimization algorithm')
    parser.add_argument('--rho', type=float, default=0.0, help='Rho hyperparameter (not used by default)')
    parser.add_argument('--alpha', type=float, default=0.0, help='Alpha hyperparameter (not used by default)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    return args

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





class TuckerParams(nn.Module):
    def __init__(self, tucker_init):
        super().__init__()
        self.core = nn.Parameter(tucker_init.core.detach().clone())
        self.factors = nn.ParameterList([nn.Parameter(f.detach().clone()) for f in tucker_init.factors])

    def get_params(self):
        return [self.core] + list(self.factors)

    def perturb(self, seed=0, d=0):
        if d == 0:
            return

        torch.manual_seed(seed)
        direction_core = torch.randn(self.core.shape, device=self.core.device)
        direction_factors = [torch.randn(f.shape, device=f.device) for f in self.factors]
        full_direction = [direction_core] + direction_factors

        total_norm_sq = sum(torch.sum(p ** 2) for p in full_direction)
        total_norm = torch.sqrt(total_norm_sq)
        
        if total_norm == 0:
            return

        with torch.no_grad():
            # Update core
            self.core.add_((direction_core / total_norm) * d)
            # Update factors
            for i, factor_dir in enumerate(direction_factors):
                self.factors[i].add_((factor_dir / total_norm) * d)

def main():

    args = get_args()

    wandb.init(project='tensor completion test', config=vars(args))
    wandb.run.name = f"{args.data}_{args.seed}"

    if args.data == 'covid19':
        data = tl.datasets.load_covid19_serology()['tensor']
    finite_mask = torch.isfinite(data).float()
    print(finite_mask.sum(), "finite values in data"
        f" ({100 * finite_mask.sum() / data.numel():.2f}%)")

    print(f"Data shape: {data.shape}, dtype: {data.dtype}, device: {data.device}")

    ranks = [8, 6, 8]

    # create 70% missing values
    torch.manual_seed(args.seed)
    mask = torch.rand(data.shape) < 0.3
    finite_mask = finite_mask * mask.float()

    tucker_init = tl.decomposition.tucker(data, rank=ranks, init='random', n_iter_max=0, tol=1e-5, mask=finite_mask,
                                      random_state=np.random.RandomState(seed=args.seed))
    tucker_params = TuckerParams(tucker_init)
    params = list(tucker_params.parameters())

    # print(params)

    if args.optimization.lower() == 'sgd':
        optimizer = SGD(params=params, lr=2e-6)
    elif args.optimization.lower() == 'sam':
        optimizer = SAM(params=params, base_optimizer=SGD, rho=args.rho, lr=2e-6)
    elif args.optimization.lower() == 'bar':
        optimizer = gBAR(model=tucker_params, params=params, base_optimizer=SGD, 
                         alpha=args.alpha, lr=2e-6, total_steps=args.num_iter)
    elif args.optimization.lower() == 'adam':
        optimizer = Adam(params=params)
    elif args.optimization.lower() == 'adam-sam':
        optimizer = SAM(params=params, base_optimizer=Adam, rho=args.rho)
    elif args.optimization.lower() == 'adam-bar':
        optimizer = gBAR(model=tucker_params, params=params, base_optimizer=Adam, 
                         alpha=args.alpha, total_steps=args.num_iter)

    Q_b = []
    if args.optimization.lower() == 'hooi':
        tucker = tl.decomposition.tucker(data, rank=ranks, init='random', n_iter_max=50000, tol=1e-5, mask=finite_mask,
                        random_state=np.random.RandomState(seed=args.seed))
        tucker_params = TuckerParams(tucker)
        params = list(tucker_params.parameters())
        reconstructed = tl.tucker_to_tensor((params[0], params[1:]))
        r2 = R2_score(data * finite_mask, reconstructed * finite_mask).item()
        param_norms = [torch.norm(p).item() ** 2 for p in params]
        Q_b.append(torch.tensor(param_norms).var().item())
        print(f"R2={r2}")
        wandb.log({'test_r2': r2})
        wandb.log({'Q_b': Q_b[-1]})
    else:
        for i in range(args.num_iter):
            optimizer.zero_grad()
            reconstructed = tl.tucker_to_tensor((params[0], params[1:]))
            loss = torch.sum(((reconstructed - data) * finite_mask) ** 2)
            loss.backward()

            if args.optimization.lower() == 'sam' or args.optimization.lower() == 'adam-sam':
                optimizer.first_step(zero_grad=True)
                # second step
                loss_p = torch.sum(((tl.tucker_to_tensor((params[0], params[1:])) - data) * finite_mask) ** 2)
                loss_p.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            if i % 1000 == 0:
                r2 = R2_score(data * finite_mask, reconstructed * finite_mask).item()
                print(f"Iter {i}: Loss={loss.item():.4f}, R2={r2:.4f}")
            param_norms = [torch.norm(p).item() ** 2 for p in params]
            Q_b.append(torch.tensor(param_norms).var().item())
            wandb.log({'train_loss': loss.item()}, step=i)
            wandb.log({'test_r2': r2}, step=i)
            wandb.log({'Q_b': Q_b[-1]}, step=i)

    


    print("\n✅ Training finished. Starting perturbation analysis...")

    # 1. Define the range for the perturbation scale 'd'
    d_values = np.linspace(-3, 3, 101)  # 101 points for a smooth plot
    perturb_seed = 999  # A fixed seed for a consistent perturbation direction

    losses_vs_d = []
    r2_scores_vs_d = []

    # 2. Loop through each 'd' value to perturb the model and record metrics
    for d in d_values:
        # Apply perturbation from the final trained state
        tucker_params.perturb(seed=perturb_seed, d=d)
        
        with torch.no_grad():
            # Evaluate loss and R2 on the perturbed model
            perturbed_params = tucker_params.get_params()
            reconstructed_perturbed = tl.tucker_to_tensor((perturbed_params[0], perturbed_params[1:]))
            
            loss_perturbed = torch.sum(((reconstructed_perturbed - data) * finite_mask) ** 2)
            r2_perturbed = R2_score(data * finite_mask, reconstructed_perturbed * finite_mask)

            losses_vs_d.append(loss_perturbed.item())
            r2_scores_vs_d.append(r2_perturbed.item())

        # IMPORTANT: Revert the perturbation to reset the model for the next 'd'
        # This ensures each perturbation starts from the same trained state.
        tucker_params.perturb(seed=perturb_seed, d=-d)

    print("Perturbation analysis finished.")

    # 3. Convert lists to NumPy arrays
    losses_np = np.array(losses_vs_d)
    r2_np = np.array(r2_scores_vs_d)

    # 4. Save the arrays to uniquely named files for later plotting
    run_identifier = f"{args.optimization}_{args.seed}_{args.rho}_{args.alpha}"
    np.save(f"./npy/losses_vs_d_{run_identifier}.npy", losses_np)
    np.save(f"./npy/r2_vs_d_{run_identifier}.npy", r2_np)
    np.save(f"./npy/d_values_{run_identifier}.npy", d_values) # Also save d_values for the x-axis

    print(f"💾 Saved perturbation results for run: {run_identifier}")

if __name__ == '__main__':
    main()