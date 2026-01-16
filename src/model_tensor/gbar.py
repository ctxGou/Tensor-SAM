import torch
import torch.nn as nn
from model_tensor import tlayer
from model_tensor.td import TensorizedModel
from model_tensor.tlayer import TensorLayer
from typing import List, Tuple, Type
import warnings

class gBAR(torch.optim.Optimizer):
    def __init__(self, model: nn.Module,
                 params, base_optimizer, alpha=0.2, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(gBAR, self).__init__(params, defaults)

        self.model = model
        # If the model does not have a tensorized layer, raise a warning
        # for degenerating to standard optimization.
        if not any(isinstance(layer, TensorLayer) for layer in model.modules()):
            raise warnings.warn(
                "The model does not contain any TensorLayer. "
                "gBAR will degenerate to standard optimization.",
                UserWarning
            )
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def update_alpha(self, current_T: int, T_max: int, value: str = "linear"):
        """
        Update the regularization coefficient alpha based on the current iteration.
        
        Args:
            current_T (int): The current iteration number.
            T_max (int): The maximum number of iterations.
            value (str): The type of update to perform, can be "linear", "cosine", or "constant".
        """
        if value == "linear":
            self.param_groups[0]["alpha"] = self.defaults["alpha"] * (1 - current_T / T_max)
        elif value == "cosine":
            self.param_groups[0]["alpha"] = self.defaults["alpha"] * (1/2 + 1/2 * torch.cos(torch.tensor(current_T / T_max * 3.141592653589793)))
        elif value == "constant":
            # Hehe! I love doing nothing.
            return
        else:
            raise ValueError("Invalid value for alpha update. Choose from 'linear', 'cosine', or 'constant'.")

    def step(self):
        for tlayer in self.model.modules():
            if not isinstance(tlayer, TensorLayer):
                continue

            # Collect parameters and their gradients
            params = [p for p in tlayer.parameters() if p.grad is not None]
            if not params:
                continue

            # Compute grad norm statistics
            grad_norms_sq_mean, _, grad_norms_sq = tlayer.factor_grad_norms_sq_statistics()

            # Hyperparameters
            scal = self.param_groups[0]["alpha"] * self.param_groups[0]["lr"]


            # Compute normalization denominator once
            sqrt_sum_grad_norms_sq = torch.sqrt(torch.sum(torch.stack(grad_norms_sq)))

            # Compute norm of each parameter (no_grad here isn't strictly required but avoids unnecessary graph building)
            with torch.no_grad():
                param_norms = torch.stack([
                    p.norm(p=2)
                    for p in params
                ])

                grad_devs = torch.stack(grad_norms_sq) - grad_norms_sq_mean
                scale_factors = 1 + scal * grad_devs / (sqrt_sum_grad_norms_sq * param_norms)

                # Apply in-place scaling
                torch._foreach_mul_(params, scale_factors.tolist())
        # Perform the base optimizer step
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups