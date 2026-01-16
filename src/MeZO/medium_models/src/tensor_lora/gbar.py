import torch
import torch.nn as nn
from typing import List, Tuple, Type
import warnings
from .layers_flora import Linear as FloraLinear

class gBAR(torch.optim.Optimizer):
    def __init__(self, model: nn.Module,
                 params, base_optimizer, alpha=0.2, total_steps=-1, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(gBAR, self).__init__(params, defaults)

        self.model = model
        # If the model does not have a tensorized layer, raise a warning
        # for degenerating to standard optimization.

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        if total_steps > 0:
            self.param_groups[0]["total_steps"] = total_steps
            self.current_T = 0


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
            self.param_groups[0]["alpha"] = self.defaults["alpha"] * (1 + torch.cos(torch.tensor(current_T / T_max * 3.141592653589793)))
        elif value == "constant":
            # Hehe! I love doing nothing.
            return
        else:
            raise ValueError("Invalid value for alpha update. Choose from 'linear', 'cosine', or 'constant'.")

    def step(self):
        # The gradients are expected to be computed before calling this method.
        # Search for TensorLayer instances in the model.
        for tlayer in self.model.modules():
            if isinstance(tlayer, FloraLinear):
                # Calculate the squared gradient norms on lora_A, lora_B, and lora_E
                grad_norms_sq = [
                    torch.norm(factor.grad, p=2).pow(2) for factor in tlayer.parameters() if factor.grad is not None
                ]
                grad_norms_sq_mean = torch.mean(torch.tensor(grad_norms_sq))
                sqrt_sum_grad_norms_sq = torch.sqrt(sum(grad_norms_sq))
                the_magical_scalars = [
                    (1 + self.param_groups[0]["alpha"] * self.param_groups[0]["lr"] * (grad_norm_sq - grad_norms_sq_mean) / (sqrt_sum_grad_norms_sq) / (1e-8 + torch.norm(factor.data, p=2)))
                    for grad_norm_sq, factor in zip(grad_norms_sq, tlayer.parameters())
                ]
                # Update the parameters of the tensorized layer
                for factor, scalar in zip(tlayer.parameters(), the_magical_scalars):
                    if factor.grad is not None:
                        factor.data.mul_(scalar)

        # Perform the base optimizer step
        self.base_optimizer.step()
        if "total_steps" in self.param_groups[0]:
            self.current_T += 1
            # Update alpha based on the current step
            self.update_alpha(self.current_T, self.param_groups[0]["total_steps"],
                              value=self.param_groups[0].get("alpha_scheduler", "linear"))

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