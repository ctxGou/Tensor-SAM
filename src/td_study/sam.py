import torch
import torch.nn as nn

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
            self.param_groups[0]["alpha"] = self.defaults["alpha"] * (1/2 + 1/2 * torch.cos(torch.tensor(current_T / T_max * 3.141592653589793)))
        elif value == "constant":
            # Hehe! I love doing nothing.
            return
        else:
            raise ValueError("Invalid value for alpha update. Choose from 'linear', 'cosine', or 'constant'.")

    def step(self):
        """
        ### Algorithm: gBAR (Generalized Balancing-Aware Regularization)**

        1.  **Initialize:** learning rate $\{\eta_t\}$, regularization coefficient $\{\alpha_t\}$, parameter cores $\{\mathcal{G}_1, \dots, \mathcal{G}_k\}$.
        2.      **for** $t = 0, \dots, T-1$ **do**
        3.                   Get stochastic gradients $\{\mathbf{g}_{\mathcal{G}_1}, \dots, \mathbf{g}_{\mathcal{G}_k}\}$.
        4.                   Calculate the squared gradient norms $\gamma_m \leftarrow \|\mathbf{g}_{\mathcal{G}_m}\|_F^2$ for all $m$.
        5.                   Calculate the mean of the squared gradient norms: $\bar{\gamma} \leftarrow \frac{1}{k}\sum_{m=1}^k \gamma_m$.
        6.                      **for** each core $m=1, \dots, k$ **do**
        7.                              // *Scale the core based on its gradient's deviation from the mean.*
        8.                              $\mathcal{G}_m \leftarrow \left(1 + \alpha_t \eta_t \left(1 - \frac{\gamma_m}{\bar{\gamma}}\right)\right) \mathcal{G}_m$
        9.                      **end for**
        10.         Optimizer update (e.g., SGD or Adam on the original gradients $\mathbf{g}_{\mathcal{G}_m}$).
        11.     **end for**
        """
        # The gradients are expected to be computed before calling this method.
        # Search for TensorLayer instances in the model.

        # Calculate the squared gradient norms on lora_A, lora_B, and lora_E
        params = list(self.model.parameters())
        factors = [p for p in params]
        grad_norms_sq = [
            torch.norm(factor.grad, p=2).pow(2) for factor in factors if factor.grad is not None
        ]
        factor_norms_sq = [
            torch.norm(factor, p=2).pow(2) for factor in factors
        ]
        grad_norms_sq_mean = torch.mean(torch.tensor(grad_norms_sq))
        grad_norms_sq_sum = torch.sum(torch.tensor(grad_norms_sq))
        
        # the_magical_scalars = [
        #     torch.sqrt(1 + self.param_groups[0]["alpha"] * self.param_groups[0]["lr"] * (grad_norm_sq - grad_norms_sq_mean) / (factor_norm_sq + 1e-6))
        #     for grad_norm_sq, factor_norm_sq in zip(grad_norms_sq, factor_norms_sq)
        # ]
        # the_magical_scalars = [
        #     torch.sqrt(1 + self.param_groups[0]["alpha"] * self.param_groups[0]["lr"] * (grad_norm_sq - grad_norms_sq_mean) / (factor_norm_sq * torch.sqrt(grad_norms_sq_sum)))
        #     for grad_norm_sq, factor_norm_sq in zip(grad_norms_sq, factor_norms_sq)
        # ]
        the_magical_scalars = [
            # (1 + self.param_groups[0]["alpha"] * self.param_groups[0]["lr"] * torch.tanh(grad_norm_sq/grad_norms_sq_mean - 1) )
            1 + self.param_groups[0]["alpha"] * self.param_groups[0]["lr"] * (grad_norm_sq - grad_norms_sq_mean) / (factor_norm_sq * torch.sqrt(grad_norms_sq_sum) + 1e-8)
            for grad_norm_sq, factor_norm_sq in zip(grad_norms_sq, factor_norms_sq)
        ]
        if self.current_T % 1000 == 0:
            print("The magical scalars > 1:",
                  [scalar.item()>1 for scalar in the_magical_scalars])
            print("factor norms squared:", 
                    [factor_norm_sq.item() for factor_norm_sq in factor_norms_sq])
            print("grad norms squared:", 
                    [grad_norm_sq.item() for grad_norm_sq in grad_norms_sq])
        # Update the parameters of the tensorized layer
        for factor, scalar in zip(factors, the_magical_scalars):
            if factor.grad is not None:
                factor.data.mul_(scalar)

        # Perform the base optimizer step
        self.base_optimizer.step()
        if "total_steps" in self.param_groups[0]:
            self.current_T += 1
            # Update alpha based on the current step
            self.update_alpha(self.current_T, self.param_groups[0]["total_steps"],
                              value=self.param_groups[0].get("alpha_scheduler", "linear"))

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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