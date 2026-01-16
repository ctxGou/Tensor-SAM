#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers_LoRA import LoRALayer 
from typing import Optional, List 


class Linear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, r)
            ) 
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def weight_norm_gradient_norm_statistics(self) -> List[float]:
        norm_sq = [self.lora_A.norm(p=2).item() ** 2,
                   self.lora_B.norm(p=2).item() ** 2,
                   self.lora_E.norm(p=2).item() ** 2]
        grad_norm_sq = [self.lora_A.grad.norm(p=2).item() ** 2,
                     self.lora_B.grad.norm(p=2).item() ** 2,
                     self.lora_E.grad.norm(p=2).item() ** 2]
        import numpy as np
        norm_sq_var = np.var(norm_sq)
        grad_norm_sq_var = np.var(grad_norm_sq)
        norm_sq_AvsMean = norm_sq[0] - np.mean(norm_sq)
        norm_sq_BvsMean = norm_sq[1] - np.mean(norm_sq)
        norm_sq_EvsMean = norm_sq[2] - np.mean(norm_sq)
        grad_norm_sq_AvsMean = grad_norm_sq[0] - np.mean(grad_norm_sq)
        grad_norm_sq_BvsMean = grad_norm_sq[1] - np.mean(grad_norm_sq)
        grad_norm_sq_EvsMean = grad_norm_sq[2] - np.mean(grad_norm_sq)
        return {
            'norm_sq_var': norm_sq_var,
            'grad_norm_sq_var': grad_norm_sq_var,
            'norm_sq_AvsMean': norm_sq_AvsMean,
            'norm_sq_BvsMean': norm_sq_BvsMean,
            'norm_sq_EvsMean': norm_sq_EvsMean,
            'grad_norm_sq_AvsMean': grad_norm_sq_AvsMean,
            'grad_norm_sq_BvsMean': grad_norm_sq_BvsMean,
            'grad_norm_sq_EvsMean': grad_norm_sq_EvsMean
        }

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A.T @ self.lora_E.T) @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)