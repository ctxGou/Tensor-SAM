import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
import abc
from typing import List, Tuple



class TensorizedModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for all tensor decomposition models.
    
    Any new decomposition (e.g., Hierarchical Tucker) should inherit from this
    class and implement the `reconstruct` method.
    """
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape
        self.dimensionality = len(shape)

    @abc.abstractmethod
    def reconstruct(self) -> torch.Tensor:
        """
        Reconstructs the full tensor from the stored low-rank factors.
        This method must be implemented by all subclasses.
        """
        pass

    @abc.abstractmethod
    def compute_channel_fan(self) -> float:
        """
        Computes the size φ * Π(e_ij)/C for the tensorized model.
        C = C_{in} or C_{out} depending on your favorite convention.
        I divided by C because C is decided after instantializing neural layers.

        If you are interested, please refer to the paper:
        "A Unified Weight Initialization Paradigm for Tensorial Convolutional Neural Networks"
        by Yu, et al. ICML 2022.
        """
        pass

    def forward(self) -> torch.Tensor:
        """
        The forward call for a TensorizedModel is defined as reconstruction.
        This makes it seamlessly integrable with torch.nn.Module logic.
        """
        return self.reconstruct()

    def dof(self) -> int:
        """
        Computes the number of degrees of freedom in the tensorized model.
        This is the total number of parameters across all factors.
        """
        return sum(factor.numel() for factor in self.parameters())

    def factor_norms(self) -> List[torch.Tensor]:
        """
        Computes the Frobenius norms of each factor in the tensorized model.
        """
        return [torch.norm(factor, p='fro') for factor in self.parameters()]
    
    def factor_norms_sq_statistics(self) -> Tuple[float, float]:
        """
        Computes the mean and variance of the squared norms of each factor in the tensorized model.
        """
        norms = self.factor_norms()
        norms_sq = [torch.square(norm) for norm in norms]
        mean_sq = torch.mean(torch.stack(norms_sq))
        var_sq = torch.var(torch.stack(norms_sq))
        return mean_sq.item(), var_sq.item()

    def factor_grad_norms(self, squared=False) -> List[torch.Tensor]:
        """
        Computes the Frobenius norms (or squared norms) of the gradients of each factor
        in the tensorized model.
        
        Returns:
            List of scalar tensors, each representing the (squared) norm of a factor's gradient.
        """
        norms = []
        for factor in self.parameters():
            if factor.grad is None:
                continue
            grad = factor.grad
            if squared:
                norms.append(torch.sum(grad * grad))  # Equivalent to squared Frobenius norm
            else:
                norms.append(torch.norm(grad, p='fro'))
        return norms
    
    def factor_grad_norms_sq_statistics(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Computes statistics for the squared gradient norms of factors.
        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]:
            - mean_sq (Tensor): The mean of the squared gradient norms.
            - var_sq (Tensor): The variance of the squared gradient norms.
            - grad_norms_sq (List[Tensor]): A list of squared gradient norms for each factor.
        """
        grad_norms_sq = self.factor_grad_norms(squared=True)
        mean_sq = torch.mean(torch.stack(grad_norms_sq))
        var_sq = torch.var(torch.stack(grad_norms_sq))
        return mean_sq, var_sq, grad_norms_sq

# Implementation of Tensor Decompositions
class TensorTrain(TensorizedModel):
    """
    Tensor Train (TT) decomposition model.
    
    Args:
        shape (Tuple[int, ...]): The shape of the tensor to be decomposed.
        ranks (List[int]): The ranks for each dimension in the TT decomposition.
    """
    def __init__(self, shape: Tuple[int, ...], ranks: List[int]):
        super().__init__(shape)
        if len(ranks) == 1:
            ranks = [ranks[0]] * (len(shape) - 1)
        if len(shape) != len(ranks) + 1:
            raise ValueError("Ranks must be one less than the number of dimensions in shape.")
        self.ranks = ranks
        # shape of the factors will be (rank_i, dim_i, rank_{i+1}), first and last rank are 1
        ranks = [1] + ranks + [1]
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(ranks[i], shape[i], ranks[i + 1])) for i in range(len(shape))]
        )

    def reconstruct(self) -> torch.Tensor:
        return tl.tt_to_tensor(self.factors)  # tensorly for reconstruction

    def compute_channel_fan(self) -> float:
        # product of ranks
        return np.prod(self.ranks)

class TensorTrainMatrix(TensorizedModel):
    """
    Tensor Train decomposition with cores as 4-order tensors.
    
    Args:
        shape (Tuple[Tuple[int, int], ...]): The shape of the tensor to be decomposed.
        ranks (List[int]): The ranks for each dimension in the Tensor Train decomposition.
    """
    def __init__(self, shape: Tuple[Tuple[int, int], ...], ranks: List[int]):
        super().__init__(shape)
        if len(ranks) == 1:
            ranks = [ranks[0]] * (len(shape) - 1)
        if len(shape) != len(ranks) + 1:
            raise ValueError("Ranks must be one less than the number of dimensions in shape.")
        self.ranks = ranks
        # shape of the factors will be (rank_i, dim_i, rank_{i+1}), first and last rank are 1
        ranks = [1] + ranks + [1]
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(ranks[i], shape[i][0], shape[i][1], ranks[i + 1])) for i in range(len(shape))]
        )

    def reconstruct(self) -> torch.Tensor:
        return tl.tt_matrix_to_tensor(self.factors).contiguous()

    def compute_channel_fan(self) -> float:
        # product of ranks
        return np.prod(self.ranks)

class TensorRing(TensorizedModel):
    """
    Tensor Ring decomposition model.
    
    Args:
        shape (Tuple[int, ...]): The shape of the tensor to be decomposed.
        ranks (List[int]): The ranks for each dimension in the Tensor Ring decomposition.
    """
    def __init__(self, shape: Tuple[int, ...], ranks: List[int]):
        super().__init__(shape)
        if len(ranks) == 1:
            ranks = [ranks[0]] * len(shape)
        if len(shape) != len(ranks):
            raise ValueError("Ranks must match the number of dimensions in shape.")
        self.ranks = ranks
        # shape of the factors will be (rank_i, dim_i, rank_{i+1}), with last rank connected to first
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(ranks[i], shape[i], ranks[(i + 1) % len(shape)])) for i in range(len(shape))]
        )

    def reconstruct(self) -> torch.Tensor:
        return tl.tr_to_tensor(self.factors)  # tensorly for reconstruction

    def compute_channel_fan(self) -> float:
        # product of ranks
        return np.prod(self.ranks)

class CP(TensorizedModel):
    """
    CP decomposition model (CANDECOMP/PARAFAC).
    
    Args:
        shape (Tuple[int, ...]): The shape of the tensor to be decomposed.
        rank (int): The rank for the CP decomposition.
    """
    def __init__(self, shape: Tuple[int, ...], ranks: List[int]):
        super().__init__(shape)
        if len(ranks) != 1:
            raise ValueError("CP decomposition requires a single rank value.")
        self.rank = ranks[0]
        # factors will be of shape (dim_i, rank) for each dimension
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(shape[i], self.rank)) for i in range(len(shape))]
        )

    def reconstruct(self) -> torch.Tensor:
        return tl.cp_to_tensor((None, self.factors))  # tensorly for reconstruction

    def compute_channel_fan(self) -> float:
        return self.rank

class Tucker(TensorizedModel):
    """
    Tucker decomposition model.
    
    Args:
        shape (Tuple[int, ...]): The shape of the tensor to be decomposed.
        core_shape (Tuple[int, ...]): The shape of the core tensor.
    """
    def __init__(self, shape: Tuple[int, ...], ranks: List[int]):
        if len(ranks) == 1:
            ranks = [ranks[0]] * len(shape)
        core_shape = tuple(ranks)
        super().__init__(shape)
        if len(shape) != len(core_shape):
            raise ValueError("Core shape must match the number of dimensions in shape.")
        # first factor is the core tensor, followed by the factor matrices
        self.core_shape = core_shape
        self.factors = nn.ParameterList(
                [nn.Parameter(torch.randn(core_shape))] + \
                [nn.Parameter(torch.randn(shape[i], core_shape[i])) for i in range(len(shape))]
            )

    def reconstruct(self) -> torch.Tensor:
        # Note: tensorly's tucker_to_tensor produces a non-contiguous tensor
        return tl.tucker_to_tensor((self.factors[0], self.factors[1:])).contiguous()

    def compute_channel_fan(self) -> float:
        # product of core shape/all ranks
        return np.prod(self.core_shape)

if __name__ == "__main__":


    ttm_model = TensorTrainMatrix(((4, 4), (4, 4)), [2])
    print(ttm_model.reconstruct().shape)  # Should print: torch.Size([4, 4, 4, 4])

