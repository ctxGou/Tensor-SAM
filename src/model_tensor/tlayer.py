from .td import TensorizedModel

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
import abc
from typing import List, Tuple


class TensorLayer(nn.Module, abc.ABC):
    """
    Abstract Base Class for tensorized layers.
    
    A TensorLayer holds a TensorizedModel, which represents its weights.
    The forward pass first reconstructs the weight tensor and then performs
    the layer's specific operation.
    """
    def __init__(self, weight_model: TensorizedModel, bias: bool = True):
        super().__init__()
        self.weight_model = weight_model
        
        # Bias is handled just like in a standard layer
        self.bias_param = nn.Parameter(torch.zeros(self.get_bias_shape())) if bias else None

    @abc.abstractmethod
    def get_bias_shape(self) -> Tuple[int, ...]:
        """Returns the appropriate shape for the bias term."""
        pass
        
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass logic for the layer."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight_model={self.weight_model.__class__.__name__}, ranks: {self.weight_model.ranks} " \
               f"bias={self.bias_param is not None}), " \
               f"parameters={self.weight_model.dof()} + {self.bias_param.numel() if self.bias_param is not None else 0})"

    def factor_norms_sq_statistics(self) -> Tuple[float, float]:
        """See TensorizedModel.factor_norms_sq_statistics."""
        return self.weight_model.factor_norms_sq_statistics()
    
    def factor_grad_norms_sq(self) -> List[torch.Tensor]:
        """See TensorizedModel.factor_grad_norms."""
        return self.weight_model.factor_grad_norms(squared=True)

    def factor_grad_norms_sq_statistics(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """See TensorizedModel.factor_grad_norms_sq_statistics."""
        return self.weight_model.factor_grad_norms_sq_statistics()

class TensorLinear(TensorLayer):
    """
    A tensorized linear layer.
    
    Args:
        weight_model (TensorizedModel): The model representing the weights of the layer.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        init_balance_mode (str): Initialization mode for balancing the tensor.
        fan_mode (str): Mode for determining the fan-in or fan-out.
        activation_scale (float): Scaling factor for the activation. !!! Use 0.5 for ReLU activation.
    """
    def __init__(self, weight_model: TensorizedModel, in_features: int, out_features: int, bias: bool = True, 
            init_balance_mode: str = 'equal_var', fan_mode: str = 'in', activation_scale: float = 0.5):
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(weight_model, bias)

        # check if prod(weight.shape) == in_features * out_features
        if torch.prod(torch.tensor(weight_model.shape)) != in_features * out_features:
            raise ValueError("The shape of the weight model does not match in_features and out_features.")


        if fan_mode == 'in':
            C = in_features
        elif fan_mode == 'out':
            C = out_features
        else:
            raise ValueError("fan_mode must be either 'in' or 'out'.")

        # Initialize the weight model
        if init_balance_mode == 'equal_var':
            channel_fan = weight_model.compute_channel_fan()
            denominator = np.power(activation_scale * C * channel_fan, 1.0 / len(weight_model.factors))
            var = 1.0 / denominator
            bound = np.sqrt(3.0 * var)
            for factor in weight_model.factors:
                factor.data.uniform_(-bound, bound)
        elif init_balance_mode == 'equal_norm':
            # equal expected norm initialization
            channel_fan = weight_model.compute_channel_fan()
            prod_size_factors = np.prod([float(np.prod(factor.shape)) for factor in weight_model.factors])
            denominator_full = np.power(activation_scale * C * channel_fan / prod_size_factors, 1.0 / len(weight_model.factors))
            for factor in weight_model.factors:
                var = 1.0 / denominator_full / np.prod(factor.shape)
                bound = np.sqrt(3.0 * var)
                factor.data.uniform_(-bound, bound)
        else:
            raise ValueError("init_balance_mode must be 'equal_variance'.")

        # Initialize the bias parameter
        if bias:
            self.bias_param.data.fill_(0.0)


    def get_bias_shape(self) -> Tuple[int, ...]:
        return (self.out_features,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_tensor = self.weight_model.reconstruct()
        # print(f"Weight tensor shape: {weight_tensor.shape}, expected shape: ({self.out_features}, {self.in_features})")
        weight_tensor = weight_tensor.view(self.out_features, self.in_features)
        output = F.linear(x, weight_tensor, self.bias_param)
        return output


class TensorConv2d(TensorLayer):
    """
    A tensorized 2D convolutional layer.

    Args:
        weight_model (TensorizedModel): The model representing the weights of the layer.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or Tuple[int, int]): Size of the convolving kernel.
        stride (int or Tuple[int, int]): Stride of the convolution.
        padding (int or Tuple[int, int]): Padding added to both sides of the input.
        bias (bool): Whether to include a bias term.
    """
    def __init__(self, weight_model: TensorizedModel, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True,
                 init_balance_mode: str = 'equal_var', fan_mode: str = 'in', activation_scale: float = 0.5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        super().__init__(weight_model, bias)

        # check if prod(weight.shape) == out_channels * in_channels * kernel_size * kernel_size
        if torch.prod(torch.tensor(weight_model.shape)) != out_channels * in_channels * kernel_size * kernel_size:
            raise ValueError("The shape of the weight model does not match in_channels, out_channels, and kernel_size.")
        # Initialize the weight model
        if fan_mode == 'in':
            C = in_channels
        elif fan_mode == 'out':
            C = out_channels
        else:
            raise ValueError("fan_mode must be either 'in' or 'out'.")

        # Initialize the weight model
        if init_balance_mode == 'equal_var':
            channel_fan = weight_model.compute_channel_fan()
            denominator = np.power(activation_scale * C * channel_fan * kernel_size * kernel_size, 1.0 / len(weight_model.factors))
            var = 1.0 / denominator
            bound = np.sqrt(3.0 * var)
            for factor in weight_model.factors:
                factor.data.uniform_(-bound, bound)
        elif init_balance_mode == 'equal_norm':
            # equal expected norm initialization
            channel_fan = weight_model.compute_channel_fan()
            prod_size_factors = np.prod([float(np.prod(factor.shape)) for factor in weight_model.factors])
            denominator_full = np.power(activation_scale * C * channel_fan * kernel_size * kernel_size / prod_size_factors, 1.0 / len(weight_model.factors))
            for factor in weight_model.factors:
                # print(f"Factor shape: {factor.shape}, prod: {np.prod(factor.shape)}")
                var = 1.0 / denominator_full / np.prod(factor.shape)
                bound = np.sqrt(3.0 * var)
                factor.data.uniform_(-bound, bound)
        else:
            raise ValueError("init_balance_mode must be 'equal_variance'.")

        # Initialize the bias parameter
        if bias:
            self.bias_param.data.fill_(0.0)


    def get_bias_shape(self) -> Tuple[int, ...]:
        return (self.out_channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_tensor = self.weight_model.reconstruct()
        # print(f"Weight tensor shape: {weight_tensor.shape}, expected shape: ({self.out_channels}, {self.in_channels}, {self.kernel_size}, {self.kernel_size})")
        weight_tensor = weight_tensor.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        output = F.conv2d(x, weight_tensor, self.bias_param, stride=self.stride, padding=self.padding)
        return output


if __name__ == "__main__":
    # Example usage
    from .td import TensorTrain, TensorRing, CP, Tucker, TensorTrainMatrix

    # # define a input and calculate its variance
    # torch.manual_seed(2)
    # input_tensor = torch.randn(50000, 1024)  # Example input tensor
    # input_variance = torch.var(input_tensor, dim=0).mean().item()
    # print(f"Input variance: {input_variance}")

    # def forward_test(the_layer: TensorLayer, input_tensor: torch.Tensor):
    #     print(f"Testing layer: {the_layer}")
    #     output_tensor = the_layer(input_tensor)
    #     print(f"Output tensor shape: {output_tensor.shape}")
    #     output_variance = torch.var(output_tensor, dim=0).mean().item()
    #     print(f"Output variance: {output_variance}")


    # # Test linear layer from 1024 -> 512

    # # original linear
    # linear_layer = nn.Linear(in_features=1024, out_features=512, bias=True)
    # # kaiming init with activation scale of 1
    # nn.init.kaiming_uniform_(linear_layer.weight, a=0, mode='fan_in', nonlinearity='linear')
    # output_tensor = linear_layer(input_tensor)
    # print(f"Output tensor shape (original linear): {output_tensor.shape}")
    # output_variance = torch.var(output_tensor, dim=0).mean().item()
    # print(f"Output variance (original linear): {output_variance}")


    # # # TT model
    # # tt_model = TensorTrain(shape=(32, 32, 32, 16), ranks=[50, 50, 50])
    # # tt_layer = TensorLinear(tt_model, in_features=1024, out_features=512, bias=True, 
    # #                         init_balance_mode='equal_variance', fan_mode='in', activation_scale=1)
    # # forward_test(tt_layer, input_tensor)

    # # # TR model
    # # tr_model = TensorRing(shape=(32, 32, 32, 16), ranks=[50, 50, 50, 50])
    # # tr_layer = TensorLinear(tr_model, in_features=1024, out_features=512, bias=True, 
    # #                         init_balance_mode='equal_variance', fan_mode='in', activation_scale=1)
    # # forward_test(tr_layer, input_tensor)

    # # # CP model
    # # cp_model = CP(shape=(16, 2, 32, 8, 4, 4, 4), rank=1000)
    # # cp_layer = TensorLinear(cp_model, in_features=1024, out_features=512, bias=True, 
    # #                         init_balance_mode='equal_variance', fan_mode='in', activation_scale=1)
    # # forward_test(cp_layer, input_tensor)

    # # Tucker model
    # tucker_model = Tucker(shape=(32, 32, 32, 16), core_shape=(10, 10, 10, 10))
    # tucker_layer = TensorLinear(tucker_model, in_features=1024, out_features=512, bias=True, 
    #                             init_balance_mode='equal_variance', fan_mode='in', activation_scale=1)
    # print(tucker_model)
    # forward_test(tucker_layer, input_tensor)

    # # # Tensor Train Matrix model
    # # ttm_model = TensorTrainMatrix(shape=((32, 32), (32, 16)), ranks=[4])
    # # ttm_layer = TensorLinear(ttm_model, in_features=1024, out_features=512, bias=True, 
    # #                          init_balance_mode='equal_variance', fan_mode='in', activation_scale=1)
    # # forward_test(ttm_layer, input_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Conv2d layer
    conv_layer = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    # Kaiming initialization with activation scale of 1
    nn.init.kaiming_uniform_(conv_layer.weight, a=0, mode='fan_in', nonlinearity='linear')
    input_tensor = torch.randn(64, 256, 32, 32).to(device)
    input_variance = torch.var(input_tensor, dim=(0, 2, 3)).mean().item()

    def forward_test(the_layer: TensorLayer, input_tensor: torch.Tensor):
        print(f"Testing layer: {the_layer}")
        output_tensor = the_layer(input_tensor)
        if hasattr(the_layer, 'factor_norms_sq_statistics'):
            print(f"Tensors Norm Statistics: {the_layer.factor_norms_sq_statistics()}")
        print(f"Output tensor shape: {output_tensor.shape}")
        output_variance = torch.var(output_tensor, dim=(0, 2, 3)).mean().item()
        print(f"Output variance: {output_variance}")

    print(f"Input variance: {input_variance}")
    forward_test(conv_layer, input_tensor)

    init = 'equal_variance'

    tt_model = TensorTrain(shape=(16, 16, 16, 32, 3, 3), ranks=[4, 4, 4, 4, 4]).to(device)
    tt_conv_layer = TensorConv2d(tt_model, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                bias=False, init_balance_mode=init, fan_mode='in', activation_scale=1).to(device)
    forward_test(tt_conv_layer, input_tensor)

    tr_model = TensorRing(shape=(16, 16, 16, 32, 3, 3), ranks=[4, 4, 4, 4, 4, 4]).to(device)
    tr_conv_layer = TensorConv2d(tr_model, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                bias=False, init_balance_mode=init, fan_mode='in', activation_scale=1).to(device) 
    forward_test(tr_conv_layer, input_tensor)

    cp_model = CP(shape=(256, 9, 512), ranks=[100]).to(device)
    cp_conv_layer = TensorConv2d(cp_model, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                bias=False, init_balance_mode=init, fan_mode='in', activation_scale=1).to(device)
    forward_test(cp_conv_layer, input_tensor)

    tucker_model = Tucker(shape=(16, 16, 16, 32, 3, 3), ranks=[4, 4, 4, 4, 4, 4]).to(device)
    tucker_conv_layer = TensorConv2d(tucker_model, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                    bias=False, init_balance_mode=init, fan_mode='in', activation_scale=1).to(device)
    forward_test(tucker_conv_layer, input_tensor)

    ttm_model = TensorTrainMatrix(shape=((16, 16), (16, 32), (3, 3)), ranks=[4, 4]).to(device)
    ttm_conv_layer = TensorConv2d(ttm_model, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                 bias=False, init_balance_mode=init, fan_mode='in', activation_scale=1).to(device)
    forward_test(ttm_conv_layer, input_tensor)