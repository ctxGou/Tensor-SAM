'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from model_tensor.tlayer import TensorConv2d, TensorLayer, TensorLinear
from model_tensor.td import TensorizedModel, TensorTrain, TensorRing, TensorTrainMatrix, CP, Tucker
from model_tensor.resnet import get_resnet
from typing import List, Tuple, Type

def get_planes_size_tensorized_resnet(td: Type[TensorizedModel]) -> List[Tuple[int, ...]]:
    """
    Returns the size of each layer in the tensorized ResNet model.
    This is useful for initializing the model with the correct dimensions.
    """
    if issubclass(td, TensorTrain):
        return [(4, 2, 2), (4, 4, 2), (4, 4, 4)]
    elif issubclass(td, TensorRing):
        return [(4, 2, 2), (4, 4, 2), (4, 4, 4)]
    elif issubclass(td, Tucker):
        return [(16,), (32,), (64,)]
    elif issubclass(td, CP):
        return [(16,), (32,), (64,)]

    # Note: I have no ideas how to handle TensorTrainMatrix yet.
    # Note2: I have no ideas how to tensorize the in/out channels yet.
    # Note3: This is painful!!!!

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TensorBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, weight_model_class: Type[TensorizedModel],
                 in_planes: Tuple[int, ...], planes: Tuple[int, ...], ranks: List[int],
                 stride=1, option='A', init: str = "equal_var"):
        super(TensorBasicBlock, self).__init__()

        self.conv1 = TensorConv2d(
            weight_model=weight_model_class(
                shape=in_planes + (3*3,) + planes,
                ranks=ranks,
            ),
            init_balance_mode=init,
            in_channels=np.prod(in_planes),
            out_channels=np.prod(planes),
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(np.prod(planes))

        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = TensorConv2d(
            weight_model=weight_model_class(
                shape=planes + (3*3,) + planes,
                ranks=ranks,
            ),
            init_balance_mode=init,
            in_channels=np.prod(planes),
            out_channels=np.prod(planes),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(np.prod(planes))

        self.shortcut = nn.Sequential()
        if stride != 1 or np.prod(in_planes) != np.prod(planes):
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, np.prod(planes)//4, np.prod(planes)//4), "constant", 0))
            elif option == 'B':
                # Oops, let's not Tensorize the shortcut. I won't use it anyway.
                self.shortcut = nn.Sequential(
                     nn.Conv2d(np.prod(in_planes), self.expansion * np.prod(planes), kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TensorResNet(nn.Module):
    def __init__(self, planes_size: List[Tuple[int, ...]],
                 ranks: List[int],
                 weight_model_class: Type[TensorizedModel],
                 block: Type[TensorBasicBlock],
                 num_blocks: List[int],
                 num_classes=10,
                 init: str = "equal_var"):
        super().__init__()

        self.weight_model_class = weight_model_class
        self.init = init
        self.in_planes = planes_size[0]
        self.conv1 = TensorConv2d(
            weight_model=weight_model_class(
                shape=(3,) + (3*3,) + planes_size[0],
                ranks=ranks,
            ),
            init_balance_mode=init,
            in_channels=3,
            out_channels=np.prod(planes_size[0]),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(np.prod(planes_size[0]))
        self.layer1 = self._make_layer(block, planes_size[0], num_blocks[0], ranks, stride=1)
        self.layer2 = self._make_layer(block, planes_size[1], num_blocks[1], ranks, stride=2)
        self.layer3 = self._make_layer(block, planes_size[2], num_blocks[2], ranks, stride=2)
        
        # Note: Let's not tensorize this. It is small!
        self.linear = nn.Linear(64, num_classes)


    def _make_layer(self, block, planes, num_blocks, ranks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.weight_model_class,
                self.in_planes, planes, 
                ranks,
                stride,
                init=self.init))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def dof(self) -> int:
        """
        Calculate the number of degrees of freedom (DOF) in the model.
        This is a rough estimate based on the number of parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def td_layer_factor_norm_statistics(self, reduce_avg: bool = True) -> Tuple[float, float]:
        """
        Computes the mean and variance of the squared norms of factors in each tensorized layer.
        """
        # Get all tensorized layers
        tensorized_layers = [layer for layer in self.modules() if isinstance(layer, TensorLayer)]
        mean_layers_norms_sq = []
        var_layers_norms_sq = []
        for layer in tensorized_layers:
            norms_sq = layer.factor_norms_sq_statistics()
            mean_layers_norms_sq.append(norms_sq[0])
            var_layers_norms_sq.append(norms_sq[1])

        if reduce_avg:
            mean_sq = np.mean(mean_layers_norms_sq).item()
            var_sq = np.mean(var_layers_norms_sq).item()
        else:
            mean_sq = np.array(mean_layers_norms_sq)
            var_sq = np.array(var_layers_norms_sq)
            
        return mean_sq, var_sq

def get_tresnet(depth: int = 20, 
            weight_model_class: Type[TensorizedModel] = TensorTrain, 
            rank: int = 100,
            num_classes: int = 10,
            init: str = "equal_var"
        ) -> Tuple[TensorResNet, int, float]:
    """
    Creates a tensorized ResNet model (TensorResNet) with the specified depth, tensorization method, rank, and number of output classes.

    Args:
        depth (int, optional): Depth of the ResNet model. Must be one of [20, 32, 44, 56, 110]. Default is 20.
        weight_model_class (Type[TensorizedModel], optional): The tensorization class to use for the model's weights (e.g., TensorTrain). Default is TensorTrain.
        rank (int, optional): The tensor rank to use for tensorized layers. Default is 100.
        num_classes (int, optional): Number of output classes for classification. Default is 10.

    Returns:
        Tuple[TensorResNet, int, float]: 
            - The constructed TensorResNet model.
            - The number of degrees of freedom (dof) in the tensorized model.
            - The compression ratio compared to the reference (non-tensorized) ResNet model.
    """
    planes_size = get_planes_size_tensorized_resnet(weight_model_class)
    model = TensorResNet(
        planes_size, 
        [rank], 
        weight_model_class, 
        TensorBasicBlock, 
        {
            20: [3, 3, 3],
            32: [5, 5, 5],
            44: [7, 7, 7],
            56: [9, 9, 9],
            110: [18, 18, 18]
        }[depth],
        num_classes=num_classes,
        init=init
    )
    _, reference_dof, _ = get_resnet(depth, num_classes=num_classes)
    compression_ratio = reference_dof / model.dof()
    return model, model.dof(), compression_ratio

if __name__ == "__main__":
    # Test the TensorResNet model
    # for rank in [10, 20, 50, 100, 200]:
    #     model, dof, compression_ratio = get_tresnet(depth=20, weight_model_class=CP, rank=rank)
    #     print(f"Rank: {rank}")
    #     print(f"Degrees of Freedom (DOF): {dof}")
    # print(f"Compression Ratio: {compression_ratio:.2f}")
    for rank in [6, 9, 13, 21, 24]:
        model, dof, compression_ratio = get_tresnet(depth=20, weight_model_class=Tucker, rank=rank)
        print(f"Rank: {rank}")
        print(f"Degrees of Freedom (DOF): {dof}")
        print(f"Compression Ratio: {compression_ratio:.2f}")