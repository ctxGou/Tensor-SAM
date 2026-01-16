import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from model_tensor.tlayer import TensorConv2d, TensorLayer, TensorLinear
from model_tensor.td import TensorizedModel, TensorTrain, TensorRing, TensorTrainMatrix, CP, Tucker
from model_tensor.resnet import get_resnet
from typing import List, Tuple, Type
from model_tensor.wrn import Wide_ResNet

import sys
import numpy as np

def get_planes_size_tensorized_resnet(td: Type[TensorizedModel], widen_factor: int) -> List[Tuple[int, ...]]:
    """
    Returns the size of each layer in the tensorized ResNet model.
    nStages: [16, 16*k, 32*k, 64*k]
    """
    k = widen_factor
    if issubclass(td, TensorTrain):
        return [(4, 2, 2), (4, 2, 2, k), (4, 4, 2, k), (4, 4, 4, k)]
    elif issubclass(td, TensorRing):
        return [(4, 2, 2), (4, 2, 2, k), (4, 4, 2, k), (4, 4, 4, k)]
    elif issubclass(td, Tucker):
        return [(16,), (16*k,), (32*k,), (64*k,)]
    elif issubclass(td, CP):
        return [(16,), (16*k,), (32*k,), (64*k,)]


def tconv3x3(weight_model_class: Type[TensorizedModel], ranks: List[int],
        in_planes, out_planes, stride=1, init="equal_var"):
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    return TensorConv2d(
        weight_model=weight_model_class(
            shape=in_planes + (3*3,) + out_planes,
            ranks=ranks,
        ),
        in_channels=np.prod(in_planes),
        out_channels=np.prod(out_planes),
        init_balance_mode=init,
        kernel_size=3, stride=stride, padding=1, bias=True
    )

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class tensor_wide_basic(nn.Module):
    def __init__(self, 
                    weight_model_class: Type[TensorizedModel],
                    ranks: List[int],
                    in_planes: Tuple[int, ...],
                    planes: Tuple[int, ...],
                    dropout_rate, 
                    stride=1,
                    init="equal_var"
                ):
        super(tensor_wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(np.prod(in_planes))
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        # print(f"tensor_wide_basic: in_planes={in_planes}, planes={planes}, ranks={ranks}")
        self.conv1 = TensorConv2d(
            weight_model=weight_model_class(
                shape=in_planes + (3*3,) + planes,
                ranks=ranks,
            ),
            in_channels=np.prod(in_planes),
            out_channels=np.prod(planes),
            init_balance_mode=init,
            kernel_size=3, padding=1, bias=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(np.prod(planes))
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = TensorConv2d(
            weight_model=weight_model_class(
                shape=planes + (3*3,) + planes,
                ranks=ranks,
            ),
            in_channels=np.prod(planes),
            out_channels=np.prod(planes),
            init_balance_mode=init,
            kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or np.prod(in_planes) != np.prod(planes):
            # Let's not tensorize this.
            self.shortcut = nn.Sequential(
                nn.Conv2d(np.prod(in_planes), np.prod(planes), kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Tensor_Wide_ResNet(nn.Module):
    def __init__(self, nStages_tensorized: List[Tuple[int, ...]],
                    ranks: List[int],
                    weight_model_class: Type[TensorizedModel],
                    depth: int, 
                    widen_factor: int, 
                    dropout_rate, 
                    num_classes,
                    init="equal_var"
                    ):
        super(Tensor_Wide_ResNet, self).__init__()
        self.in_planes = nStages_tensorized[0]

        self.weight_model_class = weight_model_class
        self.init = init
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = nStages_tensorized

        self.conv1 = tconv3x3(weight_model_class, ranks,
                              (3,), nStages[0], stride=1)
        self.layer1 = self._wide_layer(tensor_wide_basic, nStages[1], n, dropout_rate, ranks, stride=1)
        self.layer2 = self._wide_layer(tensor_wide_basic, nStages[2], n, dropout_rate, ranks, stride=2)
        self.layer3 = self._wide_layer(tensor_wide_basic, nStages[3], n, dropout_rate, ranks, stride=2)
        self.bn1 = nn.BatchNorm2d(np.prod(nStages[3]), momentum=0.9)
        self.linear = nn.Linear(np.prod(nStages[3]), num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, ranks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.weight_model_class, ranks,
                                self.in_planes, planes, dropout_rate, stride, self.init))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
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

def get_twrn(
        weight_model_class: Type[TensorizedModel],
        rank: int,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3,
        num_classes: int = 10,
        init: str = "equal_var"
    ) -> Tensor_Wide_ResNet:
    """
    Returns a tensorized Wide ResNet model.
    """
    ranks = [rank]
    nStages_tensorized = get_planes_size_tensorized_resnet(weight_model_class, widen_factor)
    model = Tensor_Wide_ResNet(
        nStages_tensorized=nStages_tensorized,
        ranks=ranks,
        weight_model_class=weight_model_class,
        depth=depth,
        widen_factor=widen_factor,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        init=init
    )
    uncompressed_model = Wide_ResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
    reference_dof = sum(p.numel() for p in uncompressed_model.parameters() if p.requires_grad)
    compression_ratio = reference_dof / model.dof()
    return model, model.dof(), compression_ratio

if __name__ == '__main__':
    for rank in [5, 24, 84, 400]:
        model, dof, compression_ratio = get_twrn(depth=28, widen_factor=10, weight_model_class=CP, rank=rank)
        print(f"Rank: {rank}")
        print(f"Degrees of Freedom (DOF): {dof}")
        print(f"Compression Ratio: {compression_ratio:.2f}")