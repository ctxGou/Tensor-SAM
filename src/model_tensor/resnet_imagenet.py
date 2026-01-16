import torch

import torchvision.models as models
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from model_tensor.tlayer import TensorConv2d, TensorLayer, TensorLinear
from model_tensor.td import TensorizedModel, TensorTrain, TensorRing, TensorTrainMatrix, CP, Tucker
import warnings

import tensorly as tl
tl.set_backend('pytorch')

def get_resnet18(pretrained=True):
    """
    Load the ResNet-18 model from torchvision.

    Args:
        pretrained (bool): If True, loads the pretrained weights. Default is True.

    Returns:
        model (torch.nn.Module): ResNet-18 model.
    """
    model = models.resnet18(weights='DEFAULT' if pretrained else None)
    return model

def get_tt_resnet18(save_path=None, load_path="model_tensor/resnet18_tt.pth"):
    model = get_resnet18(pretrained=True)
    # find all 3x3 conv layers in the model
    # conv3x3_layers = []
    # for name, layer in model.named_modules():
    #     if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
    #         conv3x3_layers.append((name, layer))
    # print("Found 3x3 conv layers:")
    # for name, layer in conv3x3_layers:
    #     print(f"{name}: {layer}")

    ranks = {
        'layer1.0.conv1': [64, 64],
        'layer1.0.conv2': [64, 64],
        'layer1.1.conv1': [64, 64],
        'layer1.1.conv2': [64, 64],
        'layer2.0.conv1': [120, 60],
        'layer2.0.conv2': [100, 100],
        'layer2.1.conv1': [100, 100],
        'layer2.1.conv2': [100, 100],
        'layer3.0.conv1': [200, 150],
        'layer3.0.conv2': [135, 135],
        'layer3.1.conv1': [135, 135],
        'layer3.1.conv2': [135, 135],
        'layer4.0.conv1': [320, 200],
        'layer4.0.conv2': [170, 170],
        'layer4.1.conv1': [170, 170],
        'layer4.1.conv2': [170, 170]
    }
    # find all 3x3 and replace by TensorConv2d
    new_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
            if name in ranks:
                rank = ranks[name]
                new_layer = TensorConv2d(TensorTrain(shape=(layer.in_channels, 9, layer.out_channels),
                                            ranks=rank),
                                         layer.in_channels, layer.out_channels, layer.kernel_size[0],
                                         stride=layer.stride, padding=layer.padding, 
                                         bias=(layer.bias is not None))
                if load_path is None:
                    new_layer_weight_data_full = layer.weight.data.view(layer.in_channels, 9, layer.out_channels)
                    # tt decomposition
                    new_layer_weight_data_tt = tl.decomposition.tensor_train(new_layer_weight_data_full, rank=[1]+rank+[1]).factors
                    # assign the factors to the new layer
                    for i, factor in enumerate(new_layer_weight_data_tt):
                        # assign with padding
                        new_layer.weight_model.factors[i].data.zero_()
                        factorized_shape = factor.shape
                        new_layer.weight_model.factors[i].data[:factorized_shape[0], :factorized_shape[1], :factorized_shape[2]] = factor

                    if layer.bias is not None:
                        new_layer.bias_param.data = layer.bias.data
                new_layers.append((name, new_layer))

    # replace the layers in the model
    for name, new_layer in new_layers:
        parent_module = model
        for part in name.split('.')[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name.split('.')[-1], new_layer)
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
        print(f"Loaded TT ResNet-18 model from {load_path}")

    return model



def get_tt_resnet50(save_path=None, load_path="model_tensor/resnet50_tt.pth"):
    model = models.resnet50(weights='IMAGENET1K_V2')
    # find all 3x3 conv layers in the model
    # conv3x3_layers = []
    # for name, layer in model.named_modules():
    #     if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
    #         conv3x3_layers.append((name, layer))
    # print("Found 3x3 conv layers:")
    # for name, layer in conv3x3_layers:
    #     print(f"{name}: {layer}")

    ranks = {
        # 'layer1.0.conv2': [64, 64],
        # 'layer1.1.conv2': [64, 64],
        # 'layer1.2.conv2': [64, 64],

        # 'layer2.0.conv2': [120, 60],
        # 'layer2.1.conv2': [100, 100],
        # 'layer2.2.conv2': [100, 100],
        # 'layer2.3.conv2': [100, 100],
        
        # 'layer3.0.conv2': [200, 150],
        # 'layer3.1.conv2': [135, 135],
        # 'layer3.2.conv2': [135, 135],
        # 'layer3.3.conv2': [135, 135],
        # 'layer3.4.conv2': [135, 135],
        # 'layer3.5.conv2': [135, 135],

        # 'layer4.0.conv2': [320, 200],
        'layer4.1.conv2': [170, 170],
        'layer4.2.conv2': [170, 170]
    }
    # find all 3x3 and replace by TensorConv2d
    new_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
            if name in ranks:
                rank = ranks[name]
                new_layer = TensorConv2d(TensorTrain(shape=(layer.in_channels, 9, layer.out_channels),
                                            ranks=rank),
                                         layer.in_channels, layer.out_channels, layer.kernel_size[0],
                                         stride=layer.stride, padding=layer.padding, 
                                         bias=(layer.bias is not None))
                if load_path is None:
                    new_layer_weight_data_full = layer.weight.data.view(layer.in_channels, 9, layer.out_channels)
                    # tt decomposition
                    new_layer_weight_data_tt = tl.decomposition.tensor_train(new_layer_weight_data_full, rank=[1]+rank+[1]).factors
                    # assign the factors to the new layer
                    for i, factor in enumerate(new_layer_weight_data_tt):
                        # assign with padding
                        new_layer.weight_model.factors[i].data.zero_()
                        factorized_shape = factor.shape
                        new_layer.weight_model.factors[i].data[:factorized_shape[0], :factorized_shape[1], :factorized_shape[2]] = factor

                    if layer.bias is not None:
                        new_layer.bias_param.data = layer.bias.data
                new_layers.append((name, new_layer))

    # replace the layers in the model
    for name, new_layer in new_layers:
        parent_module = model
        for part in name.split('.')[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name.split('.')[-1], new_layer)
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
        print(f"Loaded TT ResNet-18 model from {load_path}")
    return model

def get_vit(pretrained=True, size='b_32'):
    """
    Load the Vision Transformer (ViT) model from torchvision.

    Args:
        pretrained (bool): If True, loads the pretrained weights. Default is True.

    Returns:
        model (torch.nn.Module): ViT model.
    """
    from torchvision.models import vit_b_16, vit_b_32
    if size == 'b_32':
        model = vit_b_16(weights='DEFAULT' if pretrained else None)
    elif size == 'b_16':
        model = vit_b_32(weights='DEFAULT' if pretrained else None)
    return model

def get_tr_vit(save_path=None, rank=50, load_path="model_tensor/tr_alss_vit_b_32.pth", size='b_16'):
    """
    Load the Tensorized Vision Transformer (ViT) model.

    Args:
        pretrained (bool): If True, loads the pretrained weights. Default is True.
        load_path (str): Path to load the model weights from.

    Returns:
        model (torch.nn.Module): Tensorized ViT model.
    """
    from torchvision.models import vit_b_16, vit_b_32
    from tensorly.decomposition._tr_als import tensor_ring_als_sampled, tensor_ring_als
    tl.set_backend('pytorch')
    if size == 'b_32':
        model = vit_b_32(weights='DEFAULT')
    elif size == 'b_16':
        model = vit_b_16(weights='DEFAULT')
    
    model = model.to('cuda:0')
    # Convert the ViT model to a tensorized version
    # Assuming a similar conversion process as in ResNet
    new_layers = []
    # Find all MLP
    for name, layer in model.named_modules():
        if 'mlp' in name and isinstance(layer, nn.Linear):
            # Convert to TensorLinear
            # 768*3072 -> 32*32*32*72
            new_layer = TensorLinear(TensorRing(shape=(32, 32, 32, 72),
                                            ranks=[rank]),
                                     layer.in_features, layer.out_features)
            if load_path is None:
                new_layer_weight_data_full = layer.weight.data.view(32, 32, 32, 72)
                print(new_layer_weight_data_full.device)
                # tr decomposition
                new_layer_weight_data_tr = tensor_ring_als(new_layer_weight_data_full, rank=rank, verbose=True, n_iter_max=30).factors
                # assign the factors to the new layer
                for i, factor in enumerate(new_layer_weight_data_tr):
                    new_layer.weight_model.factors[i].data.copy_(factor)
                if layer.bias is not None:
                    new_layer.bias_param.data = layer.bias.data
            new_layers.append((name, new_layer))
            print(f"Converted {name} to TensorLinear with shape {new_layer.weight_model.shape} and ranks {new_layer.weight_model.ranks}")
    # replace the layers in the model
    for name, new_layer in new_layers:
        parent_module = model
        for part in name.split('.')[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name.split('.')[-1], new_layer)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
        print(f"Loaded Tensorized ViT-{size} model from {load_path}")

    return model

if __name__ == "__main__":
    # Example usage
    device1 = 'cuda:0'
    device2 = 'cuda:1'
    model = get_tt_resnet50(save_path="model_tensor/resnet50_tt.pth").to(device1)
    model_original = models.resnet50(weights='IMAGENET1K_V2').to(device2)
    

    # model = get_tr_vit(rank=50, save_path=None, load_path="model_tensor/tr_alss_vit_b_32.pth", size='b_32').to(device1)
    # model_original = get_vit(pretrained=True).to(device2)

    dof = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dof_original = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
    print(f"Degrees of freedom (Tensorized): {dof}, cr: {dof_original / dof:.2f}")
    print(f"Degrees of freedom (Original): {dof_original}")


    from datasets import load_dataset

    # Load the validation set only
    dataset = load_dataset("imagenet-1k", split="validation", num_proc=5, cache_dir='/data1/imagenet/datasets/')
    print(f"Dataset size: {len(dataset)}")
    # transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    def transform_function(batch):
        # Apply transform to each image in the batch
        batch['image'] = [transform(img) for img in batch['image']]
        return batch
    dataset.set_transform(transform_function)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                             num_workers=8, pin_memory=True, prefetch_factor=4)
    # evaluate the model

    model.eval()
    model_original.eval()

    total, correct_a, correct_b = 0, 0, 0
    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        images_1 = batch['image'].to(device1)
        images_2 = batch['image'].to(device2)
        labels_1 = batch['label'].to(device1)
        labels_2 = batch['label'].to(device2)
        outputs_a = model(images_1)
        # outputs_b = model_original(images_2)
        _, predicted_a = torch.max(outputs_a, 1)
        # _, predicted_b = torch.max(outputs_b, 1)
        total += labels_1.size(0)
        correct_a += (predicted_a == labels_1).sum().item()
        # correct_b += (predicted_b == labels_2).sum().item()
    print(f"Accuracy of Tensorized: {100 * correct_a / total:.2f}%")
    print(f"Accuracy of Original: {100 * correct_b / total:.2f}%")

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
            self.param_groups[0]["alpha"] = self.defaults["alpha"] * (1/2 + 1/2*torch.cos(torch.tensor(current_T / T_max * 3.141592653589793)))
        elif value == "constant":
            # Hehe! I love doing nothing.
            return
        else:
            raise ValueError("Invalid value for alpha update. Choose from 'linear', 'cosine', or 'constant'.")

    def step(self):
        for tlayer in self.model.modules():
            if isinstance(tlayer, TensorLayer):
                # Calculate the squared gradient norms for each tensorized layer
                grad_norms_sq_mean, _, grad_norms_sq = tlayer.factor_grad_norms_sq_statistics()
                sqrt_sum_grad_norms_sq = torch.sqrt(torch.sum(torch.tensor(grad_norms_sq)))
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