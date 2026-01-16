import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import wandb
import argparse
import abc
from sklearn.model_selection import train_test_split

from argparse import Namespace

from model_tensor.resnet import get_resnet
from model_tensor.tresnet import get_tresnet
from model_tensor.tresnet import TensorResNet
from model_tensor.td import TensorTrain, TensorRing, CP, Tucker

from model_tensor.wrn import Wide_ResNet, get_wrn
from model_tensor.twrn import Tensor_Wide_ResNet, get_twrn

from model_tensor.gbar import gBAR

from data.noisy_cifar import CIFAR10 as NoisyCIFAR10

def get_parser():
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Trainer")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the optimizer")
    parser.add_argument("--cifar", type=int, default=10, choices=[10, 100], help="CIFAR dataset version (10 or 100)")
    parser.add_argument("--noise_rate", type=float, default=-1.0, help="Noise rate for CIFAR-10")

    # tensor decomposition weight model
    parser.add_argument("--use_tnn", type=int, default=0, help="Use Tensorial Neural Networks")
    parser.add_argument("--weight_model_class", type=str, default="TensorTrain", help="Class of the weight model to use")
    parser.add_argument("--rank", type=int, default=100, help="Rank for tensor decomposition models")
    parser.add_argument("--init", type=str, default="equal_var", choices=["equal_var", "equal_norm"], help="Initialization method for the weight model")

    # resnet configuration
    parser.add_argument("--resnet_depth", type=int, default=20, choices=[20, 28, 32, 44, 56, 110], help="Depth of the ResNet model")
    parser.add_argument("--wide_resnet_widen_factor", type=int, default=0, help="Widen factor for Wide ResNet (0 for standard ResNet)")

    # wandb configuration
    parser.add_argument("--wandb_project", type=str, default="gBAR-cifar10-100", help="WandB project name")

    # method configuration: sgd/sam/asam/gbar
    parser.add_argument("--method", type=str, default="sgd", choices=["sgd", "sam", "asam", "gbar2", "balancedinit"], help="Training method to use")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="Rho parameter for SAM")
    parser.add_argument("--gbar_alpha", type=float, default=0.2, help="Alpha parameter for gBAR")
    parser.add_argument("--gbar_alpha_scheduler", type=str, default=None, choices=["linear", "cosine", "constant"], help="Scheduler for gBAR alpha")


    # system configuration
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")


    # hyperparameter optimization (HPO) configuration
    parser.add_argument("--hpo", type=int, default=0, help="Enable hyperparameter optimization (1 for yes, 0 for no)")

    # get best HPO configuration
    parser.add_argument("--hpo_config", type=str, default=None, help="Path to the best HPO configuration file")

    parser.add_argument("--time_benchmark", action="store_true", help="Enable time benchmarking")

    args = parser.parse_args()

    if (args.hpo_config is not None) and (args.hpo_config != "None"):
        # load best HPO configuration from file
        with open(args.hpo_config, 'r') as f:
            lines = f.readlines()
        hpo_args_best = eval(lines[2]) # learned HPO solution

        args_dict = vars(args)
        args_dict.update(hpo_args_best)
        args = Namespace(**args_dict)

    if args.method == "balancedinit":
        if args.use_tnn != 1:
            raise ValueError("Balanced initialization is only supported for tensor decomposition models.")
        args.method = "sgd"
        args.init = "equal_norm"

    return args



class Trainer(abc.ABC):
    # abstract class for CIFAR trainers
    # initialize dataloader, wandb run, and model
    def __init__(self, args: argparse.Namespace):
        print(args)
        self.args = args
        self.device = torch.device(self.args.device)
        self._set_seed(self.args.seed)

        self.hpo = bool(args.hpo)

        # Initialize DataLoader
        self.train_loader, self.test_loader = self._init_dataloaders()

        # Initialize model
        if self.args.use_tnn:
            weight_model_class = {
                "TensorTrain": TensorTrain,
                "TensorRing": TensorRing,
                "CP": CP,
                "Tucker": Tucker
            }[self.args.weight_model_class]
            if self.args.wide_resnet_widen_factor == 0:
                self.model, dof, cr = get_tresnet(depth=self.args.resnet_depth, weight_model_class=weight_model_class, 
                                     rank=self.args.rank, num_classes=self.args.cifar, init=self.args.init)
            else:
                self.model, dof, cr = get_twrn(depth=self.args.resnet_depth, weight_model_class=weight_model_class,
                                     rank=self.args.rank, num_classes=self.args.cifar, 
                                     widen_factor=self.args.wide_resnet_widen_factor, init=self.args.init)
        else:
            if self.args.wide_resnet_widen_factor == 0:
                self.model, dof, cr = get_resnet(depth=self.args.resnet_depth, num_classes=self.args.cifar)
            else:
                self.model, dof, cr = get_wrn(depth=self.args.resnet_depth, num_classes=self.args.cifar, 
                                             widen_factor=self.args.wide_resnet_widen_factor)
        self.model.to(self.device)

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

        # Initialize WandB
        wandb.init(project=self.args.wandb_project, config=self.args)
        # Load dof and compression ratio into WandB
        wandb.config.update({"dof": dof, "compression_ratio": cr})
        
        if self.args.method == "sgd":
            if self.args.use_tnn:
                wandb.run.name = f"{self.args.weight_model_class}_{self.args.resnet_depth}_rank{self.args.rank}_sgd"
            else:
                wandb.run.name = f"resnet{self.args.resnet_depth}_sgd"
        elif self.args.method in ["sam", "asam"]:
            if self.args.use_tnn:
                wandb.run.name = f"{self.args.weight_model_class}_{self.args.resnet_depth}_rank{self.args.rank}_{self.args.method}_rho{self.args.sam_rho}"
            else:
                wandb.run.name = f"resnet{self.args.resnet_depth}_{self.args.method}_rho{self.args.sam_rho}"
        elif self.args.method == "gbar2":
            if self.args.use_tnn:
                wandb.run.name = f"{self.args.weight_model_class}_{self.args.resnet_depth}_rank{self.args.rank}_gbar_alpha{self.args.gbar_alpha}"
            else:
                wandb.run.name = f"resnet{self.args.resnet_depth}_gbar_alpha{self.args.gbar_alpha}"


    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _init_dataloaders(self):
        train_transform = transforms.Compose([            
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                     (0.2023, 0.1994, 0.2010)),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225)),
        ])
        data_dir = '/data1/imagenet/data'
        # If data_dir does not exist, set 
        import os
        if not os.path.exists(data_dir):
            data_dir = './data'
        if self.args.cifar == 10:
            train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
            if self.args.noise_rate >= 0.0:
                # Use NoisyCIFAR10 with the specified noise rate
                train_dataset = NoisyCIFAR10(root=data_dir, train=True, download=True, 
                                             transform=train_transform, corruption_prob=self.args.noise_rate)
                test_dataset = NoisyCIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
        elif self.args.cifar == 100:
            train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)
        else:
            raise ValueError("CIFAR version must be 10 or 100")
        
        if self.hpo:
            # balanced subsampling for reduced train dataset and validation set
            train_indices = np.arange(len(train_dataset))
            train_labels = np.array(train_dataset.targets)
            # fixed split
            train_indices, validation_indices = train_test_split(train_indices, test_size=0.1, train_size=0.9,
                                                                 stratify=train_labels, random_state=114514)


            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(train_dataset, validation_indices)


        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, prefetch_factor=2)

        return train_loader, test_loader
        
    @abc.abstractmethod
    def train(self):
        """
        Abstract method to be implemented by subclasses for training the model.
        """
        pass

    @abc.abstractmethod
    def train_epoch(self):
        pass

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.loss(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        loss /= len(self.test_loader)
        return {"accuracy": accuracy, "test_loss": loss}

    def run(self):
        """
        Run the training and evaluation process.
        """
        self.train()
        self.evaluate()
        wandb.finish()


class BaseTrainer(Trainer):
    """
    Base trainer class for CIFAR-10/100 models.
    This class provides a basic structure for training and evaluating models.
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        bn_params = []
        non_bn_params = []
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'bias' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)

        self.params = [
            {'params': bn_params, 'weight_decay': 0},
            {'params': non_bn_params, 'weight_decay': 1e-4}
        ]
        self.optimizer = torch.optim.SGD(self.params, lr=self.args.lr, momentum=0.9)
        if self.args.method == "gbar2":
            self.optimizer = gBAR(model=self.model, 
                                    params=self.params,
                                    base_optimizer=torch.optim.SGD,
                                    alpha=self.args.gbar_alpha,
                                    lr=self.args.lr,
                                    momentum=0.9)
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.num_epochs)

    def train(self):
        for epoch in range(self.args.num_epochs):
            if isinstance(self.optimizer, gBAR):
                # update gBAR alpha
                self.optimizer.update_alpha(epoch, self.args.num_epochs, 
                                            self.args.gbar_alpha_scheduler)

            self.model.train()
            train_running_loss = self.train_epoch()  # Train for one epoch

            if isinstance(self.model, TensorResNet):
                # Log tensor decomposition statistics
                mean_sq, var_sq = self.model.td_layer_factor_norm_statistics(reduce_avg=True)
                wandb.log({"mean_sq": mean_sq, "var_sq": var_sq}, step=epoch+1)

            self.scheduler.step()
            eval_results = self.evaluate()
            wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=epoch+1)
            wandb.log({"epoch": epoch + 1, "train_loss": train_running_loss} | eval_results, step=epoch+1)
            # print(f"Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}, "
            #       f"Accuracy: {eval_results['accuracy']:.2f}%, Test Loss: {eval_results['loss']:.4f}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        if self.args.time_benchmark:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            epoch_time = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.args.time_benchmark:
                start.record()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if self.args.time_benchmark:
                end.record()
                torch.cuda.synchronize()

            total_loss += loss.item()
            if self.args.time_benchmark:
                batch_time = start.elapsed_time(end) / 1000.0
                epoch_time += batch_time
            
        # **Print** averaged iteration time for the epoch
        if self.args.time_benchmark:
            print(f"Average Iteration Time: {epoch_time / len(self.train_loader):.4f} seconds")

        return total_loss / len(self.train_loader)

class SAMTrainer(BaseTrainer):
    """
    Trainer class for SAM (Sharpness-Aware Minimization) method.
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from model_tensor.sam import SAM
        if self.args.method == "sam":
            self.optimizer = SAM(params=self.params,
                                    base_optimizer=torch.optim.SGD,
                                    rho=self.args.sam_rho,
                                    adaptive=False,
                                    lr=self.args.lr,
                                    momentum=0.9)
        elif self.args.method == "asam":
            self.optimizer = SAM(params=self.params,
                                    base_optimizer=torch.optim.SGD,
                                    rho=self.args.sam_rho,
                                    adaptive=True,
                                    lr=self.args.lr,
                                    momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.num_epochs)

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.model.train()
            
            train_running_loss = self.train_epoch()  # Train for one epoch

            if isinstance(self.model, TensorResNet):
                # Log tensor decomposition statistics
                mean_sq, var_sq = self.model.td_layer_factor_norm_statistics(reduce_avg=True)
                wandb.log({"mean_sq": mean_sq, "var_sq": var_sq}, step=epoch+1)

            self.scheduler.step()
            eval_results = self.evaluate()
            # log learning rate
            wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=epoch+1)
            wandb.log({"epoch": epoch + 1, "train_loss": train_running_loss} | eval_results, step=epoch+1)

    def train_epoch(self):
        """
        Train the model for one epoch using SAM.
        """
        self.model.train()
        total_loss = 0.0
        if self.args.time_benchmark:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            epoch_time = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            if self.args.time_benchmark:
                start.record()

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            total_loss += loss.item()

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.second_step(zero_grad=True)

            if self.args.time_benchmark:
                end.record()
                torch.cuda.synchronize()
                batch_time = start.elapsed_time(end) / 1000.0
                epoch_time += batch_time

        # **Print** averaged iteration time for the epoch
        if self.args.time_benchmark:
            print(f"Average Iteration Time: {epoch_time / len(self.train_loader):.4f} seconds")

        return total_loss / len(self.train_loader)



if __name__ == "__main__":
    args = get_parser()
    if args.method in ["sam", "asam"]:
        trainer = SAMTrainer(args=args)
    else:
        trainer = BaseTrainer(args=args)
    trainer.run()
        

