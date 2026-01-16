import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model_tensor.resnet_imagenet import get_resnet18, get_tt_resnet18, get_tt_resnet50
from datasets import load_dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
import wandb

from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
# ignore scheduler step before optimizer step warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet Training with Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.008, type=float,
                        help='initial learning rate (default: 0.008)')
    parser.add_argument('--epochs', default=25, type=int,
                        help='number of total epochs to run (default: 25)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sam', 'gbar', 'sgd'],
                        help='optimizer type (default: sgd)')
    parser.add_argument('--wandb-project', default='ft-tnn', type=str,
                        help='wandb project name for logging')
    parser.add_argument('--batch-size', default=2048, type=int,
                        help='total batch size across all GPUs (default: 2048)')
    parser.add_argument('--rho', default=0.05, type=float,
                        help='rho parameter (default: 0.05)')
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='alpha parameter (default: 0.2)')
    return parser.parse_args()


def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

def transform_function(batch):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # to rgb
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    batch['image'] = [transform(img) for img in batch['image']]
    return batch


def val_transform_function(batch):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    batch['image'] = [transform(img) for img in batch['image']]
    return batch

def validate(model, val_loader, rank):
    """Validation function to evaluate the model on the validation set."""
    model.eval()
    total, top1, top5 = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', disable=(rank != 0)):
            imgs, labels = batch['image'], batch['label']
            imgs, labels = imgs.to(rank), labels.to(rank)
            outputs = model(imgs)
            total += labels.size(0)
            _, preds = outputs.topk(1, 1, True, True)
            top1 += (preds.view(-1) == labels).sum().item()
            _, preds_5 = outputs.topk(5, 1, True, True)
            top5 += (preds_5 == labels.view(-1, 1)).sum().item()
    totals = torch.tensor([total, top1, top5], device=rank)
    dist.reduce(totals, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        total, top1, top5 = totals.tolist()
        print(f'Validation - Total: {total}, Top1: {top1} ({top1/total}), Top5: {top5} ({top5/total})')
        return total, top1, top5
    else:
        return 0, 0, 0

def train_worker(rank, world_size, args=None):

    # wandb setup
    if rank == 0 and args.wandb_project:
        wandb.init(project=args.wandb_project, config=args, reinit=True)
        wandb.config.update(args)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    num_workers = min(8, os.cpu_count() // world_size)
    per_gpu_batch_size = args.batch_size // world_size
    # Data loaders with sampler

    # check hostname
    print(f'Worker {rank} started on {os.uname().nodename}')
    train_ds = load_dataset("imagenet-1k", split="train", num_proc=5, cache_dir='/data1/imagenet/datasets/')
    train_ds.set_transform(transform_function)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    loader = DataLoader(train_ds, batch_size=per_gpu_batch_size, sampler=sampler, prefetch_factor=4,
                        num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # validation dataset
    val_ds = load_dataset("imagenet-1k", split="validation", num_proc=5, cache_dir='/data1/imagenet/datasets/')
    val_ds.set_transform(val_transform_function)
    val_loader = DataLoader(val_ds, batch_size=per_gpu_batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=4,
                           sampler=DistributedSampler(val_ds, num_replicas=world_size, rank=rank),
                            pin_memory=True, persistent_workers=True)

    # Model, DDP and optimizer
    model = get_resnet18(pretrained=True)
    model = get_tt_resnet18().to(rank)
    # model = get_tt_resnet50().to(rank)
    # model = model.to(rank)
    # trainable = ['layer4.1.conv2', 'layer4.2.conv2']
    # for name, param in model.named_parameters():
    #     if not any(trainable_param in name for trainable_param in trainable):
    #         param.requires_grad = False
    # model.apply(freeze_bn)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    total, top1, top5 = validate(model, val_loader, rank)
    if total != 0:
        print(f'Initial Validation - Top1: {top1/total:.4f}, Top5: {top5/total:.4f}, Total: {total}')
    
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'sam':
        from model_tensor.resnet_imagenet import SAM
        optimizer = SAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'gbar':
        from model_tensor.resnet_imagenet import gBAR
        optimizer = gBAR(model,
            model.parameters(), optim.SGD, alpha=args.alpha, lr=args.lr, weight_decay=args.weight_decay)
    
    
    scheduler = OneCycleLR(optimizer, 
                            max_lr=args.lr,
                            steps_per_epoch=len(loader), 
                            epochs=args.epochs,
                            pct_start=0.2,
                            final_div_factor=0.1
                           )
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs*len(loader), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(rank)

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss, running_grad_norm = 0.0, 0.0
        local_epoch_time = 0.0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs} lr={optimizer.param_groups[0]["lr"]:.6f}', disable=(rank != 0)):
            # model.apply(freeze_bn)
            imgs, labels = batch['image'], batch['label']
            imgs, labels = imgs.to(rank), labels.to(rank)

            if rank == 0:
                start.record()
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item() * imgs.size(0)
            if args.optimizer != 'sam':
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.second_step()
            running_grad_norm += torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).item()

            scheduler.step()
            if rank == 0:
                end.record()
                torch.cuda.synchronize()
                local_epoch_time += start.elapsed_time(end) / 1000.0  # convert
            if args.optimizer == 'gbar':
                # optimizer.update_alpha(epoch, args.epochs)
                al = optimizer.param_groups[0]["alpha"]
        if args.optimizer == 'gbar':
            print(f'Alpha updated to {al:.6f} at epoch {epoch+1}')
        
        # Average loss and grad norm
        total = len(loader.dataset)
        loss = running_loss / total
        numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_norm = running_grad_norm / total / numel

        # gather loss and grad norm across all processes
        loss_tensor = torch.tensor(loss, device=rank)
        grad_norm_tensor = torch.tensor(grad_norm, device=rank)
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(grad_norm_tensor, dst=0, op=dist.ReduceOp.AVG)

        # Validation
        total, top1, top5 = validate(model, val_loader, rank)

        if rank == 0:
            # Log tensor decomposition statistics
            mean_sq = []
            for layer in model.modules():
                if hasattr(layer, 'factor_norms_sq_statistics'):
                    mean_sq_layer, var_sq_layer = layer.factor_norms_sq_statistics()
                    mean_sq.append(mean_sq_layer)
            mean_sq = sum(mean_sq) / len(mean_sq) if mean_sq else 0.0
            print(f'Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}, Grad Norm: {grad_norm:.4f}, Mean Sq: {mean_sq:.4f}')
            wandb.log({'epoch': epoch + 1, 'loss': loss, 'top1': top1 / total, 'top5': top5 / total, 'mean_sq': mean_sq, 'grad_norm': grad_norm_tensor.item()})
            # Save model checkpoint


        if rank == 0:
            epoch_time = torch.tensor(local_epoch_time, device=rank)
            print(f'Epoch {epoch+1}/{args.epochs} - Average Epoch Time: {epoch_time:.4f} seconds')
            wandb.log({'epoch_time': epoch_time.item()})
        # wait for all processes to finish logging
        dist.barrier() 

    if rank == 0:
        # Save the final model
        identifier = f'resnet18_tt_{args.optimizer}_rho{args.rho}_alpha{args.alpha}_bs{args.batch_size}_lr{args.lr}_epochs{args.epochs}'
        torch.save(model.state_dict(), f'{identifier}.pth')
    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
