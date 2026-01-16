import torch
import optuna
import wandb
import argparse

# Import your refactored components
from cifar_trainer import get_parser, BaseTrainer, SAMTrainer 

# Define the number of epochs for each HPO trial
HPO_EPOCHS = 80

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """`
    Objective function for a single trial.
    The 'args' object comes pre-filled from the command line.
    """
    args.lr = 0.1
    
    # Conditionally suggest parameters for SAM or gBAR
    if args.method == "sam":
        # args.sam_rho = trial.suggest_float("sam_rho", 1e-3, 5e-1, log=True)
        
        # change to {1e-3, 5e-3, 1e-2, 5e-2, 1e-1}
        args.sam_rho = trial.suggest_categorical("sam_rho", [1e-3, 5e-3, 1e-2, 5e-2, 1e-1])

    elif args.method == "gbar2":
        # args.gbar_alpha = trial.suggest_float("gbar_alpha", 1e-4, 3e-3, log=True)
        # args.gbar_alpha_scheduler = trial.suggest_categorical("gbar_alpha_scheduler", ["linear", "cosine", "constant"])

        # change to {5e-4, 1e-3, 5e-3, 1e-2, 5e-2}
        args.gbar_alpha = trial.suggest_categorical("gbar_alpha", [1e-1, 5e-2, 1e-2, 5e-3, 1e-3])
        args.gbar_alpha_scheduler = trial.suggest_categorical("gbar_alpha_scheduler", ["linear"])

    # 2. Setup and run the trainer for this trial
    # (The rest of this logic is the same as the previous example)
    
    # Set seed for this specific trial for data loading consistency
    # Note: Optuna handles seeding for its samplers internally.
    torch.manual_seed(args.seed)

    args.num_epochs = HPO_EPOCHS

    if args.method in ["sam", "asam"]:
        trainer = SAMTrainer(args)
    else:
        trainer = BaseTrainer(args)
    
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=HPO_EPOCHS)

    # Training loop with pruning
    for epoch in range(HPO_EPOCHS):
        loss = trainer.train_epoch() # Train for one epoch
        trainer.scheduler.step() # Step the learning rate scheduler
        eval_results = trainer.evaluate() # Evaluate on the validation set
        validation_accuracy = eval_results["accuracy"]
        validation_loss = eval_results["test_loss"]
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss,
            "val_accuracy": validation_accuracy,
            "val_loss": validation_loss,
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }, step=epoch + 1)
        trial.report(validation_accuracy, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.TrialPruned()

    wandb.finish()
    return validation_accuracy

if __name__ == "__main__":
    args = get_parser()
    
    # Use a unique study name for each HPO run for better organization
    study_name = f"hpo_{args.weight_model_class}_rank{args.rank}_{args.method}"
    if args.use_tnn == 0:
        study_name = f"hpo_{args.method}"
    # Use a pruner for efficiency
    # pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10)
    # change to a do-nothing pruner for now
    pruner = optuna.pruners.NopPruner()

    # create grid search space
    if args.method == "sam":
        search_space = {
            "sam_rho": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        }
    elif args.method == "gbar2":
        search_space = {
            "gbar_alpha": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
            "gbar_alpha_scheduler": ["linear"],
        }

    # calculate the number of trials based on the search space
    n_trials = 1
    for key, values in search_space.items():
        n_trials *= len(values)

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name, load_if_exists=True,
                                sampler=optuna.samplers.GridSampler(search_space))
    
    # Use a lambda to pass the fixed args from the command line into the objective function
    study.optimize(lambda trial: objective(trial, args), n_trials=n_trials, n_jobs=1)
    
    print(f"Study {study_name} complete.")
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ", study.best_trial.params)

    # Save study and best trial results
    # Create folder /{args.resnet_depth}/{args.weight_model_class}/{args.rank}/{args.method}/
    import os
    # Create the directory structure
    output_dir = f"./results/cifar{args.cifar}/resnet{args.resnet_depth}/{args.weight_model_class}/{args.rank}/{args.method}/"
    if args.use_tnn == 0:
        output_dir = f"./results/cifar{args.cifar}/{args.method}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{study_name}_results.txt", "w") as f:
        # write args
        f.write(f"{args}\n")
        f.write(f"{study.best_trial.value}\n")
        f.write(f"{study.best_trial.params}\n")