import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from train.training.train_utils import train, evaluate, train_lnp, evaluate_lnp, train_lnp_multimodal, evaluate_lnp_multimodal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    num_params = count_parameters(model)
    print("Number of parameters:", num_params)

    assert 0 <= float(alpha) <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start RFT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} RFT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        wandb.log({
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    wandb.log({})
    print()

def train_eval_loop_multimodal(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    save_freq: int = 1,
    linear_output: bool = False,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)

    num_params = count_parameters(model)
    print("Number of parameters:", num_params)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_lnp_multimodal(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        if epoch % save_freq == 0:
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(project_folder, f"ema_latest.pth")
            print(f"Saved EMA model to {numbered_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # save optimizer
            numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_lnp_multimodal(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                    linear_output=linear_output,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})

def train_eval_loop_lnp(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    save_freq: int = 1,
    linear_output: bool = False,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)

    num_params = count_parameters(model)
    print("Number of parameters:", num_params)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_lnp(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        if epoch % save_freq == 0:
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(project_folder, f"ema_latest.pth")
            print(f"Saved EMA model to {numbered_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # save optimizer
            numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_lnp(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                    linear_output=linear_output,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})

def train_eval_loop_lnp_clip(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    save_freq: int = 1,
    linear_output: bool = False,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75) #[TODO] for diffusion model
        
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_lnp_clip(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
                linear_output=linear_output,
            )
            lr_scheduler.step()

        if epoch % save_freq == 0:
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(project_folder, f"ema_latest.pth")
            print(f"Saved EMA model to {numbered_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(model.state_dict(), numbered_path)
            torch.save(model.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # save optimizer
            numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_lnp_clip(
                    eval_type=dataset_type,
                    #ema_model=ema_model,
                    ema_model=model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                    linear_output=linear_output,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "lnp":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params