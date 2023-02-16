import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

import normflows as nf

from pathlib import Path
import sys
import os

from .context_LUL import ContextLULinearPermute
from .context_nsf import (
    ContextCoupledRationalQuadraticSpline,
    ContextAutoregressiveRationalQuadraticSpline,
)
from .context_flow import ContextNormalizingFlow


def create_model(
    num_splines,
    num_input_channels,
    num_hidden_channels,
    num_blocks,
    transform_type="rq-coupling",
    num_context_channels=None,
    num_bins=8,
    tails="linear",
    tail_bound=3.0,
    activation=nn.ReLU,
    dropout_probability=0.0,
    reverse_mask=False,
    permute_mask=False,
    init_identity=True,
    batch_norm=False,
):
    """Build a sequence of NSF transforms, which maps parameters x into the
    base distribution u (noise). Transforms are conditioned on strain data y.
    Note that the forward map is f^{-1}(x, y).
    Each step in the sequence consists of
        * A linear transform of x, which in particular permutes components
        * A NSF transform of x, conditioned on y.
    There is one final linear transform at the end.
    This function was adapted from the uci.py example in
    https://github.com/bayesiains/nsf
    Arguments:
        num_flow_steps {int} -- number of transforms in sequence
        param_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        base_transform_kwargs {dict} -- hyperparameters for NSF step
    Returns:
        Transform -- the constructed transform
    """
    if transform_type == "rq-coupling":
        selected_flow = ContextCoupledRationalQuadraticSpline(
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            num_context_channels=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            activation=activation,
            dropout_probability=dropout_probability,
            reverse_mask=reverse_mask,
            init_identity=init_identity,
            batch_norm=batch_norm,
        )

    elif transform_type == "rq-autoregressive":
        selected_flow = ContextAutoregressiveRationalQuadraticSpline(
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            num_context_channels=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            activation=activation,
            dropout_probability=dropout_probability,
            permute_mask=permute_mask,
            init_identity=init_identity,
            batch_norm=batch_norm,
        )

    else:
        raise ValueError

    flows = []
    for i in range(num_splines):
        flows += [selected_flow]
        flows += [ContextLULinearPermute(num_input_channels)]

    q0 = nf.distributions.DiagGaussian(num_input_channels, trainable=False)
    flow = ContextNormalizingFlow(q0=q0, flows=flows)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.
    flow.model_hyperparams = {
        "num_splines": num_splines,
        "num_input_channels": num_input_channels,
        "num_blocks": num_blocks,
        "num_hidden_channels": num_hidden_channels,
        "transform_type": transform_type,
        "num_context_channels": num_context_channels,
        "num_bins": num_bins,
        "tails": tails,
        "tail_bound": tail_bound,
        "activation": activation,
        "dropout_probability": dropout_probability,
        "reverse_mask": reverse_mask,
        "permute_mask": permute_mask,
        "init_identity": init_identity,
        "batch_norm": batch_norm,
    }

    return flow


def train_epoch(
    flow,
    train_loader,
    optimizer,
    epoch,
    device=None,
    output_freq=50,
    args=None,
    add_noise=True,
    annealing=False,
):
    """Train model for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})
    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0
    train_log_p = 0.0
    train_log_det = 0.0

    for batch_idx, (y, z) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            z = z.to(device, non_blocking=True)
            if args.y_dim is not None:
                y = y.to(device, non_blocking=True)

        # Compute log prob
        log_p, log_det = flow.forward_kld(z, context=y)
        loss = log_p + log_det

        # Keep track of total loss. w is a weight to be applied to each
        # element.
        train_loss += (loss.detach()).sum()
        train_log_p += (log_p.detach()).sum()
        train_log_det += (log_det.detach()).sum()

        loss = (loss).mean()
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    train_loss = train_loss.item() * args.batch_size / len(train_loader.dataset)
    print(
        "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}".format(
            args.log_name, epoch, train_loss
        )
    )

    return train_loss, train_log_p, train_log_det


def test_epoch(flow, test_loader, epoch, args, device=None):
    """Calculate test loss for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        test_loader {DataLoader} -- test set data loader
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPu) (default: {None})
    Returns:
        float -- test loss
    """

    with torch.no_grad():
        flow.eval()
        test_loss = 0.0
        test_log_p = 0.0
        test_log_det = 0.0
        for y, z in test_loader:

            if device is not None:
                z = z.to(device, non_blocking=True)
                if args.y_dim is not None:
                    y = y.to(device, non_blocking=True)

            # Compute log prob
            log_p, log_det = flow.forward_kld(z, context=y)
            loss = log_p + log_det

            # Keep track of total loss
            test_loss += (loss).sum()
            test_log_p += (log_p).sum()
            test_log_det += (log_det).sum()

        test_loss = test_loss.item() * args.batch_size / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print("Test set: Average loss: {:.4f}\n".format(test_loss))

        return test_loss, test_log_p, test_log_det


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    device,
    name,
    model_dir,
    args,
    writer=None,
    output_freq=100,
    save_freq=10,
    res_epoch=0,
    val_func=None,
):
    """Train the model.
    Args:
            epochs:     number of epochs to train for
            output_freq:    how many iterations between outputs
    """
    train_history = []
    test_history = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    for epoch in range(0 + res_epoch, epochs + 1 + res_epoch):

        print(
            "Learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"])
        )

        train_loss, train_log_p, train_log_det = train_epoch(
            model, train_loader, optimizer, epoch, device, output_freq, args=args
        )
        test_loss, test_log_p, test_log_det = test_epoch(model, test_loader, epoch, args, device)

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("train/log_p", train_log_p, epoch)
            writer.add_scalar("test/log_p", test_log_p, epoch)
            writer.add_scalar("train/log_det", train_log_det, epoch)
            writer.add_scalar("test/log_det", test_log_det, epoch)

        if epoch % args.val_freq == 0:
            val_func(
                test_loader,
                model,
                epoch,
                writer,
                save_dir=args.log_name,
                args=args,
                device=args.device,
                clf_loaders=None,
            )

        if epoch % save_freq == 0:

            save_model(
                epoch,
                model,
                scheduler,
                train_history,
                test_history,
                name,
                model_dir=model_dir,
                optimizer=optimizer,
            )
            print("saving model")

    return train_history, test_history


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.model_hyperparams,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def load_model(device, model_dir=None, filename=None):
    """Load a saved model.
    Args:
        filename:       File name
    """

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location=device)

    model_hyperparams = checkpoint["model_hyperparams"]
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["optimizer_state_dict"]["param_groups"][-1]["initial_lr"]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )
