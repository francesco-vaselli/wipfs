

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from pathlib import Path
import h5py
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from nflows import distributions, flows, transforms, utils
import nflows.nn.nets as nn_
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join("..", "utils"))
sys.path.insert(0, os.path.join("..", "models"))
from dataset import FakesDataset
from basic_nflow import create_NDE_model
from double_flow import LatentFlow
from fake_utils import (
    AverageValueMeter,
    save,
    resume,
    init_np_seed,
    reduce_tensor,
    set_random_seed,
    get_new_datasets,
    get_simple_datasets,
    validate_latent_flow,
    validate_simple_flow,
)
from args_fake_jets_only_latent import get_args
from tensorboardX import SummaryWriter

# define hyperparams
# total_epochs = 500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_linear_transform(param_dim):
    """Create the composite linear transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_base_transform(
    i,
    param_dim,
    context_dim=None,
    hidden_dim=512,
    num_transform_blocks=2,
    activation="relu",
    dropout_probability=0.0,
    batch_norm=True,
    num_bins=8,
    tail_bound=1.0,
    apply_unconditional_transform=False,
    base_transform_type="rq-coupling",
):
    """Build a base NSF transform of x, conditioned on y.
    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).
    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.
    A coupling flow fixes half the components of x, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.
    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on x_fixed and y. The residual
    network consists of a sequence of two-layer fully-connected blocks.
    Arguments:
        i {int} -- index of transform in sequence
        param_dim {int} -- dimensionality of x
    Keyword Arguments:
        context_dim {int} -- dimensionality of y (default: {None})
        hidden_dim {int} -- number of hidden units per layer (default: {512})
        num_transform_blocks {int} -- number of transform blocks comprising the
                                      transform (default: {2})
        activation {str} -- activation function (default: {'relu'})
        dropout_probability {float} -- probability of dropping out a unit
                                       (default: {0.0})
        batch_norm {bool} -- whether to use batch normalization
                             (default: {False})
        num_bins {int} -- number of bins for the spline (default: {8})
        tail_bound {[type]} -- [description] (default: {1.})
        apply_unconditional_transform {bool} -- whether to apply an
                                                unconditional transform to
                                                fixed components
                                                (default: {False})
        base_transform_type {str} -- type of base transform
                                     ([rq-coupling], rq-autoregressive)
    Returns:
        Transform -- the NSF transform
    """

    if activation == "elu":
        activation_fn = F.elu
    elif activation == "relu":
        activation_fn = F.relu
    elif activation == "leaky_relu":
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.relu  # Default
        print("Invalid activation function specified. Using ReLU.")

    if base_transform_type == "rq-coupling":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(
                lambda in_features, out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_transform_blocks,
                    activation=activation_fn,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm,
                )
            ),
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )

    elif base_transform_type == "rq-autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm,
        )

    else:
        raise ValueError


def create_transform(num_flow_steps, param_dim, context_dim, base_transform_kwargs):
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
    """
        transforms.CompositeTransform([
            create_linear_transform(param_dim),
            create_base_transform(0, param_dim, context_dim=context_dim,
                                  batch_norm=False, **base_transform_kwargs)
        ])] +
        [transforms.CompositeTransform([
            create_linear_transform(param_dim),
            create_base_transform(1, param_dim, context_dim=context_dim,
                                  batch_norm=False, **base_transform_kwargs)
        ])] +
        [
    """
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, **base_transform_kwargs
                    ),
                ]
            )
            for i in range(0, num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )
    return transform


def create_NDE_model(input_dim, context_dim, num_flow_steps, base_transform_kwargs):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.
    This models the posterior distribution p(x|y).
    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y
    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps: should put num_transform_blocks=10,
                          activation='elu',
                          batch_norm=True
    Returns:
        Flow -- the model
    """

    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim, base_transform_kwargs
    )
    flow = flows.Flow(transform, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "num_flow_steps": num_flow_steps,
        "context_dim": context_dim,
        "base_transform_kwargs": base_transform_kwargs,
    }

    return flow


def train_epoch(
    flow,
    train_loader,
    optimizer,
    epoch,
    args,
    device=None,
    output_freq=50,
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

    for batch_idx, (_, y, z) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        # Compute log prob
        loss = -flow.log_prob(z, context=y.view(-1, args.y_dim))

        # Keep track of total loss. w is a weight to be applied to each
        # element.
        train_loss += (loss.detach()).sum()

        # loss = (w * loss).sum() / w.sum()
        loss = (loss).mean()

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

    train_loss = train_loss.item() / len(train_loader.dataset)
    print("Train Epoch: {} \tAverage Loss: {:.4f}".format(epoch, train_loss))

    return train_loss


def test_epoch(flow, test_loader, epoch, args, writer=None, device=None):
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
        for _, y, z in test_loader:

            if device is not None:
                z = z.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Compute log prob
            loss = -flow.log_prob(z, context=y.view(-1, args.y_dim))

            # Keep track of total loss
            test_loss += (loss).sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print("Test set: Average loss: {:.4f}\n".format(test_loss))

        return test_loss


def train(model, train_loader, test_loader, args, save_dir, writer=None, epochs=500, output_freq=100, device=None):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(0, epochs + 1):

        print(
            "Learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"])
        )

        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, args, device=device, output_freq=output_freq
        )
        test_loss = test_epoch(model, test_loader, epoch, args, writer=writer, device=device)

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)

        if epoch % 10 == 0:
            validate_simple_flow(test_loader, model, epoch, writer, save_dir, args, device, clf_loaders=None)
            save_model(
                epoch,
                model,
                scheduler,
                train_history,
                test_history,
                model_dir=save_dir,
            )
            print("saving model")

    return train_history, test_history


def save_model(epoch, model, scheduler, train_history, test_history, model_dir=None):
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

    filename = f"model_fakes_@epoch_{epoch}.pt"

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


def load_model(model_dir=None, filename=None):
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
    model = create_NDE_model(**model_hyperparams)
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


if __name__ == "__main__":

    # define the train and validations datasets
    # initialize datasets and loaders
    # command line args
    args = get_args()
    
    #wirter 
    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    writer = SummaryWriter(logdir=log_dir)
    save_dir = os.path.join("checkpoints", args.log_name)

    tr_dataset, te_dataset = get_simple_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.n_load_cores,
        pin_memory=True,
    )
    if args.shuffle_train == False:
        print('train dataset NOT shuffled')

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=10000, # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=args.n_load_cores,
        pin_memory=True,
    )


    # define additional model parameters
    param_dict = args.latent_flow_param_dict['base_transform_kwargs']
    # {
    #     "num_transform_blocks": 4,
    #     "activation": "relu",
    #     "num_bins": 64,
    #     "hidden_dim": 128,
    #     "batch_norm": True,
    #     "dropout_probability": 0.0,
    # }  # batch norm added

    # create model
    flow = create_NDE_model(args.zdim, args.y_dim, args.zdim, param_dict)

    # print total params number and stuff
    total_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(total_params)
    print(len(train_loader.dataset))

    # set optimizer, send to device and train
    # remember that the model is being saved every 10 epochs
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr_latent)
    flow.to(torch.device(args.device))

    trh, tsh = train(flow, train_loader, test_loader, args, save_dir, writer=writer, epochs=args.epochs, device=torch.device(args.device))