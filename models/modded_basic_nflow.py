import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from nflows import distributions, flows, transforms, utils
import nflows.nn.nets as nn_

from pathlib import Path
import sys
import os

dirpath = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(dirpath, "..", "utils"))
from masks import create_block_binary_mask, create_identity_mask
from permutations import BlockPermutation, IdentityPermutation

from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import (AutoregressiveTransform)
from nflows.transforms import made as made_module
from nflows.transforms.splines.cubic import cubic_spline
from nflows.transforms.splines.linear import linear_spline
from nflows.transforms.splines.quadratic import (
    quadratic_spline,
    unconstrained_quadratic_spline,
)
# from nflows.transforms.splines import rational_quadratic
# from nflows.transforms.splines.rational_quadratic import (
#    rational_quadratic_spline,
#    unconstrained_rational_quadratic_spline,)

from modded_spline import unconstrained_rational_quadratic_spline, rational_quadratic_spline
import modded_spline
from nflows.utils import torchutils
# from nflows.transforms import splines
from torch.nn.functional import softplus

from modded_coupling import PiecewiseCouplingTransformM
from modded_base_flow import FlowM


class MaskedAffineAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity = True
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        self.init_identity = init_identity
        if init_identity:
          torch.nn.init.constant_(made.final_layer.weight, 0.0)
          torch.nn.init.constant_(
              made.final_layer.bias,
              0.5414 # the value k to get softplus(k) = 1.0
          )

        super(MaskedAffineAutoregressiveTransformM, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        # print(scale, shift)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        if self.init_identity:
            shift = shift - 0.5414
        # print(unconstrained_scale, shift)
        return unconstrained_scale, shift


class MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        if init_identity:
          torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
          torch.nn.init.constant_(
              autoregressive_net.final_layer.bias,
              np.log(np.exp(1 - min_derivative) - 1),
          )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class PiecewiseRationalQuadraticCouplingTransformM(PiecewiseCouplingTransformM):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )


def create_random_transform(param_dim):
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


def create_block_transform(param_dim, block_size):
    """Create the composite block transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform(
        [
            BlockPermutation(features=param_dim, block_size=block_size),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_identity_transform(param_dim):
    """Create the composite block transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """
    return transforms.CompositeTransform(
        [
            IdentityPermutation(features=param_dim),
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
    batch_norm=False,
    num_bins=8,
    tail_bound=3.0, # new value also passed in args
    apply_unconditional_transform=False,
    base_transform_type="rq-coupling",
    mask_type="block-binary",
    block_size=1,
    init_identity=True,
):
    """
    NOTE: we are now using block masking
    Build a base NSF transform of x, conditioned on y.
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

    if mask_type == "block-binary":
        mask = create_block_binary_mask(param_dim, block_size)
    elif mask_type == "alternating-binary":
        mask = utils.create_alternating_binary_mask(param_dim, even=(i % 2 == 0))
    elif mask_type == "identity":
        mask = create_identity_mask(param_dim)
    else:
        raise ValueError

    if base_transform_type == "rq-coupling":
        return PiecewiseRationalQuadraticCouplingTransformM(
            mask=mask,
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
            init_identity=init_identity
        )

    elif base_transform_type == "rq-autoregressive":
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(
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
            init_identity=init_identity # modded version with init_identity
        )

    else:
        raise ValueError


def create_transform(
    num_flow_steps,
    param_dim,
    context_dim,
    base_transform_kwargs,
    transform_type,
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

    if transform_type == "block-permutation":
        block_size = base_transform_kwargs["block_size"]
        selected_transform = create_block_transform(param_dim, block_size)
    elif transform_type == "random-permutation":
        selected_transform = create_random_transform(param_dim)
    elif transform_type == "no-permutation":
        selected_transform = create_identity_transform(param_dim)
    else:
        raise ValueError

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    # selected_transform,
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, **base_transform_kwargs
                    ),
                    selected_transform
                ]
            )
            for i in range(num_flow_steps)
        ]
        # + [transforms.LULinear(param_dim, identity_init=True)]
    )
    return transform


def create_NDE_model(
    input_dim, context_dim, num_flow_steps, base_transform_kwargs, transform_type
):
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
        num_flow_steps, input_dim, context_dim, base_transform_kwargs, transform_type
    )
    flow = FlowM(transform, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "num_flow_steps": num_flow_steps,
        "context_dim": context_dim,
        "base_transform_kwargs": base_transform_kwargs,
        "transform_type": transform_type,
    }

    return flow


def create_mixture_flow_model(
    input_dim, context_dim, base_kwargs, transform_type
):
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
    transform = []
    for _ in range(base_kwargs["num_steps_maf"]):
        transform.append(
            MaskedAffineAutoregressiveTransformM(
                features=input_dim,
                use_residual_blocks=base_kwargs["use_residual_blocks_maf"],
                num_blocks=base_kwargs["num_transform_blocks_maf"],
                hidden_features=base_kwargs["hidden_dim_maf"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_maf"],
                use_batch_norm=base_kwargs["batch_norm_maf"],
            )
        )
        transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["num_steps_arqs"]):
        transform.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(
                features=input_dim,
                tails="linear",
                use_residual_blocks=base_kwargs["use_residual_blocks_arqs"],
                hidden_features=base_kwargs["hidden_dim_arqs"],
                num_blocks=base_kwargs["num_transform_blocks_arqs"],
                tail_bound=base_kwargs["tail_bound_arqs"],
                num_bins=base_kwargs["num_bins_arqs"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_arqs"],
                use_batch_norm=base_kwargs["batch_norm_arqs"],
            )
        )
        transform.append(create_random_transform(param_dim=input_dim))

    transform_fnal = CompositeTransform(transform)

    flow = FlowM(transform_fnal, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": base_kwargs,
        "transform_type": transform_type,
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

    for batch_idx, (z, y) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        # Compute log prob
        loss = -flow.log_prob(z, context=y)

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
    print("Model:{} Train Epoch: {} \tAverage Loss: {:.4f}".format(args.log_name, epoch, train_loss))

    return train_loss


def test_epoch(flow, test_loader, epoch, device=None):
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
        for z, y in test_loader:

            if device is not None:
                z = z.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Compute log prob
            loss = -flow.log_prob(z, context=y)

            # Keep track of total loss
            test_loss += (loss).sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print("Test set: Average loss: {:.4f}\n".format(test_loss))

        return test_loss


def train(model, train_loader, test_loader, epochs, optimizer, device, name, model_dir, args, writer=None, output_freq=100, save_freq=10, res_epoch=0, val_func=None):
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

    for epoch in range(0+res_epoch, epochs + 1 + res_epoch):

        print(
            "Learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"])
        )

        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, device, output_freq, args=args
        )
        test_loss = test_epoch(model, test_loader, epoch, device)

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)

        if epoch % args.val_freq == 0:
            val_func(
                test_loader, model, epoch, writer, save_dir=args.log_name, args=args, device=args.device, clf_loaders=None
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
                optimizer=optimizer
            )
            print("saving model")

    return train_history, test_history


def save_model(epoch, model, scheduler, train_history, test_history, name, model_dir=None, optimizer=None):
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
    resume_filename = 'checkpoint-latest.pt'

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
        dict["last_lr"] = scheduler.get_last_lr()

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
    checkpoint = torch.load(p / filename, map_location="cpu")

    model_hyperparams = checkpoint["model_hyperparams"]
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_NDE_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
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


def load_mixture_model(device, model_dir=None, filename=None):
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
    checkpoint = torch.load(p / filename, map_location="cpu")

    model_hyperparams = checkpoint["model_hyperparams"]
    # added because of a bug in the old create_mixture_flow_model function
    try:
        if checkpoint["model_hyperparams"]["base_transform_kwargs"] is not None:
            checkpoint["model_hyperparams"]["base_kwargs"] = checkpoint["model_hyperparams"]["base_transform_kwargs"]
            del checkpoint["model_hyperparams"]["base_transform_kwargs"]
    except KeyError:
        pass
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_mixture_flow_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
    elif checkpoint["last_lr"] is not None:
        flow_lr = checkpoint["last_lr"][0]
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