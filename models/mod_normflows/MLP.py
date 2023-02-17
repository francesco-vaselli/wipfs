"""Implementations multi-layer perceptrons."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Good ole multi-layer perceptron."""

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_sizes,
        context_shape=None,
        activation=nn.ReLU(),
        activate_output=False,
        batch_norm=False,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            context_shape: tuple, list or torch.Size, the shape of the context.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
            batch_norm: bool, whether to use batch normalization.
        """
        super().__init__()

        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self.use_batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(in_shape, eps=1e-3) for _ in range(len(hidden_sizes))]
            )

        if context_shape is not None:
            self._context_shape = torch.Size(context_shape)
            self._input_layer = nn.Linear(np.prod(in_shape)+np.prod(context_shape), hidden_sizes[0])
        else:
            self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        # self._context_layer = nn.Linear(np.prod(context_shape), hidden_sizes[0])

        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs, context=None):
        if inputs.shape[1:] != self._in_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self._in_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        
        if context is not None:
            context = context.reshape(-1, np.prod(self._context_shape))
            inputs_cated = torch.cat([inputs, context], dim=1)
            outputs = self._input_layer(inputs_cated)
        else:
            outputs = self._input_layer(inputs)

        outputs = self._activation(outputs)

        if self.use_batch_norm:
            for batch_norm_layer, hidden_layer in zip(
                self.batch_norm_layers, self._hidden_layers
            ):
                outputs = batch_norm_layer(outputs)
                outputs = hidden_layer(outputs)
                outputs = self._activation(outputs)
        else:
            for hidden_layer in self._hidden_layers:
                outputs = hidden_layer(outputs)
                outputs = self._activation(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs