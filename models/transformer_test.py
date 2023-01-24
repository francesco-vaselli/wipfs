import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn

from torch.nn import Transformer

# maybe I need just an encoder  
class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        custom_encoder=None,
        custom_decoder=None,
        layer_norm_eps=1e-05,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
        n_linear=1,
        dim_linear=512,
        input_size=512,
        output_size=512,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.device = device
        self.dtype = dtype
        self.n_linear = n_linear
        self.dim_linear = dim_linear
        self.input_size = input_size
        self.output_size = output_size
        # src: (S,E)(S,E) for unbatched input, (S,N,E)(S,N,E) if batch_first=False or (N, S, E) if batch_first=True.
        # tgt: (T,E)(T,E) for unbatched input, (T,N,E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.
        # output: (T,E)(T,E) for unbatched input, (T,N,E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.
        self.transformer = Transformer(
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.custom_encoder,
            self.custom_decoder,
            self.layer_norm_eps,
            self.batch_first,
            self.norm_first,
            self.device,
            self.dtype,
        )

        self.first_linear = nn.Linear(self.d_model, self.dim_linear)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.dim_linear, self.dim_linear) for _ in range(self.n_linear-2)])
