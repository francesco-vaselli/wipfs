import torch
from torch.utils.data import DataLoader
from torch import optim

import sys
import os

sys.path.insert(0, os.path.join("..", "utils"))
sys.path.insert(0, os.path.join("..", "models"))
from dataset import FakesDataset
from basic_nflow import create_NDE_model
from encoder_double_flow import FakeDoubleFlow

BATCH_SIZE = 2048


if __name__ == "__main__":
    # the args dictionary defining all the parameters for the FakeDoubleFlow model
    args = {
        'distributed' : False,
        'zdim': 128,
        'input_dim': 3,
        'optimizer': 'adam',
        'lr': 0.001,
        'weight_decay': 0.0,
        'beta1': 0.9,
        'beta2': 0.999,
        'entropy_weight': 1.0,
        'prior_weight': 1.0,
        'recon_weight': 1.0,
        'use_deterministic_encoder': False,
        'use_latent_flow': True,
        'latent_flow_param_dict': {
            "input_dim" : 17,
            "context_dim" : 14,
            "num_flow_steps" : 15,

            "base_transform_kwargs" : {
            "num_transform_blocks": 10,
            "activation": "relu",
            "batch_norm": True,
            "num_bins": 128,
            "hidden_dim": 298,
            "block_size": 3,
            "mask_type" : "identity"
            },

            "transform_type" : "no-permutation" 
        },
        'reco_flow_param_dict': {
                    "input_dim" : 17,
            "context_dim" : 14,
            "num_flow_steps" : 15,

            "base_transform_kwargs" : {
            "num_transform_blocks": 10,
            "activation": "relu",
            "batch_norm": True,
            "num_bins": 128,
            "hidden_dim": 298,
            "block_size": 3,
            "mask_type" : "identity"
            },

            "transform_type" : "no-permutation" 
        },
    }

    # create the model
    model = FakeDoubleFlow(args)

    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    # define dataset
    train_ds = FakesDataset(["./datasets/fake_jets.hdf5"], x_dim=30, y_dim=6, limit=1000000)
    train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=9
        )
    print(train_loader.next())