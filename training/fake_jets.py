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
        'zdim': 15,
        'input_dim': 30,
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
            "input_dim" : 16,
            "context_dim" : 6,
            "num_flow_steps" : 8,

            "base_transform_kwargs" : {
            "num_transform_blocks": 5, # DNN layers per coupling
            "activation": "relu",
            "batch_norm": True,
            "num_bins": 16,
            "hidden_dim": 128,
            "block_size": 8,
            "mask_type" : "block-binary"
            },

            "transform_type" : "block-permutation" 
        },
        'reco_flow_param_dict': {
            "input_dim" : 30,
            "context_dim" : 16,
            "num_flow_steps" : 8,

            "base_transform_kwargs" : {
            "num_transform_blocks": 5, # DNN layers per coupling
            "activation": "relu",
            "batch_norm": True,
            "num_bins": 16,
            "hidden_dim": 128,
            "block_size": 10,
            "mask_type" : "block-binary"
            },

            "transform_type" : "block-permutation" 
        },
    }

    # create the model
    model = FakeDoubleFlow(args)

    # print total params number and stuff NOW IN MODEL DEFINITION
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(total_params)

    # define dataset
    train_ds = FakesDataset(["./datasets/fake_jets.hdf5"], x_dim=30, y_dim=6, limit=1000000)
    train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=9
        )
    print(next(iter(train_loader))[0].size(), next(iter(train_loader))[1].size(), next(iter(train_loader))[2].size())