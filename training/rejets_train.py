import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
import scipy.misc
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

import sys
import os

sys.path.insert(0, os.path.join("..", "utils"))
sys.path.insert(0, os.path.join("..", "models"))
sys.path.insert(0, os.path.join("..", "models"))
from dataset import MyDataset
from modded_basic_nflow import create_NDE_model, train, load_model
from double_flow import LatentFlow
from fake_utils import (
    get_nozero_datasets,
    get_sorted_nozero_datasets,
    get_new_datasets,
    get_newvars_datasets,
)
from args_basic_train import get_args
from validate_rejets import validate_rejets


def main():

    args = get_args()
    print(args)
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    writer = SummaryWriter(logdir=log_dir)
    # save hparams to tensorboard
    writer.add_hparams(vars(args), {})

    # define model
    flow_param_dict = {
        "input_dim": args.zdim,
        "context_dim": args.y_dim,
        "num_flow_steps": args.num_flow_steps,  # increasing this could improve conditioning
        "base_transform_kwargs": {
            "num_transform_blocks": args.num_transform_blocks,  # DNN layers per coupling
            "activation": args.activation,
            "dropout_probability": args.dropout_probability,
            "batch_norm": args.batch_norm,
            "num_bins": args.num_bins,
            "tail_bound": args.tail_bound,
            "hidden_dim": args.hidden_dim,
            "base_transform_type": args.base_transform_type,  # "rq-autoregressive",
            "block_size": args.block_size,  # useless param if we have alternating-binary mask
            "mask_type": args.mask_type,
            "init_identity": args.init_identity,
        },
        "transform_type": args.transform_type,
    }

    model = create_NDE_model(**flow_param_dict)

    if args.device == "cuda":  # Single process, single GPU per process
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            print("!!  USING GPU  !!")

    else:
        print("!!  USING CPU  !!")

    # resume checkpoints
    res_epoch = 0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if args.resume_checkpoint is None and os.path.exists(
        os.path.join(save_dir, "checkpoint-latest.pt")
    ):
        args.resume_checkpoint = os.path.join(
            save_dir, "checkpoint-latest.pt"
        )  # use the latest checkpoint
    if args.resume_checkpoint is not None and args.resume == True:
        model, _, _, res_epoch, _, _ = load_model(
            device,
            model_dir=save_dir,
            filename="checkpoint-latest.pt",
        )
        print(f"Resumed from: {res_epoch}")

    dirpath = os.path.dirname(__file__)

    tr_dataset = MyDataset([os.path.join(dirpath, "datasets", "Ajets_and_muons1+.hdf5")], limit=args.train_limit)    
    te_dataset = MyDataset([os.path.join(dirpath, "datasets", "Ajets_and_muons7+.hdf5")], limit=args.test_limit)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=~args.sorted_dataset,
        num_workers=args.n_load_cores,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        # worker_init_fn=init_np_seed,
    )
    if args.sorted_dataset:
        print("train dataset NOT shuffled")

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=args.batch_size,  # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    print(len(train_loader.dataset))

    trh, tsh = train(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        device=torch.device(args.device),
        name="model",
        model_dir=save_dir,
        args=args,
        writer=writer,
        output_freq=100,
        save_freq=args.save_freq,
        res_epoch=res_epoch,
        val_func=validate_rejets,
    )


if __name__ == "__main__":
    main()
