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
from dataset import FakesDataset
from basic_nflow import create_NDE_model
from encoder_double_flow import FakeDoubleFlow
from fake_utils import AverageValueMeter, save, resume, init_np_seed, reduce_tensor, set_random_seed, get_datasets, validate
from args_fake_jets import get_args


if __name__=='__main__':
    args = get_args()
    
    tr_dataset, te_dataset = get_datasets(args)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    model = FakeDoubleFlow(args)
    model = model.cuda()
    model, _, _ = resume('checkpoints/saves_fakes/checkpoint-499.pt', model, strict=False)

    writer = SummaryWriter('checkpoints/saves_fakes')
    validate(test_loader, model, epoch=500, writer, save_dir='.', args, clf_loaders=None)