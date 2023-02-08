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
from modded_basic_nflow import create_NDE_model
from double_flow import LatentFlow
from fake_utils import (
    AverageValueMeter,
    save,
    resume,
    init_np_seed,
    reduce_tensor,
    set_random_seed,
    get_simple_datasets,
    get_new_datasets,
    validate_latent_flow,
    validate_simple_flow
)
from args_fake_jets_only_latent import get_args

faulthandler.enable()


def main_worker(gpu, save_dir, ngpus_per_node, args):
    
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
        # save hparams to tensorboard
        writer.add_hparams(vars(args), {})
    else:
        writer = None

    # multi-GPU setup
    latent_model = LatentFlow(args)
    if args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        latent_model = latent_model.cuda(args.gpu)
        device = torch.device("cuda")
    else: 
        print("!!  USING CPU  !!")

    # resume checkpoints
    start_epoch = 0
    optimizer_latent = latent_model.make_optimizer(args)

    if args.resume_latent_checkpoint is None and os.path.exists(
        os.path.join(save_dir, "latent_checkpoint-latest.pt")
    ):
        args.resume_latent_checkpoint = os.path.join(
            save_dir, "latent_checkpoint-latest.pt"
        )  # use the latest checkpoint
    if args.resume_latent_checkpoint is not None:
        if args.resume_latent_optimizer:
            latent_model, optimizer_latent, start_epoch = resume(
                args.resume_latent_checkpoint,
                latent_model,
                optimizer_latent,
                strict=(not args.resume_non_strict),
            )
        else:
            latent_model, _, start_epoch = resume(
                args.resume_latent_checkpoint,
                latent_model,
                optimizer=None,
                strict=(not args.resume_non_strict),
            )
        print("Resumed from: " + args.resume_latent_checkpoint)

    
    # initialize datasets and loaders
    tr_dataset, te_dataset = get_new_datasets(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None) and args.shuffle_train,
        num_workers=args.n_load_cores, # need to find a way to set this automatically
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        # worker_init_fn=init_np_seed,
    )
    if (train_sampler is None) and args.shuffle_train == False:
        print('train dataset NOT shuffled')

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=10000, # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=init_np_seed,
    )

    # initialize the learning rate scheduler
    if args.scheduler == "exponential":
        scheduler_latent = optim.lr_scheduler.ExponentialLR(optimizer_latent, args.exp_decay)
    elif args.scheduler == "step":
        scheduler_latent = optim.lr_scheduler.StepLR(
            optimizer_latent, step_size=args.epochs // 2, gamma=0.1
        )
    elif args.scheduler == "linear":

        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l

        scheduler_latent = optim.lr_scheduler.LambdaLR(optimizer_latent, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # epoch 0 validation
    if args.validate_at_0:
        epoch = 0
        validate_latent_flow(
                test_loader, latent_model, epoch, writer, save_dir, args, device=device, clf_loaders=None
            )
    # main training loop
    start_time = time.time()
    latent_nats_avg_meter = AverageValueMeter()

    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            _, y, z = data[0], data[1], data[2]
            step = bidx + len(train_loader) * epoch
            latent_model.train()

            inputs_y = y.cuda(args.gpu, non_blocking=True)
            inputs_z = z.cuda(args.gpu, non_blocking=True)
            prior_nats = latent_model(inputs_y, inputs_z, optimizer_latent, step, epoch, writer)

            latent_nats_avg_meter.update(prior_nats)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    "TRAIN: [Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] LatentFlowLoss %2.5f"
                    % (
                        args.rank,
                        epoch,
                        bidx,
                        len(train_loader),
                        duration,
                        latent_nats_avg_meter.avg,
                        
                    )
                )

        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler_latent.step()
            if writer is not None:
                writer.add_scalar("lr/optimizer_latent", scheduler_latent.get_last_lr(), epoch)

        if (not args.no_validation and (epoch + 1) % args.val_freq == 0):
            # evaluate on the validation set
            for bidx, data in enumerate(test_loader):
                _, y, z = data[0], data[1], data[2]
                step = bidx + len(test_loader) * epoch
                latent_model.eval()
                
                inputs_y = y.cuda(args.gpu, non_blocking=True)
                inputs_z = z.cuda(args.gpu, non_blocking=True)
                
                prior_nats = latent_model(inputs_y, inputs_z, optimizer_latent, step, epoch, writer, val=True)

                latent_nats_avg_meter.update(prior_nats)
                if step % args.log_freq == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    print(
                        "TEST: [Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] LatentFlowLoss %2.5f"
                        % (
                            args.rank,
                            epoch,
                            bidx,
                            len(test_loader),
                            duration,
                            latent_nats_avg_meter.avg,
                        )
                    )

        if not args.no_validation and (epoch + 1) % args.val_freq == 0:
            validate_latent_flow(
                test_loader, latent_model, epoch, writer, save_dir, args, device=device, clf_loaders=None
            )

        # save checkpoints
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq == 0:
                save(
                    latent_model,
                    optimizer_latent,
                    epoch + 1,
                    os.path.join(save_dir, "latent_checkpoint-%d.pt" % epoch),
                )
                save(
                    latent_model,
                    optimizer_latent,
                    epoch + 1,
                    os.path.join(save_dir, "latent_checkpoint-latest.pt"),
                )


def main():
    # command line args
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "images"))

    with open(os.path.join(save_dir, "command.sh"), "w") as f:
        f.write("python -X faulthandler " + " ".join(sys.argv))
        f.write("\n")

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args)
        )
    else:
        main_worker(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == "__main__":
    main()