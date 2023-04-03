import sys
import os
import time
import numpy as np
import warnings

import torch
from torch.backends import (
    cudnn,
)
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.join("..", "..", "utils"))
sys.path.insert(0, os.path.join("..", "..", "models"))
sys.path.insert(0, "..")

from dataset import MyDataset, NewFakesDataset, MaskAllFakesDataset # x, y, z ordered
from modded_basic_nflow import create_mixture_flow_model, load_model, save_model, create_NDE_model

from args import get_args
from validate_fakes import validate_fakes


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def trainer(gpu, save_dir, ngpus_per_node, args, val_func):

    # basic setup
    cudnn.benchmark = False  # to be tried later
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            # args.dist_url = os.environ["MASTER_ADDR"]
            args.gpu = int(os.environ["LOCAL_RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method= args.dist_url, # dist.TCPStore("r246n05", 29800),# args.dist_url,
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

    # define model, we got maf and arqs parts
    flow_param_dict = {
        "input_dim": args.x_dim,
        "context_dim": args.y_dim,
        "base_kwargs": {
            "num_steps_maf": args.num_steps_maf,
            "num_steps_arqs": args.num_steps_arqs,
            "num_transform_blocks_maf": args.num_transform_blocks_maf,  # DNN layers per coupling
            "num_transform_blocks_arqs": args.num_transform_blocks_arqs,  # DNN layers per coupling
            "activation": args.activation,
            "dropout_probability_maf": args.dropout_probability_maf,
            "dropout_probability_arqs": args.dropout_probability_arqs,
            "use_residual_blocks_maf": args.use_residual_blocks_maf,
            "use_residual_blocks_arqs": args.use_residual_blocks_arqs,
            "batch_norm_maf": args.batch_norm_maf,
            "batch_norm_arqs": args.batch_norm_arqs,
            "num_bins_arqs": args.num_bins,
            "tail_bound_arqs": args.tail_bound,
            "hidden_dim_maf": args.hidden_dim_maf,
            "hidden_dim_arqs": args.hidden_dim_arqs,
            # "base_transform_type": args.base_transform_type,  # "rq-autoregressive",
            # "block_size": args.block_size,  # useless param if we have alternating-binary mask
            # "mask_type": args.mask_type,
            "init_identity": args.init_identity,
            
        },
        "permute_type": args.permute_type,
    }

    model = create_mixture_flow_model(**flow_param_dict)

    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    start_epoch = 0
    if args.resume_checkpoint is None and os.path.exists(
        os.path.join(save_dir, "checkpoint-latest.pt")
    ):
        args.resume_checkpoint = os.path.join(
            save_dir, "checkpoint-latest.pt"
        )  # use the latest checkpoint
    if args.resume_checkpoint is not None and args.resume == True:
        model, _, args.lr, start_epoch, _, _, optimizer_state_dict = load_model(
            model,
            model_dir=save_dir,
            filename="checkpoint-latest.pt",
        )
        print(f"Resumed from: {start_epoch}")

    # multi-GPU setup
    if args.distributed:  # Multiple processes, single GPU per process
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu) # changed into assignment
            ddp_model = DDP(
                model,
                device_ids=[args.gpu],
                output_device=args.gpu,
            )
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
            print("going parallel")
        else:
            assert (
                0
            ), "DistributedDataParallel constructor should always set the single device scope"
    elif args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        ddp_model = model.cuda(args.gpu)
        print("going single gpu")
    else:  # Single process, multiple GPUs per process
        model = model.cuda()
        ddp_model = torch.nn.DataParallel(model)
        print("going multi gpu")

    lr = args.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    if args.resume_checkpoint is not None and args.resume == True:
        optimizer.load_state_dict(optimizer_state_dict)

    dirpath = os.path.dirname(__file__)

    tr_dataset = MaskAllFakesDataset(
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.z_dim,
        h5_paths = [os.path.join(dirpath, "..", "datasets", "full_fake_with_mask_ptmean0.hdf5")],
        limit=args.train_limit,
    )
    te_dataset = MaskAllFakesDataset(
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.z_dim,
        h5_paths =[os.path.join(dirpath, "..", "datasets", "full_fake_with_mask_ptmean0.hdf5")],
        start=args.train_limit,
        limit=args.test_limit,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_load_cores,
        pin_memory=True,
        drop_last=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        worker_init_fn=init_np_seed
        # worker_init_fn=init_np_seed,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=10000,  # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_np_seed,
    )
    print("train size: %d" % len(tr_dataset))
    print("test size: %d" % len(te_dataset))
    print("batch size: %d" % args.batch_size)
    print("len test loader: %d" % len(test_loader))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # main training loop
    start_time = time.time()
    output_freq = 50
    train_history = []
    test_history = []

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        if val_func is not None:
            if args.validate_at_0:
                model.eval()
                val_func(
                    test_loader,
                    model,
                    start_epoch,
                    writer,
                    save_dir,
                    args,
                    args.gpu,
                )
                print('done with validation')

    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if writer is not None:
            writer.add_scalar("lr/optimizer", scheduler.get_last_lr(), epoch)

        # train for one epoch
        train_loss = 0.0
        train_log_p = 0.0
        train_log_det = 0.0
        ddp_model.train()
        for batch_idx, (_, y, z) in enumerate(train_loader):
            optimizer.zero_grad()

            if gpu is not None:
                z = z.cuda(args.gpu, non_blocking=True)
                y = y.cuda(args.gpu, non_blocking=True)

            # Compute log prob
            log_p, log_det = ddp_model(z, context=y)
            loss = -log_p - log_det

            # Keep track of total loss.
            train_loss += (loss.detach()).sum()
            train_log_p += (-log_p.detach()).sum()
            train_log_det += (-log_det.detach()).sum()

            # loss = (w * loss).sum() / w.sum()
            loss = (loss).mean()

            loss.backward()
            optimizer.step()

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    "[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                    % (
                        args.rank,
                        epoch,
                        batch_idx,
                        len(train_loader),
                        duration,
                        loss.item(),
                    )
                )

        train_loss = (train_loss.item() / len(train_loader.dataset)) * args.world_size
        train_log_p = (train_log_p.item() / len(train_loader.dataset)) * args.world_size
        train_log_det = (
            train_log_det.item() / len(train_loader.dataset)
        ) * args.world_size
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/log_p", train_log_p, epoch)
            writer.add_scalar("train/log_det", train_log_det, epoch)
        print(
            "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}, \tAverage log p: {:.4f}, \tAverage log det: {:.4f}".format(
                args.log_name, epoch, train_loss, train_log_p, train_log_det
            )
        )
        # evaluate on the validation set (on just one gpu)
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            with torch.no_grad():
                ddp_model.eval()
                test_loss = 0.0
                test_log_p = 0.0
                test_log_det = 0.0

                for _, y, z in test_loader:

                    if gpu is not None:
                        z = z.cuda(args.gpu, non_blocking=True)
                        y = y.cuda(args.gpu, non_blocking=True)

                    # Compute log prob
                    log_p, log_det = ddp_model(z, context=y)
                    loss = -log_p - log_det

                    # Keep track of total loss.
                    test_loss += (loss.detach()).sum()
                    test_log_p += (-log_p.detach()).sum()
                    test_log_det += (-log_det.detach()).sum()

                test_loss = test_loss.item() / len(test_loader.dataset)
                test_log_p = test_log_p.item() / len(test_loader.dataset)
                test_log_det = test_log_det.item() / len(test_loader.dataset)
                if not args.distributed or (args.rank % ngpus_per_node == 0):
                    writer.add_scalar("test/loss", test_loss, epoch)
                    writer.add_scalar("test/log_p", test_log_p, epoch)
                    writer.add_scalar("test/log_det", test_log_det, epoch)
                # test_loss = test_loss.item() / total_weight.item()
                print(
                    "Test set: Average loss: {:.4f}, \tAverage log p: {:.4f}, \tAverage log det: {:.4f}".format(
                        test_loss, test_log_p, test_log_det
                    )
                )
                train_history.append(train_loss)
                test_history.append(test_loss)

        scheduler.step()
        if epoch % args.val_freq == 0:
            if not args.distributed or (args.rank % ngpus_per_node == 0):
                if val_func is not None:
                    val_func(
                        test_loader,
                        model,
                        epoch,
                        writer,
                        save_dir,
                        args,
                        args.gpu,
                    )
        # save checkpoints
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0:
                save_model(
                    epoch,
                    model,
                    scheduler,
                    train_history,
                    test_history,
                    name="model",
                    model_dir=save_dir,
                    optimizer=optimizer,
                )

    print("done")


def main():
    args = get_args()

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    print("Arguments:")
    print(args)

    args.log_name = args.log_name
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    val_func = validate_fakes
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(
            trainer,
            nprocs=ngpus_per_node,
            args=(save_dir, ngpus_per_node, args, val_func),
        )
    else:
        trainer(args.gpu, save_dir, ngpus_per_node, args, val_func)


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    main()
