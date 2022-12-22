import torch
import random
import numpy as np
import torch.distributed as dist
from math import log, pi
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join("..", "utils"))
from dataset import FakesDataset


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, start_epoch


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(args):
    
    tr_dataset = FakesDataset(["./datasets/fake_jets.hdf5"], x_dim=30, y_dim=6, limit=1000000)
    te_dataset = FakesDataset(["./datasets/fake_jets.hdf5"], x_dim=30, y_dim=6, start=1000000, limit=1200000)

    return tr_dataset, te_dataset


def validate(test_loader, model, epoch, writer, save_dir, args, clf_loaders=None):
    model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, 'epoch-%d' % epoch)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    if args.use_latent_flow:
        with torch.no_grad():

            pts = [[] for _ in range(2)]
            etas = [[] for _ in range(2)]
            phis = [[] for _ in range(2)]
            rpts = [[] for _ in range(2)]
            retas = [[] for _ in range(2)]
            rphis = [[] for _ in range(2)]
            N_true_int = [[] for _ in range(2)]
            N_true_fakes = [[] for _ in range(2)]
            delta_phi_full = []
            delta_phi_flash = []
            for bidx, data in enumerate(test_loader):
                x, y, N = data[0], data[1], data[2]
                # print('x', x.shape, 'y', y.shape, 'N', N.shape)
                inputs_y = y.cuda(args.gpu, non_blocking=True)
                # print('inputs_y', inputs_y.shape)
                z_sampled, x_sampled = model.sample(y=inputs_y, batch_size=None, num_points=1)

                z_sampled = z_sampled.cpu().detach().numpy()
                x_sampled = x_sampled.cpu().detach().numpy()
                inputs_y = inputs_y.cpu().detach().numpy()
                x = x.cpu().detach().numpy()

                x = x.reshape(-1, 30)
                x_sampled = x_sampled.reshape(-1, 30)
                print(x.shape, x_sampled.shape)
                pts = np.concatenate((pts, x[:, :10]), axis=0)
                etas = np.concatenate((etas, x[:, 10:20]), axis=0)
                phis = np.concatenate((phis, x[:, 20:30]), axis=0)
                rpts = np.concatenate((rpts, x_sampled[:, :10]), axis=0)
                retas = np.concatenate((retas, x_sampled[:, 10:20]), axis=0)
                rphis = np.concatenate((rphis, x_sampled[:, 20:30]), axis=0)

                N_true_int = np.concatenate((N_true_int, inputs_y[:, 2]), axis=0)
                N_true_fakes = np.concatenate((N_true_fakes, np.count_nonzero(x_sampled[:, :10]>0)), axis=0)
                print('done 10k')

            # delta_phi_full = np.concatenate((delta_phi_full, np.abs(x[:, 20:30] - inputs_y[:, 0])), axis=0)

        full_sim = [pts, etas, phis, N_true_int]
        flash_sim = [rpts, retas, rphis, N_true_fakes]
        names = ['pt', 'eta', 'phi', 'N_true_int']

        for i in range(0, len(full_sim)):
            test_values = full_sim[i]
            generated_sample = flash_sim[i]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

            _, rangeR, _ = ax1.hist(test_values, histtype='step', label='FullSim', lw=1, bins=100)
            generated_sample = np.where(generated_sample < rangeR.min(), rangeR.min(), generated_sample)
            generated_sample = np.where(generated_sample > rangeR.max(), rangeR.max(), generated_sample)
            ax1.hist(generated_sample, bins=100,  histtype='step', lw=1,
                    range=[rangeR.min(), rangeR.max()], label=f'FlashSim')
            fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
            ax1.legend(frameon=False, loc='upper right')

            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.set_yscale("log")
            ax2.hist(test_values, histtype='step', lw=1, bins=100)
            ax2.hist(generated_sample, bins=100,  histtype='step', lw=1,
                    range=[rangeR.min(), rangeR.max()])
            #ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
            # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
            plt.savefig(f"./figures/comparison_{names[i]}.png")
            plt.close()
