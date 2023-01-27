import torch
import random
import numpy as np
import torch.distributed as dist
from math import log, pi
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join("..", "utils"))
from dataset import FakesDataset, H5FakesDataset, NewFakesDataset


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
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"], strict=strict)
    start_epoch = ckpt["epoch"]
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
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

    tr_dataset = FakesDataset(
        ["./datasets/train_dataset_fake_jets.hdf5"],
        x_dim=30,
        y_dim=6,
        start=0,
        limit=5000000,
    )
    # H5FakesDataset(
    #     [
    #         "./datasets/fake_jets1.hdf5",
    #         "./datasets/fake_jets2.hdf5",
    #         "./datasets/fake_jets3.hdf5",
    #         "./datasets/fake_jets4.hdf5",
    #         "./datasets/fake_jets5.hdf5",
    #     ],
    #     x_dim=30,
    #     y_dim=6,
    #     limit=5000000,
    # )
    te_dataset = FakesDataset(
        ["./datasets/fake_jets6.hdf5"], x_dim=30, y_dim=6, start=0, limit=100000
    )

    return tr_dataset, te_dataset


def get_new_datasets(args):

    path = "./datasets/train_dataset_fake_jets_only_flows.hdf5"
    if args.rescale_data == True:
        path = "./datasets/train_dataset_fake_jets_only_flow_rescaled.hdf5"

    tr_dataset = NewFakesDataset(
        [path],
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.zdim,
        start=0,
        limit=5000000,
    )
    # H5FakesDataset(
    #     [
    #         "./datasets/fake_jets1.hdf5",
    #         "./datasets/fake_jets2.hdf5",
    #         "./datasets/fake_jets3.hdf5",
    #         "./datasets/fake_jets4.hdf5",
    #         "./datasets/fake_jets5.hdf5",
    #     ],
    #     x_dim=30,
    #     y_dim=6,
    #     limit=5000000,
    # )
    te_dataset = NewFakesDataset(
        [path],
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.zdim,
        start=5000000,
        limit=5100000,
    )

    return tr_dataset, te_dataset


def delta_phi1v9(pts, phis):
    filtered_phi = np.where(pts > 0, phis, np.inf)
    dphi = np.expand_dims(filtered_phi[:, 0], axis=-1) - filtered_phi[:, 1:10]
    dphi.flatten()
    dphi = dphi[np.isfinite(dphi)]

    # constraints the angles in the -pi,pi range
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)

    return dphi


def mod_sum_pt(pts):
    """module sum of pts

    Args:
        pts (np.array): pts array shape [n events, 10 fake jets], 0 for empty jets
    Returns:
        spt: sum of pts [n events]
    """
    spt = np.sum(pts, axis=1)
    spt.flatten()

    return spt


def vec_sum_pt(pts, phis):
    """vector sum of pts

    Args:
        pts (np.array): pts array shape [n events, 10 fake jets], 0 for empty jets
        phis (np.array): phis array shape [n events, 10 fake jets], 0 for empty jets

    Returns:
        spt: vector sum of pts [n events]
        angle: angle of the vector sum of pts [n events]
    """
    px = np.sum(pts * np.cos(phis), axis=1)
    py = np.sum(pts * np.sin(phis), axis=1)

    spt = np.sqrt(px**2 + py**2)
    spt.flatten()

    angle = np.arctan2(py, px)
    angle.flatten()

    return spt, angle


def sum_px_py(pts, phis):
    """sum of pts components

    Args:
        pts (np.array): pts array shape [n events, 10 fake jets], 0 for empty jets
        phis (np.array): phis array shape [n events, 10 fake jets], 0 for empty jets

    Returns:
        px: vector sum of pts on x [n events]
        py: vector sum of pts on y [n events]
    """
    px = np.sum(pts * np.cos(phis), axis=1)
    py = np.sum(pts * np.sin(phis), axis=1)

    px.flatten()
    py.flatten()

    return px, py


def delta_pt(pts):
    filtered_pt = np.where(pts > 0, pts, np.inf)
    dpt = filtered_pt[:, 0] - filtered_pt[:, 1]
    dpt.flatten()
    dpt = dpt[np.isfinite(dpt)]

    return dpt


def sample_double_flow(latent_model, reco_model, y, n_samples, args):
    # Sample from the latent flow
    z = latent_model.sample(n_samples, y)

    # Sample from the reconstruction flow
    x = reco_model.sample(n_samples, z.view(-1, args.zdim))

    return z, x


def validate_double_flow(
    test_loader,
    latent_model,
    reco_model,
    epoch,
    writer,
    save_dir,
    args,
    clf_loaders=None,
):
    latent_model.eval()
    reco_model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    if args.use_latent_flow:
        with torch.no_grad():

            pts = []
            etas = []
            phis = []
            # dphis = []
            rpts = []
            retas = []
            rphis = []
            # rdphis = []
            PU_n_true_int = []
            N_true_fakes_reco = []
            N_true_fakes_latent = []
            N_true_fakes_full = []
            mod_pt_full = []
            mod_pt_flash = []
            px_full = []
            py_full = []
            px_flash = []
            py_flash = []
            # delta_phi_full = []
            # delta_phi_flash = []
            for bidx, data in enumerate(test_loader):
                x, y, z = data[0], data[1], data[2]
                # print('x', x.shape, 'y', y.shape, 'N', N.shape)
                inputs_y = y.cuda(args.gpu, non_blocking=True)
                # print('inputs_y', inputs_y.shape)
                z_sampled, x_sampled = sample_double_flow(
                    latent_model, reco_model, y=inputs_y, n_samples=1, args=args
                )

                z_sampled = z_sampled.cpu().detach().numpy()
                x_sampled = x_sampled.cpu().detach().numpy()
                inputs_y = inputs_y.cpu().detach().numpy()
                x = x.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                N = z[:, 0]

                x = x.reshape(-1, args.x_dim)
                x_sampled = x_sampled.reshape(-1, args.x_dim)
                z_sampled = z_sampled.reshape(-1, args.zdim)
                N_sampled = z_sampled[:, 0]
                print(x.shape, x_sampled.shape)
                pts.append(x[:, :10])
                etas.append(x[:, 10:20])
                phis.append(x[:, 20:30])
                # dphis.append(np.expand_dims(x[:, 20], axis=-1) - x[:, 21:30])
                # dphis.append(delta_phi1v9(x[:, :10], x[:, 20:30]))
                rpts.append(x_sampled[:, :10])
                retas.append(x_sampled[:, 10:20])
                rphis.append(x_sampled[:, 20:30])
                # rdphis.append(np.expand_dims(x_sampled[:, 20], axis=-1) - x_sampled[:, 21:30])
                # rdphis.append(delta_phi1v9(x_sampled[:, :10], x_sampled[:, 20:30]))
                PU_n_true_int.append(inputs_y[:, 2])
                N_true_fakes_latent.append(N_sampled)
                N_true_fakes_reco.append(np.sum(x_sampled[:, :10] > 0, axis=1))
                # N_true_fakes_full.append(np.sum(x[:, :10] > 0, axis=1))
                N_true_fakes_full.append(N)
                mod_pt_full.append(z[:, 1])
                mod_pt_flash.append(z_sampled[:, 1])
                px_full.append(z[:, 2])
                py_full.append(z[:, 3])
                px_flash.append(z_sampled[:, 2])
                py_flash.append(z_sampled[:, 3])

                print("done 10k")

        print(np.shape(pts))
        # delta_phi_full = np.concatenate((delta_phi_full, np.abs(x[:, 20:30] - inputs_y[:, 0])), axis=0)
        pts = np.reshape(pts, (-1, 10))
        etas = np.reshape(etas, (-1, 10))
        phis = np.reshape(phis, (-1, 10))
        dphis = delta_phi1v9(pts, phis)
        delta_pt_full = delta_pt(pts)
        mod_pt_full = np.reshape(mod_pt_full, (-1, 1)).flatten()
        px_full = np.reshape(px_full, (-1, 1)).flatten()
        py_full = np.reshape(py_full, (-1, 1)).flatten()
        # np.reshape(dphis, (-1, 9))
        # constraints the angles in the -pi,pi range
        # dphis = np.where(dphis < np.pi, dphis, dphis - 2*np.pi)
        # dphis = np.where(dphis > -np.pi, dphis, dphis + 2*np.pi)
        rpts = np.reshape(rpts, (-1, 10))
        retas = np.reshape(retas, (-1, 10))
        rphis = np.reshape(rphis, (-1, 10))
        rdphis = delta_phi1v9(rpts, rphis)
        delta_pt_flash = delta_pt(rpts)
        mod_pt_flash = np.reshape(mod_pt_flash, (-1, 1)).flatten()
        px_flash = np.reshape(px_flash, (-1, 1)).flatten()
        py_flash = np.reshape(py_flash, (-1, 1)).flatten()

        PU_n_true_int = np.reshape(PU_n_true_int, (-1, 1)).flatten()
        N_reco = np.reshape(N_true_fakes_reco, (-1, 1)).flatten()
        N_latent = np.rint(np.reshape(N_true_fakes_latent, (-1, 1)).flatten())
        N_full = np.reshape(N_true_fakes_full, (-1, 1)).flatten()
        N_true_fakes_latent = np.reshape(N_true_fakes_latent, (-1, 1)).flatten()
        N_true_fakes_reco = np.reshape(N_true_fakes_reco, (-1, 1)).flatten()
        N_true_fakes_full = np.reshape(N_true_fakes_full, (-1, 1)).flatten()
        print(N_true_fakes_full, N_true_fakes_full.shape)
        full_sim = [
            pts,
            etas,
            phis,
            dphis,
            delta_pt_full,
            N_full,
            N_full,
            mod_pt_full,
            px_full,
            py_full,
        ]
        flash_sim = [
            rpts,
            retas,
            rphis,
            rdphis,
            delta_pt_flash,
            N_reco,
            N_latent,
            mod_pt_flash,
            px_flash,
            py_flash,
        ]
        names = [
            "pt",
            "eta",
            "phi",
            "delta_phi",
            "delta_pt",
            "N_reco",
            "N_latent",
            "mod_pt",
            "px",
            "py",
        ]
        # print(N_true_fakes_latent)

        for i in range(0, len(full_sim)):
            test_values = full_sim[i].flatten()
            generated_sample = flash_sim[i].flatten()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

            _, rangeR, _ = ax1.hist(
                test_values, histtype="step", label="FullSim", lw=1, bins=100
            )
            generated_sample = np.where(
                generated_sample < rangeR.min(), rangeR.min(), generated_sample
            )
            generated_sample = np.where(
                generated_sample > rangeR.max(), rangeR.max(), generated_sample
            )

            if names[i] == "N_true_int":
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim, {np.mean(generated_sample):.2f}",
                )
            else:
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim",
                )
            fig.suptitle(f"Comparison of {names[i]} @ epoch {epoch}", fontsize=16)
            ax1.legend(frameon=False, loc="upper right")

            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.set_yscale("log")
            ax2.hist(test_values, histtype="step", lw=1, bins=100)
            ax2.hist(
                generated_sample,
                bins=100,
                histtype="step",
                lw=1,
                range=[rangeR.min(), rangeR.max()],
            )
            # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
            # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
            # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
            writer.add_figure(f"comparison_{names[i]}", fig, global_step=epoch)
            plt.close()

            if names[i] == "pt":
                test_values_pt = full_sim[i]
                generated_sample_pt = flash_sim[i]

                for j in range(0, 3):
                    test_values = test_values_pt[:, j].flatten()
                    generated_sample = generated_sample_pt[:, j].flatten()
                    fig, (ax1, ax2) = plt.subplots(
                        1, 2, figsize=(9, 4.5), tight_layout=False
                    )

                    _, rangeR, _ = ax1.hist(
                        test_values, histtype="step", label="FullSim", lw=1, bins=100
                    )
                    generated_sample = np.where(
                        generated_sample < rangeR.min(), rangeR.min(), generated_sample
                    )
                    generated_sample = np.where(
                        generated_sample > rangeR.max(), rangeR.max(), generated_sample
                    )
                    ax1.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                        label=f"FlashSim",
                    )
                    fig.suptitle(
                        f"Comparison of Jet_pt{j} @ epoch {epoch}", fontsize=16
                    )
                    ax1.legend(frameon=False, loc="upper right")

                    ax1.spines["right"].set_visible(False)
                    ax1.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    ax2.spines["top"].set_visible(False)
                    ax2.set_yscale("log")
                    ax2.hist(test_values, histtype="step", lw=1, bins=100)
                    ax2.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                    )
                    # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
                    # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
                    # plt.savefig(os.path.join(save_dir, f"comparison_Jet_pt{j}.png"))
                    writer.add_figure(f"comparison_Jet_pt{j}", fig, global_step=epoch)

        # a plt hist2d of N_true_fakes_full vs PU_n_true_int
        # with another hist2 of N_true_fakes_latent vs PU_n_true_int
        # and another hist2 of N_true_fakes_reco vs PU_n_true_int
        # same style as before (lw etc) and labels
        # gen_unique = np.diff(np.unique(PU_n_true_int))
        # full_unique = np.diff(np.unique(N_true_fakes_full))
        # latent_unique = np.diff(np.unique(N_true_fakes_latent))
        # reco_unique = np.diff(np.unique(N_true_fakes_reco))
        # if (
        #     gen_unique.size > 0
        #     and full_unique.size > 0
        #     and latent_unique.size > 0
        #     and reco_unique.size > 0
        # ):

        #     d = gen_unique.min()
        #     left_of_first_bin = PU_n_true_int.min() - float(d) / 2
        #     right_of_last_bin = PU_n_true_int.max() + float(d) / 2

        #     # same for N_true_fakes_full
        #     d1 = full_unique.min()
        #     left_of_first_bin1 = N_true_fakes_full.min() - float(d1) / 2
        #     right_of_last_bin1 = N_true_fakes_full.max() + float(d1) / 2

        #     # same for N_true_fakes_latent
        #     d2 = latent_unique.min()
        #     left_of_first_bin2 = N_true_fakes_latent.min() - float(d2) / 2
        #     right_of_last_bin2 = N_true_fakes_latent.max() + float(d2) / 2

        #     # same for N_true_fakes_reco
        #     d3 = reco_unique.min()
        #     left_of_first_bin3 = N_true_fakes_reco.min() - float(d3) / 2
        #     right_of_last_bin3 = N_true_fakes_reco.max() + float(d3) / 2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4.5), tight_layout=False)
        ax1.hist2d(
            PU_n_true_int,
            N_true_fakes_full,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin1, right_of_last_bin1 + d1, d1),
            # ],
            cmap="Blues",
            label="FullSim",
        )
        ax1.set_xlabel("PU_n_true_int")
        ax1.set_ylabel("N_true_fakes_full")
        ax2.hist2d(
            PU_n_true_int,
            N_true_fakes_latent,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin2, right_of_last_bin2 + d2, d2),
            # ],
            range=[[0, 100], [0, 11]],
            cmap="Reds",
            label="FlashSim Latent",
        )
        ax2.set_ylim([0, 11])
        ax2.set_xlabel("PU_n_true_int")
        ax2.set_ylabel("N_true_fakes_latent")
        ax3.hist2d(
            PU_n_true_int,
            N_true_fakes_reco,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin3, right_of_last_bin3 + d3, d3),
            # ],
            range=[[0, 100], [0, 11]],
            cmap="Greens",
            label="FlashSim Reco",
        )
        ax3.set_ylim([0, 11])
        ax3.set_xlabel("PU_n_true_int")
        ax3.set_ylabel("N_true_fakes_reco")
        fig.suptitle(
            "Comparison of N_true_fakes_full vs N_true_fakes_latent vs N_true_fakes_reco",
            fontsize=16,
        )
        ax1.legend(frameon=False, loc="upper right")
        # plt.savefig(os.path.join(save_dir, f"comparison_N_true_fakes.png"))
        writer.add_figure(
            "comparison_N_true_fakes",
            fig,
            global_step=epoch,
        )
        plt.close()


def validate(test_loader, model, epoch, writer, save_dir, args, clf_loaders=None):
    model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    if args.use_latent_flow:
        with torch.no_grad():

            pts = []
            etas = []
            phis = []
            # dphis = []
            rpts = []
            retas = []
            rphis = []
            # rdphis = []
            PU_n_true_int = []
            N_true_fakes_reco = []
            N_true_fakes_latent = []
            N_true_fakes_full = []
            # delta_phi_full = []
            # delta_phi_flash = []
            for bidx, data in enumerate(test_loader):
                x, y, N = data[0], data[1], data[2]
                # print('x', x.shape, 'y', y.shape, 'N', N.shape)
                inputs_y = y.cuda(args.gpu, non_blocking=True)
                # print('inputs_y', inputs_y.shape)
                z_sampled, x_sampled = model.sample(
                    y=inputs_y, batch_size=None, num_points=1
                )

                z_sampled = z_sampled.cpu().detach().numpy()
                x_sampled = x_sampled.cpu().detach().numpy()
                inputs_y = inputs_y.cpu().detach().numpy()
                x = x.cpu().detach().numpy()
                N = N.cpu().detach().numpy()

                x = x.reshape(-1, 30)
                x_sampled = x_sampled.reshape(-1, 30)
                z_sampled = z_sampled.reshape(-1, 16)
                print(x.shape, x_sampled.shape)
                pts.append(x[:, :10])
                etas.append(x[:, 10:20])
                phis.append(x[:, 20:30])
                # dphis.append(np.expand_dims(x[:, 20], axis=-1) - x[:, 21:30])
                # dphis.append(delta_phi1v9(x[:, :10], x[:, 20:30]))
                rpts.append(x_sampled[:, :10])
                retas.append(x_sampled[:, 10:20])
                rphis.append(x_sampled[:, 20:30])
                # rdphis.append(np.expand_dims(x_sampled[:, 20], axis=-1) - x_sampled[:, 21:30])
                # rdphis.append(delta_phi1v9(x_sampled[:, :10], x_sampled[:, 20:30]))
                PU_n_true_int.append(inputs_y[:, 2])
                N_true_fakes_latent.append(z_sampled[:, 15])
                N_true_fakes_reco.append(np.sum(x_sampled[:, :10] > 0, axis=1))
                # N_true_fakes_full.append(np.sum(x[:, :10] > 0, axis=1))
                N_true_fakes_full.append(N)

                print("done 10k")

        print(np.shape(pts))
        # delta_phi_full = np.concatenate((delta_phi_full, np.abs(x[:, 20:30] - inputs_y[:, 0])), axis=0)
        pts = np.reshape(pts, (-1, 10))
        etas = np.reshape(etas, (-1, 10))
        phis = np.reshape(phis, (-1, 10))
        dphis = delta_phi1v9(pts, phis)
        delta_pt_full = delta_pt(pts)
        # np.reshape(dphis, (-1, 9))
        # constraints the angles in the -pi,pi range
        # dphis = np.where(dphis < np.pi, dphis, dphis - 2*np.pi)
        # dphis = np.where(dphis > -np.pi, dphis, dphis + 2*np.pi)
        rpts = np.reshape(rpts, (-1, 10))
        retas = np.reshape(retas, (-1, 10))
        rphis = np.reshape(rphis, (-1, 10))
        rdphis = delta_phi1v9(rpts, rphis)
        delta_pt_flash = delta_pt(rpts)

        PU_n_true_int = np.reshape(PU_n_true_int, (-1, 1)).flatten()
        N_reco = np.reshape(N_true_fakes_reco, (-1, 1)).flatten()
        N_latent = np.reshape(N_true_fakes_latent, (-1, 1)).flatten()
        N_full = np.reshape(N_true_fakes_full, (-1, 1)).flatten()
        N_true_fakes_latent = np.rint(
            np.reshape(N_true_fakes_latent, (-1, 1)).flatten()
        )
        N_true_fakes_reco = np.rint(np.reshape(N_true_fakes_reco, (-1, 1)).flatten())
        N_true_fakes_full = np.rint(np.reshape(N_true_fakes_full, (-1, 1)).flatten())
        print(N_true_fakes_full, N_true_fakes_full.shape)
        full_sim = [pts, etas, phis, dphis, delta_pt_full, N_full, N_full]
        flash_sim = [rpts, retas, rphis, rdphis, delta_pt_flash, N_reco, N_latent]
        names = ["pt", "eta", "phi", "delta_phi", "delta_pt", "N_reco", "N_latent"]
        print(N_true_fakes_latent)

        for i in range(0, len(full_sim)):
            test_values = full_sim[i].flatten()
            generated_sample = flash_sim[i].flatten()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

            _, rangeR, _ = ax1.hist(
                test_values, histtype="step", label="FullSim", lw=1, bins=100
            )
            generated_sample = np.where(
                generated_sample < rangeR.min(), rangeR.min(), generated_sample
            )
            generated_sample = np.where(
                generated_sample > rangeR.max(), rangeR.max(), generated_sample
            )

            if names[i] == "N_true_int":
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim, {np.mean(generated_sample):.2f}",
                )
            else:
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim",
                )
            fig.suptitle(f"Comparison of {names[i]} @ epoch {epoch}", fontsize=16)
            ax1.legend(frameon=False, loc="upper right")

            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.set_yscale("log")
            ax2.hist(test_values, histtype="step", lw=1, bins=100)
            ax2.hist(
                generated_sample,
                bins=100,
                histtype="step",
                lw=1,
                range=[rangeR.min(), rangeR.max()],
            )
            # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
            # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
            # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
            writer.add_figure(f"comparison_{names[i]}", fig, epoch)
            plt.close()

            if names[i] == "pt":
                test_values_pt = full_sim[i]
                generated_sample_pt = flash_sim[i]

                for j in range(0, 3):
                    test_values = test_values_pt[:, j].flatten()
                    generated_sample = generated_sample_pt[:, j].flatten()
                    fig, (ax1, ax2) = plt.subplots(
                        1, 2, figsize=(9, 4.5), tight_layout=False
                    )

                    _, rangeR, _ = ax1.hist(
                        test_values, histtype="step", label="FullSim", lw=1, bins=100
                    )
                    generated_sample = np.where(
                        generated_sample < rangeR.min(), rangeR.min(), generated_sample
                    )
                    generated_sample = np.where(
                        generated_sample > rangeR.max(), rangeR.max(), generated_sample
                    )
                    ax1.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                        label=f"FlashSim",
                    )
                    fig.suptitle(
                        f"Comparison of Jet_pt{j} @ epoch {epoch}", fontsize=16
                    )
                    ax1.legend(frameon=False, loc="upper right")

                    ax1.spines["right"].set_visible(False)
                    ax1.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    ax2.spines["top"].set_visible(False)
                    ax2.set_yscale("log")
                    ax2.hist(test_values, histtype="step", lw=1, bins=100)
                    ax2.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                    )
                    # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
                    # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
                    # plt.savefig(os.path.join(save_dir, f"comparison_Jet_pt{j}.png"))
                    writer.add_figure(f"comparison_Jet_pt{j}", fig, epoch)

        # a plt hist2d of N_true_fakes_full vs PU_n_true_int
        # with another hist2 of N_true_fakes_latent vs PU_n_true_int
        # and another hist2 of N_true_fakes_reco vs PU_n_true_int
        # same style as before (lw etc) and labels
        gen_unique = np.diff(np.unique(PU_n_true_int))
        full_unique = np.diff(np.unique(N_true_fakes_full))
        latent_unique = np.diff(np.unique(N_true_fakes_latent))
        reco_unique = np.diff(np.unique(N_true_fakes_reco))
        if (
            gen_unique.size > 0
            and full_unique.size > 0
            and latent_unique.size > 0
            and reco_unique.size > 0
        ):

            d = gen_unique.min()
            left_of_first_bin = PU_n_true_int.min() - float(d) / 2
            right_of_last_bin = PU_n_true_int.max() + float(d) / 2

            # same for N_true_fakes_full
            d1 = full_unique.min()
            left_of_first_bin1 = N_true_fakes_full.min() - float(d1) / 2
            right_of_last_bin1 = N_true_fakes_full.max() + float(d1) / 2

            # same for N_true_fakes_latent
            d2 = latent_unique.min()
            left_of_first_bin2 = N_true_fakes_latent.min() - float(d2) / 2
            right_of_last_bin2 = N_true_fakes_latent.max() + float(d2) / 2

            # same for N_true_fakes_reco
            d3 = reco_unique.min()
            left_of_first_bin3 = N_true_fakes_reco.min() - float(d3) / 2
            right_of_last_bin3 = N_true_fakes_reco.max() + float(d3) / 2

            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(9, 4.5), tight_layout=False
            )
            ax1.hist2d(
                PU_n_true_int,
                N_true_fakes_full,
                bins=[
                    np.arange(left_of_first_bin, right_of_last_bin + d, d),
                    np.arange(left_of_first_bin1, right_of_last_bin1 + d1, d1),
                ],
                cmap="Blues",
                label="FullSim",
            )
            ax1.set_xlabel("PU_n_true_int")
            ax1.set_ylabel("N_true_fakes_full")
            ax2.hist2d(
                PU_n_true_int,
                N_true_fakes_latent,
                bins=[
                    np.arange(left_of_first_bin, right_of_last_bin + d, d),
                    np.arange(left_of_first_bin2, right_of_last_bin2 + d2, d2),
                ],
                range=[[0, 100], [0, 11]],
                cmap="Reds",
                label="FlashSim Latent",
            )
            ax2.set_ylim([0, 11])
            ax2.set_xlabel("PU_n_true_int")
            ax2.set_ylabel("N_true_fakes_latent")
            ax3.hist2d(
                PU_n_true_int,
                N_true_fakes_reco,
                bins=[
                    np.arange(left_of_first_bin, right_of_last_bin + d, d),
                    np.arange(left_of_first_bin3, right_of_last_bin3 + d3, d3),
                ],
                range=[[0, 100], [0, 11]],
                cmap="Greens",
                label="FlashSim Reco",
            )
            ax3.set_ylim([0, 11])
            ax3.set_xlabel("PU_n_true_int")
            ax3.set_ylabel("N_true_fakes_reco")
            fig.suptitle(
                "Comparison of N_true_fakes_full vs N_true_fakes_latent vs N_true_fakes_reco",
                fontsize=16,
            )
            ax1.legend(frameon=False, loc="upper right")
            # plt.savefig(os.path.join(save_dir, f"comparison_N_true_fakes.png"))
            writer.add_figure("comparison_N_true_fakes", fig, epoch)
            plt.close()


def validate_latent_flow(
    test_loader,
    latent_model,
    epoch,
    writer,
    save_dir,
    args,
    clf_loaders=None,
):
    latent_model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    if args.use_latent_flow:
        with torch.no_grad():

            PU_n_true_int = []
            N_true_fakes_reco = []
            N_true_fakes_latent = []
            N_true_fakes_full = []
            mod_pt_full = []
            mod_pt_flash = []
            px_full = []
            py_full = []
            px_flash = []
            py_flash = []

            for bidx, data in enumerate(test_loader):
                _, y, z = data[0], data[1], data[2]
                # print('x', x.shape, 'y', y.shape, 'N', N.shape)
                inputs_y = y.cuda(args.gpu, non_blocking=True)
                # print('inputs_y', inputs_y.shape)
                z_sampled = latent_model.sample(num_samples=1, context=inputs_y)

                z_sampled = z_sampled.cpu().detach().numpy()
                inputs_y = inputs_y.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                N = z[:, 0]

                z_sampled = z_sampled.reshape(-1, args.zdim)
                N_sampled = z_sampled[:, 0]

                PU_n_true_int.append(inputs_y[:, 2])
                N_true_fakes_latent.append(N_sampled)
                # N_true_fakes_full.append(np.sum(x[:, :10] > 0, axis=1))
                N_true_fakes_full.append(N)
                mod_pt_full.append(z[:, 1])
                mod_pt_flash.append(z_sampled[:, 1])
                px_full.append(z[:, 2])
                py_full.append(z[:, 3])
                px_flash.append(z_sampled[:, 2])
                py_flash.append(z_sampled[:, 3])

                print("done 10k")


        mod_pt_full = np.reshape(mod_pt_full, (-1, 1)).flatten()
        px_full = np.reshape(px_full, (-1, 1)).flatten()
        py_full = np.reshape(py_full, (-1, 1)).flatten()

        mod_pt_flash = np.reshape(mod_pt_flash, (-1, 1)).flatten()
        px_flash = np.reshape(px_flash, (-1, 1)).flatten()
        py_flash = np.reshape(py_flash, (-1, 1)).flatten()

        PU_n_true_int = np.reshape(PU_n_true_int, (-1, 1)).flatten()
        N_latent = np.reshape(N_true_fakes_latent, (-1, 1)).flatten()
        N_full = np.reshape(N_true_fakes_full, (-1, 1)).flatten()
        N_true_fakes_latent = np.reshape(N_true_fakes_latent, (-1, 1)).flatten()
        N_true_fakes_full = np.reshape(N_true_fakes_full, (-1, 1)).flatten()
        
        full_sim = [
            N_full,
            mod_pt_full,
            px_full,
            py_full,
        ]
        flash_sim = [
            N_latent,
            mod_pt_flash,
            px_flash,
            py_flash,
        ]
        names = [
            "N_latent",
            "mod_pt",
            "px",
            "py",
        ]
        # print(N_true_fakes_latent)

        for i in range(0, len(full_sim)):
            test_values = full_sim[i].flatten()
            generated_sample = flash_sim[i].flatten()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

            _, rangeR, _ = ax1.hist(
                test_values, histtype="step", label="FullSim", lw=1, bins=100
            )
            generated_sample = np.where(
                generated_sample < rangeR.min(), rangeR.min(), generated_sample
            )
            generated_sample = np.where(
                generated_sample > rangeR.max(), rangeR.max(), generated_sample
            )

            if names[i] == "N_true_int":
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim, {np.mean(generated_sample):.2f}",
                )
            else:
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim",
                )
            fig.suptitle(f"Comparison of {names[i]} @ epoch {epoch}", fontsize=16)
            ax1.legend(frameon=False, loc="upper right")

            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.set_yscale("log")
            ax2.hist(test_values, histtype="step", lw=1, bins=100)
            ax2.hist(
                generated_sample,
                bins=100,
                histtype="step",
                lw=1,
                range=[rangeR.min(), rangeR.max()],
            )
            # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
            # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
            # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
            writer.add_figure(f"comparison_{names[i]}", fig, global_step=epoch)
            plt.close()

            if names[i] == "pt":
                test_values_pt = full_sim[i]
                generated_sample_pt = flash_sim[i]

                for j in range(0, 3):
                    test_values = test_values_pt[:, j].flatten()
                    generated_sample = generated_sample_pt[:, j].flatten()
                    fig, (ax1, ax2) = plt.subplots(
                        1, 2, figsize=(9, 4.5), tight_layout=False
                    )

                    _, rangeR, _ = ax1.hist(
                        test_values, histtype="step", label="FullSim", lw=1, bins=100
                    )
                    generated_sample = np.where(
                        generated_sample < rangeR.min(), rangeR.min(), generated_sample
                    )
                    generated_sample = np.where(
                        generated_sample > rangeR.max(), rangeR.max(), generated_sample
                    )
                    ax1.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                        label=f"FlashSim",
                    )
                    fig.suptitle(
                        f"Comparison of Jet_pt{j} @ epoch {epoch}", fontsize=16
                    )
                    ax1.legend(frameon=False, loc="upper right")

                    ax1.spines["right"].set_visible(False)
                    ax1.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    ax2.spines["top"].set_visible(False)
                    ax2.set_yscale("log")
                    ax2.hist(test_values, histtype="step", lw=1, bins=100)
                    ax2.hist(
                        generated_sample,
                        bins=100,
                        histtype="step",
                        lw=1,
                        range=[rangeR.min(), rangeR.max()],
                    )
                    # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
                    # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
                    # plt.savefig(os.path.join(save_dir, f"comparison_Jet_pt{j}.png"))
                    writer.add_figure(f"comparison_Jet_pt{j}", fig, global_step=epoch)

        # a plt hist2d of N_true_fakes_full vs PU_n_true_int
        # with another hist2 of N_true_fakes_latent vs PU_n_true_int
        # and another hist2 of N_true_fakes_reco vs PU_n_true_int
        # same style as before (lw etc) and labels
        # gen_unique = np.diff(np.unique(PU_n_true_int))
        # full_unique = np.diff(np.unique(N_true_fakes_full))
        # latent_unique = np.diff(np.unique(N_true_fakes_latent))
        # reco_unique = np.diff(np.unique(N_true_fakes_reco))
        # if (
        #     gen_unique.size > 0
        #     and full_unique.size > 0
        #     and latent_unique.size > 0
        #     and reco_unique.size > 0
        # ):

        #     d = gen_unique.min()
        #     left_of_first_bin = PU_n_true_int.min() - float(d) / 2
        #     right_of_last_bin = PU_n_true_int.max() + float(d) / 2

        #     # same for N_true_fakes_full
        #     d1 = full_unique.min()
        #     left_of_first_bin1 = N_true_fakes_full.min() - float(d1) / 2
        #     right_of_last_bin1 = N_true_fakes_full.max() + float(d1) / 2

        #     # same for N_true_fakes_latent
        #     d2 = latent_unique.min()
        #     left_of_first_bin2 = N_true_fakes_latent.min() - float(d2) / 2
        #     right_of_last_bin2 = N_true_fakes_latent.max() + float(d2) / 2

        #     # same for N_true_fakes_reco
        #     d3 = reco_unique.min()
        #     left_of_first_bin3 = N_true_fakes_reco.min() - float(d3) / 2
        #     right_of_last_bin3 = N_true_fakes_reco.max() + float(d3) / 2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4.5), tight_layout=False)
        ax1.hist2d(
            PU_n_true_int,
            N_true_fakes_full,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin1, right_of_last_bin1 + d1, d1),
            # ],
            cmap="Blues",
            label="FullSim",
        )
        ax1.set_xlabel("PU_n_true_int")
        ax1.set_ylabel("N_true_fakes_full")
        ax2.hist2d(
            PU_n_true_int,
            N_true_fakes_latent,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin2, right_of_last_bin2 + d2, d2),
            # ],
            range=[[0, 100], [0, 11]],
            cmap="Reds",
            label="FlashSim Latent",
        )
        ax2.set_ylim([0, 11])
        ax2.set_xlabel("PU_n_true_int")
        ax2.set_ylabel("N_true_fakes_latent")
        
        fig.suptitle(
            "Comparison of N_true_fakes_full vs N_true_fakes_latent vs N_true_fakes_reco",
            fontsize=16,
        )
        ax1.legend(frameon=False, loc="upper right")
        # plt.savefig(os.path.join(save_dir, f"comparison_N_true_fakes.png"))
        writer.add_figure(
            "comparison_N_true_fakes",
            fig,
            global_step=epoch,
        )
        plt.close()