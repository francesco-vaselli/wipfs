import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.lines as mlines
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance
import pandas as pd


def validate_fakes(
    test_loader,
    model,
    epoch,
    writer,
    save_dir,
    args,
    device,
    clf_loaders=None,
):
    model.eval()


    # Make epoch wise save directory
    if writer is not None:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    with torch.no_grad():

        gen = []
        reco = []
        samples = []

        for bid, (_, y, z) in enumerate(test_loader):

            inputs_y = y.cuda(device)

            z_sampled = model.sample(
                    num_samples=1, context=inputs_y.view(-1, args.y_dim)
                )
            z_sampled = z_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            z_sampled = z_sampled.reshape(-1, args.z_dim)
            gen.append(inputs_y)
            reco.append(z)
            samples.append(z_sampled)

    gen = np.array(gen).reshape((-1, args.y_dim))
    full_sim = np.array(reco).reshape((-1, args.z_dim))
    flash_sim = np.array(samples).reshape((-1, args.z_dim))

    # Samples postprocessing 
    # flash_sim[:, [1, 2]] = flash_sim[:, [1, 2]] * 200
    # full_sim[:, [1, 2]] = full_sim[:, [1, 2]] * 200
    flash_sim[:, :10] = flash_sim[:, 4:14] * 200
    full_sim[:, :10] = full_sim[:, 4:14] * 200

    # Plots
    PU_n_true_int = gen[:, 2]
    N_true_fakes_full = full_sim[:, 0]
    N_true_fakes_flash = flash_sim[:, 0]

    names0 = ["N"]
    names1 = np.array([[f"pt{i}", f"eta{i}", f"phi{i}"]  for i in range(0, 10)]).flatten()
    n_ids = np.array([[i, i, i]  for i in range(1, 11)]).flatten()
    names = np.hstack((names0, names1))


    bins_N = np.arange(-0.1, 1.1, step=0.1) - 0.05
    N_sel = np.rint(N_true_fakes_flash*10).astype(int)


    for i in range(0, len(names)):
        if i > 0:
            test_values = full_sim[:, i].flatten()[N_sel <= n_ids[i-1]]
            generated_sample = flash_sim[:, i].flatten()[N_sel <= n_ids[i-1]]
        else:
            test_values = N_true_fakes_full.flatten()
            generated_sample = N_true_fakes_flash.flatten()
            ws = wasserstein_distance(test_values, generated_sample)
            print(generated_sample.shape)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

            if i == 0:
                _, rangeR, _ = ax1.hist(
                    test_values,
                    histtype="step",
                    label="FullSim",
                    lw=1,
                    bins=bins_N,
                )
            else:
                _, rangeR, _ = ax1.hist(
                    test_values, histtype="step", label="FullSim", lw=1, bins=100
                )
                print(rangeR.shape)
                generated_sample = np.where(
                    generated_sample < rangeR.min(), rangeR.min(), generated_sample
                )
                generated_sample = np.where(
                    generated_sample > rangeR.max(), rangeR.max(), generated_sample
                )

            if i == 0:
                ax1.hist(
                    generated_sample,
                    bins=bins_N,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim, ws={round(ws, 4)}",
                )
            else:
                ax1.hist(
                    generated_sample,
                    bins=100,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                    label=f"FlashSim, ws={round(ws, 4)}",
                )
            fig.suptitle(f"Comparison of {names[i]} @ epoch {epoch}", fontsize=16)
            ax1.legend(frameon=False, loc="upper right")

            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.set_yscale("log")
            if i == 0:
                ax2.hist(test_values, histtype="step", lw=1, bins=bins_N)
                ax2.hist(
                    generated_sample,
                    bins=bins_N,
                    histtype="step",
                    lw=1,
                    range=[rangeR.min(), rangeR.max()],
                )
            else:
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
            writer.add_scalar(f"ws/{names[i]}_wasserstein_distance", ws, global_step=epoch)
            plt.close()

            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
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
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax2.hist2d(
        PU_n_true_int,
        N_true_fakes_flash,
        # bins=[
        #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
        #     np.arange(left_of_first_bin2, right_of_last_bin2 + d2, d2),
        # ],
        range=[[x_min, x_max], [y_min, y_max]],
        cmap="Reds",
        label="FlashSim Latent",
    )
    ax2.set_ylim(ax1.get_ylim())
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

    # Corner plot

    