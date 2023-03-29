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

        for bid, (_, y, x) in enumerate(test_loader):
            # print(f"Batch {bid} / {len(test_loader)}")
            inputs_y = y.cuda(device)

            while True:
                try:
                    x_sampled = model.sample(
                        num_samples=1, context=inputs_y.view(-1, args.y_dim+args.x_dim)
                    )
                    break
                except AssertionError:
                    print("Sample failed, retrying")
                    pass
            x_sampled = x_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            x_sampled = x_sampled.reshape(-1, 3)
            gen.append(inputs_y[:, :args.y_dim])
            reco.append(x)
            samples.append(x_sampled)
        del inputs_y, x, x_sampled
        torch.cuda.empty_cache()
        print("Done sampling")
    gen = np.array(gen).reshape((-1, args.y_dim))
    full_sim = np.array(reco).reshape((-1, 3))
    flash_sim = np.array(samples).reshape((-1, 3))

    # Samples postprocessing 
    # flash_sim[:, [1, 2]] = flash_sim[:, [1, 2]] * 200
    # full_sim[:, [1, 2]] = full_sim[:, [1, 2]] * 200
    # flash_sim[:, :10] = flash_sim[:, 4:14] * 200
    # full_sim[:, :10] = full_sim[:, 4:14] * 200

    # Plots
    # PU_n_true_int = gen[:, 2]
    # N_true_fakes_full = full_sim[:, 0]
    # N_true_fakes_flash = flash_sim[:, 0]

    N_sel = np.array(gen[:, 6]).flatten()
    print(N_sel)
    names = np.array([[f"pt{i}", f"eta{i}", f"phi{i}"]  for i in range(0, 2)]).flatten()


    n_ids = np.array([[i, i, i]  for i in range(1, 3)]).flatten()


    for i in range(0, len(names)):

        test_values = full_sim[:, i].flatten()[N_sel >= n_ids[i]]
        generated_sample = flash_sim[:, i].flatten()[N_sel >= n_ids[i]]
        ws = wasserstein_distance(test_values, generated_sample)
        print(generated_sample.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

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
