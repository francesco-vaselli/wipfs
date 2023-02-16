import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt


def validate_latent_flow(
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
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # samples
    with torch.no_grad():

        PU_n_true_int = []

        mod_pt_full = []
        mod_pt_flash = []
        px_full = []
        py_full = []
        px_flash = []
        py_flash = []
        angle_full = []
        angle_flash = []

        for bidx, data in enumerate(test_loader):
            y, z = data[0], data[1]
            # print('x', x.shape, 'y', y.shape, 'N', N.shape)
            inputs_y = y.to(device)
            # print('inputs_y', inputs_y.shape)
            if args.y_dim is not None:
                print(inputs_y.shape[0])
                z_sampled = model.sample(
                        num_samples=1, context=inputs_y
                )
                
            else:
                z_sampled = model.sample(num_samples=args.batch_size)

            z_sampled = z_sampled[0].cpu().detach().numpy()
            print(z_sampled)
            inputs_y = inputs_y.cpu().detach().numpy()
            # print(z_sampled.shape, inputs_y.shape)
            z = z.cpu().detach().numpy()

            z_sampled = z_sampled.reshape(-1, args.z_dim)
            print(z_sampled)
            N_sampled = z_sampled[:, 0]
            if args.y_dim is not None:
                PU_n_true_int.append(inputs_y[:, 2])

            if args.no_N:
                mod_pt_full.append(z[:, 0])
                mod_pt_flash.append(z_sampled[:, 0])
                px_full.append(z[:, 1])
                py_full.append(z[:, 2])
                px_flash.append(z_sampled[:, 1])
                py_flash.append(z_sampled[:, 2])

            # print("done test batch")

    px_full = np.reshape(px_full, (-1, 1)).flatten()
    py_full = np.reshape(py_full, (-1, 1)).flatten()
    px_flash = np.reshape(px_flash, (-1, 1)).flatten()
    py_flash = np.reshape(py_flash, (-1, 1)).flatten()
    mod_pt_full = np.reshape(mod_pt_full, (-1, 1)).flatten()
    mod_pt_flash = np.reshape(mod_pt_flash, (-1, 1)).flatten()
    print(px_full.shape, px_flash.shape)

    arctan = np.arctan2(py_full,px_full).reshape(-1,1)
    arctan_sampled = np.arctan2(py_flash,px_flash).reshape(-1,1)
    sqrt = np.sqrt(px_full**2 + py_full**2).reshape(-1,1)
    sqrt_sampled =  np.sqrt(px_flash**2 + py_flash**2).reshape(-1,1)

    full_sim = [
        mod_pt_full,
        px_full,
        py_full,
        arctan,
        sqrt
    ]
    flash_sim = [
        mod_pt_flash,
        px_flash,
        py_flash,
        arctan_sampled,
        sqrt_sampled
    ]
    names = ["mod_pt", "px", "py", "arctan", "sqrt"]

    
    for i in range(0, len(full_sim)):
        test_values = full_sim[i].flatten()
        generated_sample = flash_sim[i].flatten()
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
    
    if args.y_dim is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        _, xedges, yedges, _ = ax1.hist2d(
            PU_n_true_int,
            mod_pt_full,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin1, right_of_last_bin1 + d1, d1),
            # ],
            cmap="Blues",
            label="FullSim",
        )
        ax1.set_xlabel("PU_n_true_int")
        ax1.set_ylabel("modpt_full")
        ax2.hist2d(
            PU_n_true_int,
            mod_pt_flash,
            # bins=[
            #     np.arange(left_of_first_bin, right_of_last_bin + d, d),
            #     np.arange(left_of_first_bin2, right_of_last_bin2 + d2, d2),
            # ],
            range=[[xedges.min(), xedges.max()], [yedges.min(), yedges.max()]],
            cmap="Reds",
            label="FlashSim Latent",
        )
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel("PU_n_true_int")
        ax2.set_ylabel("modpt_latent")

        fig.suptitle(
            "Comparison of mod pt",
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