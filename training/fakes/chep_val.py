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
import mplhep as hep
import matplotlib.lines as mlines


def D_phi1v9(N, phis):
    filtered_phi = phis
    for i in range(0, 10):
        filtered_phi[:, i][N <= i+1] = np.nan
    dphi = np.expand_dims(filtered_phi[:, 0], axis=-1) - filtered_phi[:, 1:10]
    dphi = dphi.reshape(-1, 9)
    # constraints the angles in the -pi,pi range
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    # print(np.isnan(dphi).any())
    # dphi = np.where(dphi == np.nan, -5, dphi)

    return dphi


def delta_phi1v9(pts, phis):
    filtered_phi = np.where(pts > np.log(15)-3, phis, np.nan)
    dphi = np.expand_dims(filtered_phi[:, 0], axis=-1) - filtered_phi[:, 1:10]
    dphi = dphi.reshape(-1, 9)
    # constraints the angles in the -pi,pi range
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    # print(np.isnan(dphi).any())
    # dphi = np.where(dphi == np.nan, -5, dphi)

    return dphi


def delta_pt1v9(pts_ref, pts):
    filtered_pts = np.where(pts_ref > np.log(15)-3, np.exp(pts+3), np.nan)
    dpt = np.expand_dims(filtered_pts[:, 0], axis=-1) - filtered_pts[:, 1:10]
    dpt = dpt.reshape(-1, 9)

    return dpt


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
    hep.style.use("CMS")

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
            x_sampled = x_sampled.reshape(-1, args.x_dim)
            gen.append(inputs_y[:, :args.y_dim])
            reco.append(x)
            samples.append(x_sampled)
        del inputs_y, x, x_sampled
        torch.cuda.empty_cache()
        print("Done sampling")
    gen = np.array(gen).reshape((-1, args.y_dim))
    full_sim = np.array(reco).reshape((-1, args.x_dim))
    flash_sim = np.array(samples).reshape((-1, args.x_dim))

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
    names = np.array([[f"pt{i}", f"eta{i}", f"phi{i}"]  for i in range(0, 10)]).flatten()
    full_df = pd.DataFrame(full_sim, columns=names)
    flash_df = pd.DataFrame(flash_sim, columns=names)

    pts = full_sim[:, 0::3]
    etas = full_sim[:, 1::3]
    phis = full_sim[:, 2::3]
    pts_flash = flash_sim[:, 0::3]
    etas_flash = flash_sim[:, 1::3]
    phis_flash = flash_sim[:, 2::3]

    dphi = delta_phi1v9(pts, phis)
    dphi_flash = delta_phi1v9(pts, phis_flash) # using full sim pt as reference of n_jets. should adjust to N_sel
    dpt = delta_pt1v9(pts, pts)
    dpt_flash = delta_pt1v9(pts, pts_flash)

    # postprocess pts
    pts = np.exp(pts+3)
    pts_flash = np.exp(pts_flash+3)
    
    n_ids = np.array([[i, i, i]  for i in range(1, 11)]).flatten()


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
        plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}_{epoch}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}_{epoch}.pdf"))
        writer.add_figure(f"comparison_{names[i]}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{names[i]}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    n_pt = np.arange(1, 11)
    for i in range(0, len(n_pt)):

        test_values = pts[:, i].flatten()[N_sel >= n_pt[i]]
        generated_sample = pts_flash[:, i].flatten()[N_sel >= n_pt[i]]
        ws = wasserstein_distance(test_values, generated_sample)
        print(generated_sample.shape)
        fig, ax1 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')

        _, rangeR, _ = ax1.hist(
            test_values, histtype="step", label="FullSim", lw=2, bins=100, ls="--",
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
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        #fig.suptitle(f"Comparison of pt{i} @ epoch {epoch}", fontsize=16)
        ax1.set_xlabel(fr"$\Delta$ p_T 1v{i} [GeV]")
        ax1.legend(frameon=False, loc="upper right")
        plt.savefig(os.path.join(save_dir, f"comparison_pt{i}_{epoch}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_pt{i}_{epoch}.pdf"))

        fig, ax2 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')
        ax2.set_yscale("log")

        ax2.hist(test_values, histtype="step", lw=2, bins=100, ls="--", label="FullSim")
        ax2.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        ax2.set_xlabel(fr"$\Delta$ p_T 1v{i} [GeV]")
        ax2.legend(frameon=False, loc="upper right")
        # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
        # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
        # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_pt{i}_{epoch}_log.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_pt{i}_{epoch}_log.pdf"))
        writer.add_figure(f"phys_pt{i}", fig, global_step=epoch)
        writer.add_scalar(f"ws/phys_pt{i}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    for i in range(2, 10):

        test_values = dphi[:, i-1].flatten()[N_sel >= i]
        test_values = test_values[~np.isnan(test_values)]
        print(test_values.shape)
        generated_sample = dphi_flash[:, i-1].flatten()[N_sel >= i]
        generated_sample = generated_sample[~np.isnan(generated_sample)]
        print(generated_sample.shape)
        ws = wasserstein_distance(test_values, generated_sample)
        print(generated_sample.shape)
        fig, ax1 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')

        _, rangeR, _ = ax1.hist(
            test_values, histtype="step", label="FullSim", lw=2, bins=100, ls="--",
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
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        #fig.suptitle(f"Comparison of pt{i} @ epoch {epoch}", fontsize=16)
        ax1.set_xlabel(fr"$\Delta \phi$ 1v{i}")
        ax1.legend(frameon=False, loc="upper right")
        plt.savefig(os.path.join(save_dir, f"comparison_dphi{i}_{epoch}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dphi{i}_{epoch}.pdf"))

        fig, ax2 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')
        ax2.set_yscale("log")

        ax2.hist(test_values, histtype="step", lw=2, bins=100, ls="--", label="FullSim")
        ax2.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        ax2.set_xlabel(fr"$\Delta \phi$ 1v{i}")
        ax2.legend(frameon=False, loc="upper right")
        # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
        # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
        # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dphi{i}_{epoch}_log.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dphi{i}_{epoch}_log.pdf"))
        writer.add_figure(f"dphi{i}", fig, global_step=epoch)
        writer.add_scalar(f"ws/dphi{i}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    print("Done with dphi")

    for i in range(2, 10):

        test_values = dpt[:, i-1].flatten()[N_sel >= i]
        test_values = test_values[~np.isnan(test_values)]
        print(test_values.shape)
        generated_sample = dpt_flash[:, i-1].flatten()[N_sel >= i]
        generated_sample = generated_sample[~np.isnan(generated_sample)]
        print(generated_sample.shape)
        ws = wasserstein_distance(test_values, generated_sample)
        print(generated_sample.shape)
        fig, ax1 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')

        _, rangeR, _ = ax1.hist(
            test_values, histtype="step", label="FullSim", lw=2, bins=100, ls="--",
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
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        #fig.suptitle(f"Comparison of pt{i} @ epoch {epoch}", fontsize=16)
        ax1.set_xlabel(fr"$\Delta$p$_T$ 1v{i} [GeV]")
        ax1.legend(frameon=False, loc="upper right")
        plt.savefig(os.path.join(save_dir, f"comparison_dpt{i}_{epoch}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dpt{i}_{epoch}.pdf"))

        fig, ax2 = plt.subplots(1, 1) # figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')
        ax2.set_yscale("log")

        ax2.hist(test_values, histtype="step", lw=2, bins=100, ls="--", label="FullSim")
        ax2.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=2,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim",
        )
        ax2.set_xlabel(fr"$\Delta$p$_T$ 1v{i} [GeV]")
        ax2.legend(frameon=False, loc="upper right")
        # ax2.title(f"Log Comparison of {list(dff_test_reco)[i]}")
        # plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")
        # plt.savefig(os.path.join(save_dir, f"comparison_{names[i]}.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dpt{i}_{epoch}_log.png"))
        plt.savefig(os.path.join(save_dir, f"comparison_dpt{i}_{epoch}_log.pdf"))
        writer.add_figure(f"dpt{i}", fig, global_step=epoch)
        writer.add_scalar(f"ws/dpt{i}_wasserstein_distance", ws, global_step=epoch)

    print("Done with dpt")
    full_df["pt0"] = full_df["pt0"].apply(lambda x: np.exp(x+3))
    flash_df["pt0"] = flash_df["pt0"].apply(lambda x: np.exp(x+3))
    plt.rcParams.update(plt.rcParamsDefault)
    ranges = [[0, 100], [-5.5, 5.5], [-3.14, 3.14]]
    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim', lw=2,  ls='--')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim', lw=2)
    fig = corner.corner(full_df.iloc[:, [0, 1, 2]].values, labels=[r'p$_T$ [GeV]', 'Eta', 'Phi'], range=ranges, color='tab:blue',hist_kwargs ={"ls":'--'}, contour_kwargs ={"linestyles":"--"},
                        levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    corner.corner(flash_df.iloc[:, [0, 1, 2,]].values, levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange', range=ranges,
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=24, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 4.0), loc='upper right')
    #weights=weights * len(bilby_samples) / len(params_samples), range=dom)
    # plt.suptitle('Jet tagging distributions correlations', fontsize=20)
    plt.suptitle(r'$\bf{CMS}$ $\it{Simulation \; Preliminary}$', fontsize=16, x=0.5, y=1.0005, horizontalalignment='right', **{'fontname':"sans-serif"})

    plt.savefig(os.path.join(save_dir, f"corner{i}_{epoch}_log.png"))
    plt.savefig(os.path.join(save_dir, f"corner{i}_{epoch}_log.pdf"))
