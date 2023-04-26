import os

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import corner

from scipy.stats import wasserstein_distance

from postprocessing import postprocessing, gen_columns, reco_columns
from post_actions import target_dictionary, context_dictionary
from corner_plots import make_corner

import mplhep


def validate_electrons(
    test_loader,
    model,
    epoch,
    writer,
    save_dir,
    args,
    device,
    clf_loaders=None,
):
    if writer is not None:
        save_dir = os.path.join(save_dir, f"./figures/validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    model.eval()
    # Generate samples
    with torch.no_grad():
        gen = []
        reco = []
        samples = []

        for bid, data in enumerate(test_loader):
            z, y = data[0], data[1]
            inputs_y = y.cuda(device)

            z_sampled = model.sample(
                num_samples=1, context=inputs_y.view(-1, args.y_dim)
            )
            z_sampled = z_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            z_sampled = z_sampled.reshape(-1, args.zdim)
            gen.append(inputs_y)
            reco.append(z)
            samples.append(z_sampled)

    # Making DataFrames

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.zdim))
    samples = np.array(samples).reshape((-1, args.zdim))

    gen = pd.DataFrame(data=gen, columns=gen_columns)
    reco = pd.DataFrame(data=reco, columns=reco_columns)
    samples = pd.DataFrame(data=samples, columns=reco_columns)

    # Postprocessing

    reco = postprocessing(reco, target_dictionary)

    samples = postprocessing(samples, target_dictionary)

    # New DataFrame containing FullSim-range saturated samples

    saturated_samples = pd.DataFrame()

    # 1D FlashSim/FullSim comparison

    for column in reco_columns:
        ws = wasserstein_distance(reco[column], samples[column])

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        # RECO histogram
        _, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=1, bins=100, label="FullSim"
        )

        # Saturation based on FullSim range
        saturated_samples[column] = np.where(
            samples[column] < np.min(rangeR), np.min(rangeR), samples[column]
        )
        saturated_samples[column] = np.where(
            saturated_samples[column] > np.max(rangeR),
            np.max(rangeR),
            saturated_samples[column],
        )

        # Samples histogram
        axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        # Log-scale comparison

        axs[1].set_yscale("log")
        axs[1].hist(reco[column], histtype="step", lw=1, bins=100)
        axs[1].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
        )
        plt.savefig(os.path.join(save_dir, f"{column}.png"))
        # writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        # writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # Return to physical kinematic variables

    for df in [reco, samples, saturated_samples]:
        df["MElectron_pt"] = df["MElectron_ptRatio"] * gen["MGenElectron_pt"]
        df["MElectron_eta"] = df["MElectron_etaMinusGen"] + gen["MGenElectron_eta"]
        df["MElectron_phi"] = df["MElectron_phiMinusGen"] + gen["MGenElectron_phi"]

    # Zoom-in for high ws distributions

    incriminated = [
        ["MElectron_dr03HcalDepth1TowerSumEt", [0, 10]],
        ["MElectron_dr03TkSumPt", [0, 10]],
        ["MElectron_dr03TkSumPtHEEP", [0, 10]],
        ["MElectron_dr03EcalRecHitSumEt", [0, 10]],
        ["MElectron_dxyErr", [0, 0.1]],
        ["MElectron_dzErr", [0, 0.2]],
        ["MElectron_energyErr", [0, 5]],
        ["MElectron_hoe", [0, 0.4]],
        ["MElectron_ip3d", [0, 0.1]],
        ["MElectron_jetPtRelv2", [0, 10]],
        ["MElectron_jetRelIso", [0, 2]],
        ["MElectron_miniPFRelIso_all", [0, 1]],
        ["MElectron_miniPFRelIso_chg", [0, 1]],
        ["MElectron_pfRelIso03_all", [0, 0.5]],
        ["MElectron_pfRelIso03_chg", [0, 0.5]],
        ["MElectron_sieie", [0.005, 0.02]],
        ["MElectron_sip3d", [0, 10]],
        ["MElectron_pt", [0, 100]],
        ["MElectron_eta", [-3, 3]],
        ["MElectron_phi", [-3.14, 3.14]],
    ]
    for elm in incriminated:
        column = elm[0]
        rangeR = elm[1]
        inf = rangeR[0]
        sup = rangeR[1]

        full = reco[column].values
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[column].values
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        ws = wasserstein_distance(full, flash)

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        axs[0].hist(
            full, histtype="step", lw=1, bins=100, range=rangeR, label="FullSim"
        )
        axs[0].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        axs[1].set_yscale("log")
        axs[1].hist(full, histtype="step", range=rangeR, lw=1, bins=100)
        axs[1].hist(
            flash,
            histtype="step",
            lw=1,
            range=rangeR,
            bins=100,
        )
        plt.savefig(f"{save_dir}/{column}_incriminated.png", format="png")
        # writer.add_figure(f"Zoom_in_1D_Distributions/{column}", fig, global_step=epoch)
        plt.close()

    # Return to physical kinematic variables

    physical = ["MElectron_pt", "MElectron_eta", "MElectron_phi"]

    for column in physical:
        ws = wasserstein_distance(reco[column], samples[column])

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
        fig.suptitle(f"{column} comparison")

        # RECO histogram
        _, rangeR, _ = axs[0].hist(
            reco[column], histtype="step", lw=1, bins=100, label="FullSim"
        )

        # Saturation based on FullSim range
        saturated_samples[column] = np.where(
            samples[column] < np.min(rangeR), np.min(rangeR), samples[column]
        )
        saturated_samples[column] = np.where(
            saturated_samples[column] > np.max(rangeR),
            np.max(rangeR),
            saturated_samples[column],
        )

        # Samples histogram
        axs[0].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
            label=f"FlashSim, ws={round(ws, 4)}",
        )

        axs[0].legend(frameon=False, loc="upper right")

        # Log-scale comparison

        axs[1].set_yscale("log")
        axs[1].hist(reco[column], histtype="step", lw=1, bins=100)
        axs[1].hist(
            saturated_samples[column],
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=100,
        )
        plt.savefig(os.path.join(save_dir, f"{column}.png"), format="png")
        # writer.add_figure(f"1D_Distributions/{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # Conditioning

    targets = [
        "MElectron_ip3d",
        "MElectron_sip3d",
        "MElectron_jetRelIso",
        "MElectron_pfRelIso03_all",
    ]

    ranges = [[0, 0.1], [0, 10], [0, 5], [0, 1]]

    conds = [f"MGenElectron_statusFlag{i}" for i in (0, 2, 7)]
    conds.append("ClosestJet_EncodedPartonFlavour_b")

    names = [
        "isPrompt",
        "isTauDecayProduct",
        "isHardProcess",
        "ClosestJet_partonFlavour_is_b",
    ]

    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]

    for target, rangeR in zip(targets, ranges):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        axs[0].set_xlabel(f"{target}")
        axs[1].set_xlabel(f"{target}")

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(conds, colors, names):
            mask = gen[cond].values.astype(bool)
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs[0].hist(
                full, bins=50, range=rangeR, histtype="step", ls="--", color=color
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
            )

            axs[1].hist(
                full, bins=50, range=rangeR, histtype="step", ls="--", color=color
            )
            axs[1].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
            )

            del full, flash

        mask = (
            gen["ClosestJet_EncodedPartonFlavour_gluon"].values
            + gen["ClosestJet_EncodedPartonFlavour_light"].values
        ).astype(bool)
        full = reco[target].values
        full = full[mask]
        full = full[~np.isnan(full)]
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[target].values
        flash = flash[mask]
        flash = flash[~np.isnan(flash)]
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        axs[0].hist(
            full, bins=50, range=rangeR, histtype="step", ls="--", color="tab:purple"
        )
        axs[0].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="ClosestJet_partonFlavour_is_guds",
            color="tab:purple",
        )

        axs[1].hist(
            full, bins=50, range=rangeR, histtype="step", ls="--", color="tab:purple"
        )
        axs[1].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="ClosestJet_partonFlavour_is_guds",
            color="tab:purple",
        )
        del full, flash

        axs[0].legend(frameon=False, loc="upper right")
        plt.savefig(f"{save_dir}/{target}_conditioning.png", format="png")
        # writer.add_figure(
        #     f"Conditioning/{target}_conditioning.png", fig, global_step=epoch
        # )
        plt.close()

    # Normalized version

    for target, rangeR in zip(targets, ranges):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        # write cms label

        mplhep.cms.text("Simulation Preliminary")

        axs[0].set_xlabel(f"{target}")
        axs[1].set_xlabel(f"{target}")

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(conds, colors, names):
            mask = gen[cond].values.astype(bool)
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs[0].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                color=color,
                density=True,
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=True,
            )

            axs[1].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                color=color,
                density=True,
            )
            axs[1].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=True,
            )

            del full, flash

        mask = (
            gen["ClosestJet_EncodedPartonFlavour_gluon"].values
            + gen["ClosestJet_EncodedPartonFlavour_light"].values
        ).astype(bool)
        full = reco[target].values
        full = full[mask]
        full = full[~np.isnan(full)]
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[target].values
        flash = flash[mask]
        flash = flash[~np.isnan(flash)]
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        axs[0].hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            color="tab:purple",
            density=True,
        )
        axs[0].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="ClosestJet_partonFlavour_is_guds",
            color="tab:purple",
            density=True,
        )

        axs[1].hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            color="tab:purple",
            density=True,
        )
        axs[1].hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            label="ClosestJet_partonFlavour_is_udsg",
            color="tab:purple",
            density=True,
        )
        del full, flash

        axs[0].legend(frameon=False, loc="upper right")
        plt.savefig(f"{save_dir}/{target}_conditioning_normalized.png", format="png")
        # writer.add_figure(
        #     f"Conditioning/{target}_conditioning_normalized.png", fig, global_step=epoch
        # )
        plt.close()

    # Normalized version

    for target, rangeR in zip(targets, ranges):
        mplhep.style.use("CMS")
        fig, axs = plt.subplots(1, 1, figsize=(12, 12), tight_layout=False)
        mplhep.cms.text("Simulation Preliminary")

        axs.set_xlabel(f"pfRelIso03 all")

        axs.set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        labels = []

        for cond, color, name in zip(conds, colors, names):
            mask = gen[cond].values.astype(bool)
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs.hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                lw=2,
                color=color,
                density=True,
            )
            axs.hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                lw=2,
                color=color,
                density=True,
            )

            labels.append(Patch(edgecolor=color, fill=False, lw=2, label=f"{name}"))

            del full, flash

        mask = (
            gen["ClosestJet_EncodedPartonFlavour_gluon"].values
            + gen["ClosestJet_EncodedPartonFlavour_light"].values
        ).astype(bool)
        full = reco[target].values
        full = full[mask]
        full = full[~np.isnan(full)]
        full = np.where(full > sup, sup, full)
        full = np.where(full < inf, inf, full)

        flash = samples[target].values
        flash = flash[mask]
        flash = flash[~np.isnan(flash)]
        flash = np.where(flash > sup, sup, flash)
        flash = np.where(flash < inf, inf, flash)

        axs.hist(
            full,
            bins=50,
            range=rangeR,
            histtype="step",
            ls="--",
            lw=2,
            color="tab:purple",
            density=True,
        )
        axs.hist(
            flash,
            bins=50,
            range=rangeR,
            histtype="step",
            lw=2,
            color="tab:purple",
            density=True,
        )

        labels.append(
            Patch(
                edgecolor="tab:purple",
                fill=False,
                lw=2,
                label="ClosestJet partonFlavour is udsg",
            )
        )

        del full, flash

        labels.append(
            Patch(edgecolor="black", fill=False, lw=2, ls="--", label="FullSim")
        )
        labels.append(Patch(edgecolor="black", fill=False, lw=2, label="FlashSim"))

        axs.legend(handles=labels, frameon=False, loc="upper center")
        plt.savefig(
            f"{save_dir}/{target}_conditioning_normalized_cms.png", format="png"
        )
        plt.savefig(
            f"{save_dir}/{target}_conditioning_normalized_cms.pdf", format="pdf"
        )
        # writer.add_figure(
        #     f"Conditioning/{target}_conditioning_normalized.png", fig, global_step=epoch
        # )
        plt.close()

    # Corner plots:

    # Isolation

    labels = [
        "MElectron_pt",
        "MElectron_eta",
        "MElectron_jetRelIso",
        "MElectron_miniPFRelIso_all",
        "MElectron_miniPFRelIso_chg",
        "MElectron_mvaFall17V1Iso",
        "MElectron_mvaFall17V1noIso",
        "MElectron_mvaFall17V2Iso",
        "MElectron_mvaFall17V2noIso",
        "MElectron_pfRelIso03_all",
        "MElectron_pfRelIso03_chg",
    ]

    fig = make_corner(reco, saturated_samples, labels, "Isolation")
    plt.gcf()
    plt.savefig(f"{save_dir}/Isolation_corner.png", format="png")
    # writer.add_figure("Corner_plots/Isolation", fig, global_step=epoch)

    # Impact parameter (range)

    labels = [
        "MElectron_pt",
        "MElectron_eta",
        "MElectron_ip3d",
        "MElectron_sip3d",
        "MElectron_dxy",
        "MElectron_dxyErr",
        "MElectron_dz",
        "MElectron_dzErr",
    ]

    ranges = [
        (0, 200),
        (-2, 2),
        (0, 0.2),
        (0, 5),
        (-0.2, 0.2),
        (0, 0.05),
        (-0.2, 0.2),
        (0, 0.05),
    ]

    fig = make_corner(
        reco, saturated_samples, labels, "Impact parameter", ranges=ranges
    )
    plt.gcf()

    plt.savefig(f"{save_dir}/Impact_parameter_corner.png", format="png")
    # writer.add_figure("Corner_plots/Impact parameter", fig, global_step=epoch)

    # Impact parameter comparison

    reco["MElectron_sqrt_xy_z"] = np.sqrt(
        (reco["MElectron_dxy"].values) ** 2 + (reco["MElectron_dz"].values) ** 2
    )
    saturated_samples["MElectron_sqrt_xy_z"] = np.sqrt(
        (saturated_samples["MElectron_dxy"].values) ** 2
        + (saturated_samples["MElectron_dz"].values) ** 2
    )

    labels = ["MElectron_sqrt_xy_z", "MElectron_ip3d"]

    ranges = [(0, 0.2), (0, 0.2)]

    fig = make_corner(
        reco,
        saturated_samples,
        labels,
        r"Impact parameter vs $\sqrt{dxy^2 + dz^2}$",
        ranges=ranges,
    )
    plt.gcf()
    plt.savefig(f"{save_dir}/Impact_parameter_v2_corner.png", format="png")
    # writer.add_figure(
    #     r"Corner_plots/Impact parameter vs \sqrt(dxy^2 + dz^2)", fig, global_step=epoch
    # )

    # Kinematics

    labels = ["MElectron_pt", "MElectron_eta", "MElectron_phi"]

    fig = make_corner(reco, saturated_samples, labels, "Kinematics")
    plt.gcf()
    plt.savefig(f"{save_dir}/Kinematics_corner.png", format="png")
    # writer.add_figure("Corner_plots/Kinematics", fig, global_step=epoch)

    # Supercluster

    labels = [
        "MElectron_pt",
        "MElectron_eta",
        "MElectron_sieie",
        "MElectron_r9",
        "MElectron_mvaFall17V1Iso",
        "MElectron_mvaFall17V1noIso",
        "MElectron_mvaFall17V2Iso",
        "MElectron_mvaFall17V2noIso",
    ]

    ranges = [
        (0, 200),
        (-2, 2),
        (0, 0.09),
        (0, 1.5),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ]

    fig = make_corner(reco, saturated_samples, labels, "Supercluster", ranges=ranges)
    plt.gcf()
    plt.savefig(f"{save_dir}/Supercluster_corner.png", format="png")
    # writer.add_figure("Corner_plots/Supercluster", fig, global_step=epoch)

    labels = [
        "MElectron_pt",
        "MElectron_deltaEtaSC",
        "MElectron_hoe",
        "MElectron_sieie",
        "MElectron_r9",
        "MElectron_eInvMinusPInv",
    ]

    names = [r"$p_T$ [GeV]", "deltaEtaSC", "hoe", "sieie", "r9", "eInvMinusPInv"]

    ranges = [
        (0, 200),
        (-0.1, 0.1),
        (0, 0.16),
        (0, 0.05),
        (0.3, 1.2),
        (-0.1, 0.1),
    ]
    blue_line = mlines.Line2D([], [], color="tab:blue", ls="--", label="FullSim")
    red_line = mlines.Line2D([], [], color="tab:orange", label="FlashSim")

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure()
    mplhep.cms.text("Simulation Preliminary", loc=0)

    fig = corner.corner(
        reco[labels],
        range=ranges,
        labels=names,
        color="tab:blue",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        hist_kwargs={"ls": "--"},
        contour_kwargs={"linestyles": "--"},
        label_kwargs={"fontsize": 16},
    )
    corner.corner(
        samples[labels],
        range=ranges,
        labels=names,
        fig=fig,
        color="tab:orange",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        label_kwargs={"fontsize": 16},
    )
    plt.legend(
        fontsize=24,
        frameon=False,
        handles=[blue_line, red_line],
        bbox_to_anchor=(0.0, 1.0, 1.0, 4.0),
        loc="upper right",
    )
    plt.suptitle(
        r"$\bf{CMS}$ $\it{Simulation \; Preliminary}$",
        fontsize=16,
        x=0.29,
        y=1.0,
        horizontalalignment="right",
        fontname="sans-serif",
    )
    plt.savefig(f"{save_dir}/Supercluster_corner_cms.png", format="png")
    plt.savefig(f"{save_dir}/Supercluster_corner_cms.pdf", format="pdf")

    # writer.add_figure("Corner_plots/Supercluster", fig, global_step=epoch)
