import os

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

from scipy.stats import wasserstein_distance

from postprocessing import postprocessing, gen_columns, reco_columns
from post_actions import vars_dictionary
from corner_plots import make_corner
from conditioning_plot import conditioning_plot


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

    # Postprocessing for both test and samples datasets

    reco = postprocessing(reco, vars_dictionary)
    samples = postprocessing(samples, vars_dictionary)

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
        writer.add_figure(f"{column}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{column}_wasserstein_distance", ws, global_step=epoch)
        plt.close()

    # Return to physical kinematic variables

    for df in [reco, samples, saturated_samples]:
        df["MElectron_pt"] = df["MElectron_ptRatio"] * gen["MGenElectron_pt"]
        df["MElectron_eta"] = df["MElectron_etaMinusGen"] + gen["MGenElectron_eta"]
        df["MElectron_phi"] = df["MElectron_phiMinusGen"] + gen["MGenElectron_phi"]

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
    writer.add_figure("Isolation", fig, global_step=epoch)

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
    writer.add_figure("Impact parameter", fig, global_step=epoch)

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
        r"Impact parameter vs \sqrt(dxy^2 + dz^2)",
        ranges=ranges,
    )
    writer.add_figure(
        r"Impact parameter vs \sqrt(dxy^2 + dz^2)", fig, global_step=epoch
    )

    # Kinematics

    labels = ["MElectron_pt", "MElectron_eta", "MElectron_phi"]

    fig = make_corner(reco, saturated_samples, labels, "Kinematics")
    writer.add_figure("Kinematics", fig, global_step=epoch)

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
    writer.add_figure("Supercluster", fig, global_step=epoch)
       
    # Conditioning

    targets = ["MElectron_ip3d", "MElectron_sip3d", "MElectron_jetRelIso"]

    ranges = [[0, 0.1], [0, 10], [0, 5]]

    conds = [f"MGenElectron_statusFlag{i}" for i in (0, 2, 7)]
    conds.append("ClosestJet_EncodedPartonFlavour_b")

    colors = ["red", "green", "blue", "orange"]

    for target, rangeR in zip(targets, ranges):

        fig = plt.figure()

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color in zip(conds, colors):
            mask = gen[cond].values.astype(bool)

            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            plt.hist(full, bins=100, range=rangeR, histtype="step", ls="--", color=color)
            plt.hist(flash, bins=100, range=rangeR, histtype="step", label=f"{cond}", color=color)
        
        plt.legend()
        plt.savefig(f"{save_dir}/{target}_conditioning.png", format="png")
        plt.close()



