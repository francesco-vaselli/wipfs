import os

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

from scipy.stats import wasserstein_distance

from postprocessing import postprocessing, gen_columns, reco_columns
from corner_plots import make_corner


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
    model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, ".", "figure", f"validation@epoch-{epoch}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # Generate samples

    with torch.no_grad():

        gen = []
        reco = []
        samples = []

        for bid, data in enumerate(test_loader):

            z, y = data[0], data[1]
            inputs_y = y.to(device)

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

    gen = postprocessing(gen)
    reco = postprocessing(reco)
    samples = postprocessing(samples)

    # Return to physical kinematic variables

    for df in [reco, samples]:
        df["MElectron_pt"] = df["MElectron_ptRatio"] * gen["MGenElectron_pt"]
        df["MElectron_eta"] = df["MElectron_etaMinusGen"] + gen["MGenElectron_eta"]
        df["MElectron_phi"] = df["MElectron_phiMinusGen"] + gen["MGenElectron_phi"]

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
        x = np.where(samples[column] < np.min(rangeR), np.min(rangeR), samples[column])
        x = np.where(samples[column] > np.max(rangeR), np.max(rangeR), samples[column])

        # Samples histogram
        axs[0].hist(
            x,
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
        axs[1].hist(x, histtype="step", lw=1, range=[np.min(rangeR), np.max(rangeR)], bins=100)
        writer.add_figure(f"{column}", fig, global_step=epoch)
        plt.close()
        del x

    # Corner plots

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

    fig = make_corner(reco, samples, labels, "Isolation")
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

    fig = make_corner(reco, samples, labels, "Impact parameter")
    writer.add_figure("Impact parameter", fig, global_step=epoch)

    # Impact parameter comparison (range)

    reco["MElectron_sqrt_xy_z"] = np.sqrt((reco["MElectron_dxy"].values)**2 + (reco["MElectron_dz"].values)**2)
    samples["MElectron_sqrt_xy_z"] = np.sqrt((samples["MElectron_dxy"].values)**2 + (samples["MElectron_dz"].values)**2)

    labels = ["MElectron_sqrt_xy_z", "MElectron_ip3d"]

    fig = make_corner(reco, samples, labels, r"Impact parameter vs \sqrt(dxy^2 + dz^2)")
    writer.add_figure(r"Impact parameter vs \sqrt(dxy^2 + dz^2)", fig, global_step=epoch)

    # Kinematics

    labels = ["MElectron_pt", "MElectron_eta", "MElectron_phi"]

    fig = make_corner(reco, samples, labels, "Kinematics")
    writer.add_figure("Kinematics", fig, global_step=epoch)

