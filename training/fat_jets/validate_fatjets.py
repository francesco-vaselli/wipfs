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



def histANDroc(gen, gen_df, nb):
    truth = np.abs(gen_df)
    mask_b = np.where(truth[:, 7]==nb)
    mask_s = np.where(truth[:, 7]==2)
    print(truth[:, 7])
    bs = gen[mask_b, 4].flatten()
    nbs = gen[mask_s, 4].flatten()
    # nbs = nbs[0:len(bs)]

    bs = bs[bs >=-0.05]
    nbs = nbs[nbs >=-0.05]

    bs = np.where(bs<0, 0, bs)
    nbs = np.where(nbs<0, 0, nbs)

    bs = np.where(bs>1, 1, bs)
    nbs = np.where(nbs>1, 1, nbs)

    # bs = bs[0:len(nbs)]

    figure = plt.figure(figsize=(9, 6.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.hist(bs.flatten(), bins=50, label="b",  histtype='step', lw=2, color='C2')
    plt.hist(nbs.flatten(), bins=50, label="uds",  histtype='step', lw=2, color='C3')
    plt.title("FlashSim particleNetMD_XbbvsQCD for b ground truth", fontsize=16)
    plt.legend(fontsize=16, frameon=False, loc='upper left')

    y_bs = np.ones(len(bs))
    y_nbs = np.zeros(len(nbs))
    y_t = np.concatenate((y_bs, y_nbs))
    y_s = np.concatenate((bs, nbs))

    fpr, tpr, _ = roc_curve(y_t.ravel(), y_s.ravel())
    roc_auc = auc(fpr, tpr)

    return figure, fpr, tpr, roc_auc, bs, nbs


def validate_fatjets(
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

        for bid, (z, y) in enumerate(test_loader):

            inputs_y = y.cuda(device)

            while True:
                try:
                    z_sampled = model.sample(
                        num_samples=1, context=inputs_y.view(-1, args.y_dim)
                    )
                    break
                except AssertionError:
                    print("Sample failed, retrying")
                    pass
            z_sampled = z_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            z_sampled = z_sampled.reshape(-1, args.x_dim)
            gen.append(inputs_y)
            reco.append(z)
            samples.append(z_sampled)

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))

    names = [
        "Mpt",
        "Meta",
        "Mphi",
        "Mfatjet_msoftdrop",
        "Mfatjet_particleNetMD_XbbvsQCD",
    ]

    for i in range(0, args.x_dim):
        ws = wasserstein_distance(reco[:, i], samples[:, i])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(
            reco[:, i], histtype="step", label="FullSim", lw=1, bins=100
        )
        samples[:, i] = np.where(
            samples[:, i] < rangeR.min(), rangeR.min(), samples[:, i]
        )
        samples[:, i] = np.where(
            samples[:, i] > rangeR.max(), rangeR.max(), samples[:, i]
        )
        ax1.hist(
            samples[:, i],
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
        ax1.legend(frameon=False, loc="upper right")

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(reco[:, i], histtype="step", lw=1, bins=100)
        ax2.hist(
            samples[:, i],
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
        )
        writer.add_figure(f"{names[i]}", fig, global_step=epoch)
        plt.savefig(f"{save_dir}/{names[i]}.png")

    plt.close()

    # Samples postprocessing

    jet_cond = [
        "MgenjetAK8_pt",
        "MgenjetAK8_phi",
        "MgenjetAK8_eta",
        "MgenjetAK8_hadronFlavour",
        "MgenjetAK8_partonFlavour",
        "MgenjetAK8_mass",
        "MgenjetAK8_ncFlavour",
        "MgenjetAK8_nbFlavour",
    ]

    df = pd.DataFrame(data=gen, columns=jet_cond)


    samples[:, 1] = samples[:, 1] + df["MgenjetAK8_eta"].values
    # samples[:, 9] = samples[:, 9] * df['GenJet_mass'].values
    samples[:, 2] = samples[:, 2] + df["MgenjetAK8_phi"].values
    samples[:, 2] = np.where(
        samples[:, 2] < -np.pi, samples[:, 2] + 2 * np.pi, samples[:, 2]
    )
    samples[:, 2] = np.where(
        samples[:, 2] > np.pi, samples[:, 2] - 2 * np.pi, samples[:, 2]
    )
    samples[:, 0] = samples[:, 0] * df["MgenjetAK8_pt"].values

    # Reco postprocessing

    reco[:, 1] = reco[:, 1] + df["MgenjetAK8_eta"].values
    # reco[:, 9] = reco[:, 9] * df['GenJet_mass'].values
    reco[:, 2] = reco[:, 2] + df["MgenjetAK8_phi"].values
    reco[:, 2] = np.where(reco[:, 2] < -np.pi, reco[:, 2] + 2 * np.pi, reco[:, 2])
    reco[:, 2] = np.where(reco[:, 2] > np.pi, reco[:, 2] - 2 * np.pi, reco[:, 2])
    reco[:, 0] = reco[:, 0] * df["MgenjetAK8_pt"].values
    # reco[:, 7] = reco[:, 7] + df['GenJet_eta'].values
    # reco[:, 9] = reco[:, 9] * df['GenJet_mass'].values
    # reco[:, 11] = reco[:, 11] +  df['GenJet_phi'].values
    # reco[:, 11]= np.where( reco[:, 11]< -np.pi, reco[:, 11] + 2*np.pi, reco[:, 11])
    # reco[:, 11]= np.where( reco[:, 11]> np.pi, reco[:, 11] - 2*np.pi, reco[:, 11])
    # reco[:, 12] = reco[:, 12] * df['GenJet_pt'].values

    # Plots
    names = [
        "Mpt_phys",
        "Meta_phys",
        "Mphi_phys",
    ]

    for i in range(0, 3):
        ws = wasserstein_distance(reco[:, i], samples[:, i])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(
            reco[:, i], histtype="step", label="FullSim", lw=1, bins=100
        )
        samples[:, i] = np.where(
            samples[:, i] < rangeR.min(), rangeR.min(), samples[:, i]
        )
        samples[:, i] = np.where(
            samples[:, i] > rangeR.max(), rangeR.max(), samples[:, i]
        )
        ax1.hist(
            samples[:, i],
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
        ax1.legend(frameon=False, loc="upper right")

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(reco[:, i], histtype="step", lw=1, bins=100)
        ax2.hist(
            samples[:, i],
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
        )
        writer.add_figure(f"{names[i]}", fig, global_step=epoch)
        plt.savefig(f"{save_dir}/{names[i]}.png")

    plt.close()

    # Conditioning
    jet_target = [
        "Mpt_ratio",
        "Meta_sub",
        "Mphi_sub",
        "Mfatjet_msoftdrop",
        "Mfatjet_particleNetMD_XbbvsQCD",
    ]
    reco = pd.DataFrame(data=reco, columns=jet_target)
    samples = pd.DataFrame(data=samples, columns=jet_target)

    targets = ["Mfatjet_particleNetMD_XbbvsQCD"]

    ranges = [[-0.1, 1]]

    conds = [0, 1, 2]

    names = [
        "0 b",
        "1 b",
        "2 b",
    ]

    colors = ["tab:red", "tab:green", "tab:blue"]

    for target, rangeR in zip(targets, ranges):

        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        axs[0].set_xlabel(f"{target}")
        axs[1].set_xlabel(f"{target}")

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(conds, colors, names):
            nb = df["MgenjetAK8_nbFlavour"].values
            mask = np.where(nb == cond, True, False)
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
            axs[0].legend(frameon=False, loc="upper right")
            writer.add_figure(f"XbbvsQCD for b content", fig, global_step=epoch)

    print(gen.shape)

    # ROC
    fig, fpr, tpr, roc_auc, bs, nbs = histANDroc(samples, df, 1)
    cfig, cfpr, ctpr, croc_auc, cbs, cnbs  = histANDroc(reco, df, 1)

    fig = plt.figure(figsize=(9, 6.5))
    lw = 2
    plt.plot(
        tpr,
        fpr,
        color="C1",
        lw=lw,
        label=f"ROC curve (area = %0.2f) FlashSim" % roc_auc,
    )
    
    plt.plot(
        ctpr,
        cfpr,
        color="C0",
        lw=lw,
        label="ROC curve (area = %0.2f) FullSim" % croc_auc,
    )

    #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #plt.xlim([0.0, 1.0])
    plt.yscale("log")
    plt.ylim([0.0005, 1.05])
    plt.xlabel("Efficency for b-jet (TP)", fontsize=16)
    plt.ylabel("Mistagging prob (FP)", fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Receiver operating characteristic 2bvs1b", fontsize=16)
    plt.legend(fontsize=16, frameon=False,loc="best")
    writer.add_figure("ROC2v1", fig, global_step=epoch)
    plt.close()

    fig, fpr, tpr, roc_auc, bs, nbs = histANDroc(samples, df, 0)
    cfig, cfpr, ctpr, croc_auc, cbs, cnbs  = histANDroc(reco, df, 0)

    fig = plt.figure(figsize=(9, 6.5))
    lw = 2
    plt.plot(
        tpr,
        fpr,
        color="C1",
        lw=lw,
        label=f"ROC curve (area = %0.2f) FlashSim" % roc_auc,
    )
    
    plt.plot(
        ctpr,
        cfpr,
        color="C0",
        lw=lw,
        label="ROC curve (area = %0.2f) FullSim" % croc_auc,
    )

    #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #plt.xlim([0.0, 1.0])
    plt.yscale("log")
    plt.ylim([0.0005, 1.05])
    plt.xlabel("Efficency for b-jet (TP)", fontsize=16)
    plt.ylabel("Mistagging prob (FP)", fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Receiver operating characteristic 2bvs0b", fontsize=16)
    plt.legend(fontsize=16, frameon=False,loc="best")
    writer.add_figure("ROC2v0", fig, global_step=epoch)
    plt.close()