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
import corner
import mplhep as hep


def make_corner(reco, samples, labels, title, ranges=None, *args, **kwargs):
    """utility fnc for making corner plots

    Args:
        reco (df): full sim df
        samples (df): flash sim df
        labels (_type_): _description_
        title (_type_): _description_
        ranges (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    blue_line = mlines.Line2D([], [], color="tab:blue", label="FullSim")
    red_line = mlines.Line2D([], [], color="tab:orange", label="FlashSim")
    fig = corner.corner(
        reco[labels],
        range=ranges,
        labels=labels,
        color="tab:blue",
        levels=[0.5, 0.9, 0.95],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        *args,
        **kwargs
    )
    corner.corner(
        samples[labels],
        range=ranges,
        fig=fig,
        color="tab:orange",
        levels=[0.5, 0.9, 0.95],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        *args,
        **kwargs
    )
    plt.legend(
        fontsize=24,
        frameon=False,
        handles=[blue_line, red_line],
        bbox_to_anchor=(0.0, 1.0, 1.0, 4.0),
        loc="upper right",
    )
    plt.suptitle(title, fontsize=20)
    plt.close()
    return fig


def makeROC(gen, gen_df, nb):
    truth = np.abs(gen_df)
    mask_b = np.where(truth[:, 7]==nb)
    mask_s = np.where(truth[:, 7]==2)

    s = gen[mask_s, 4].flatten()
    b = gen[mask_b, 4].flatten()

    s = s[s >=-0.05]
    b = b[b >=-0.05]

    s = np.where(s<0, 0, s)
    b = np.where(b<0, 0, b)

    s = np.where(s>1, 1, s)
    b = np.where(b>1, 1, b)

    y_bs = np.ones(len(s))
    y_nbs = np.zeros(len(b))
    y_t = np.concatenate((y_bs, y_nbs))
    y_s = np.concatenate((s, b))

    fpr, tpr, _ = roc_curve(y_t.ravel(), y_s.ravel())
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, s, b


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
        signal_flag = []

        for bid, (z, y) in enumerate(test_loader):

            inputs_y = y.cuda(device)[:, : args.y_dim]
            is_signal = y.cuda(device)[:, -1]

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
            is_signal = is_signal.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            z_sampled = z_sampled.reshape(-1, args.x_dim)
            gen.append(inputs_y)
            reco.append(z)
            samples.append(z_sampled)
            signal_flag.append(is_signal)

    gen = np.array(gen).reshape((-1, args.y_dim))
    reco = np.array(reco).reshape((-1, args.x_dim))
    samples = np.array(samples).reshape((-1, args.x_dim))
    signal_flag = np.array(signal_flag).reshape((-1, 1))
    gen = np.concatenate((gen, signal_flag), axis=1)

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
        plt.savefig(f"{save_dir}/{names[i]}.png")
        plt.savefig(f"{save_dir}/{names[i]}.pdf")
        if isinstance(epoch, int):
            writer.add_figure(f"{names[i]}", fig, global_step=epoch)
        else:
            writer.add_figure(f"{epoch}/{names[i]}", fig)


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
        "is_signal"
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
        plt.savefig(f"{save_dir}/{names[i]}.png")
        plt.savefig(f"{save_dir}/{names[i]}.pdf")
        if isinstance(epoch, int):
            writer.add_figure(f"{names[i]}", fig, global_step=epoch)
        else:
            writer.add_figure(f"{epoch}/{names[i]}", fig)



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

    # postprocess softdrop
    reco["Mfatjet_msoftdrop"] = reco["Mfatjet_msoftdrop"] * df["MgenjetAK8_mass"].values
    samples["Mfatjet_msoftdrop"] = (
        samples["Mfatjet_msoftdrop"] * df["MgenjetAK8_mass"].values
    )
    reco["Mfatjet_msoftdrop"] = np.where(
        reco["Mfatjet_msoftdrop"] < 0, -10, reco["Mfatjet_msoftdrop"]
    )
    samples["Mfatjet_msoftdrop"] = np.where(
        samples["Mfatjet_msoftdrop"] < 0, -10, samples["Mfatjet_msoftdrop"]
    )
    # reco["Mfatjet_msoftdrop"] = reco["Mfatjet_msoftdrop"].clip(upper=500)
    # samples["Mfatjet_msoftdrop"] = samples["Mfatjet_msoftdrop"].clip(upper=500)


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
        plt.savefig(f"{save_dir}/XbbvsQCD for b content.png")
        plt.savefig(f"{save_dir}/XbbvsQCD for b content.pdf")
        if isinstance(epoch, int):
            writer.add_figure(f"XbbvsQCD for b content", fig, global_step=epoch)
        else:
            writer.add_figure(f"{epoch}/XbbvsQCD for b content", fig)

    print(gen.shape)

    # ROC
    fpr, tpr, roc_auc, bs, nbs = makeROC(samples.values, df.values, 1)
    cfpr, ctpr, croc_auc, cbs, cnbs  = makeROC(reco.values, df.values, 1)

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
    # plt.yscale("log")
    plt.ylim([0.0005, 1.05])
    plt.xlabel("Efficency for b-jet (TP)", fontsize=16)
    plt.ylabel("Mistagging prob (FP)", fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Receiver operating characteristic 2bvs1b", fontsize=16)
    plt.legend(fontsize=16, frameon=False,loc="best")
    plt.savefig(f"{save_dir}/ROC2v1.png")
    plt.savefig(f"{save_dir}/ROC2v1.pdf")
    if isinstance(epoch, int):
        writer.add_figure("ROC2v1", fig, global_step=epoch)
    else:
        writer.add_figure(f"{epoch}/ROC2v1", fig)
    plt.close()

    fpr, tpr, roc_auc, bs, nbs = makeROC(samples.values, df.values, 0)
    cfpr, ctpr, croc_auc, cbs, cnbs  = makeROC(reco.values, df.values, 0)

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
    plt.savefig(f"{save_dir}/ROC2v0.png")
    plt.savefig(f"{save_dir}/ROC2v0.pdf")
    if isinstance(epoch, int):
        writer.add_figure("ROC2v0", fig, global_step=epoch)
    else:
        writer.add_figure(f"{epoch}/ROC2v0", fig)
    plt.close()

    fig = make_corner(reco, samples, labels=["Mfatjet_msoftdrop", "Mfatjet_particleNetMD_XbbvsQCD"], title="Total softdrop mass vs XbbvsQCD",
                      ranges=[[0, 200], [0,1]])
    plt.suptitle("softdrop vs tagger", fontsize=16)
    plt.savefig(f"{save_dir}/corner.png")
    plt.savefig(f"{save_dir}/corner.pdf")
    if isinstance(epoch, int):
        writer.add_figure("corner", fig, global_step=epoch)
    else:
        writer.add_figure(f"{epoch}/corner", fig)
    plt.close()

    sig_reco = reco[df["is_signal"] == 1]
    sig_samples = samples[df["is_signal"] == 1]
    bkg_reco = reco[df["is_signal"] == 0]
    bkg_samples = samples[df["is_signal"] == 0]

    fig = make_corner(sig_reco, sig_samples, labels=["Mfatjet_msoftdrop", "Mfatjet_particleNetMD_XbbvsQCD"], title="Signal softdrop mass vs XbbvsQCD",
                        ranges=[[0, 200], [0,1]])
    plt.savefig(f"{save_dir}/corner_signal.png")
    plt.savefig(f"{save_dir}/corner_signal.pdf")
    if isinstance(epoch, int):
        writer.add_figure("corner_signal", fig, global_step=epoch)
    else:
        writer.add_figure(f"{epoch}/corner_signal", fig)
    plt.close()

    fig = make_corner(bkg_reco, bkg_samples, labels=["Mfatjet_msoftdrop", "Mfatjet_particleNetMD_XbbvsQCD"], title="Background softdrop mass vs XbbvsQCD",
                        ranges=[[0, 200], [0,1]])
    plt.savefig(f"{save_dir}/corner_background.png")
    plt.savefig(f"{save_dir}/corner_background.pdf")
    if isinstance(epoch, int):
        writer.add_figure("corner_background", fig, global_step=epoch)
    else:
        writer.add_figure(f"{epoch}/corner_background", fig)
    plt.close()
    
    # 1 d plot of softdrop mass for total, sig and bkg, fullsim vs flash
    recos = [reco, sig_reco, bkg_reco]
    samples1 = [samples, sig_samples, bkg_samples]
    titles = ["Fatjet_softdrop", "Fatjet_softdrop (signal)", "Fatjet_softdrop (bkg)"]
    for i in range(0, 3):
        reco1 = recos[i][["Mfatjet_msoftdrop"]].values.flatten()
        samples2 = samples1[i][["Mfatjet_msoftdrop"]].values.flatten()
        ws = wasserstein_distance(reco1, samples2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(
            reco1, histtype="step", label="FullSim", lw=1, bins=100
        )
        samples2 = np.where(
            samples2 < rangeR.min(), rangeR.min(), samples2
        )
        samples2 = np.where(
            samples2 > rangeR.max(), rangeR.max(), samples2
        )
        ax1.hist(
            samples2,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        fig.suptitle(f"Comparison of {titles[i]}", fontsize=16)
        ax1.legend(frameon=False, loc="upper right")

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(reco1, histtype="step", lw=1, bins=100)
        ax2.hist(
            samples2,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
        )
        plt.savefig(f"{save_dir}/{titles[i]}.png")
        plt.savefig(f"{save_dir}/{titles[i]}.pdf")
        if isinstance(epoch, int):
            writer.add_figure(f"{titles[i]}", fig, global_step=epoch)
        else:
            writer.add_figure(f"{epoch}/{titles[i]}", fig)


    targets = ["Mfatjet_msoftdrop"]

    ranges = [[0, 250]]

    conds = [0, 1]

    names = [
        "bkg",
        "sig",
    ]

    colors = ["tab:red", "tab:green"]
    hep.style.use("CMS")
    for target, rangeR in zip(targets, ranges):

        fig, axs = plt.subplots(1, 1) #, figsize=(9, 4.5), tight_layout=False)
        hep.cms.text('Simulation Preliminary')

        axs.set_xlabel(f"{target}")
        # axs[1].set_xlabel(f"{target}")

        axs.set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(conds, colors, names):
            nb = df["is_signal"].values
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

            # axs[0].hist(
            #     full, bins=50, range=rangeR, histtype="step", ls="--", color=color, l
            # )
            # axs[0].hist(
            #     flash,
            #     bins=50,
            #     range=rangeR,
            #     histtype="step",
            #     label=f"{name}",
            #     color=color,
            # )

            axs.hist(
                full, bins=50, range=rangeR, histtype="step", ls="--", color=color, label=f"FullSim {name}"
            )
            axs.hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                color=color,
                label=f"FlashSim {name}"
            )
        plt.savefig(f"{save_dir}/Softrdop_comp.png")
        plt.savefig(f"{save_dir}/Softrdop_comp.pdf")
        axs.legend(frameon=False, loc="upper right")
        if isinstance(epoch, int):
            writer.add_figure(f"Softrdop_comp", fig, global_step=epoch)
        else:
            writer.add_figure(f"{epoch}/Softrdop_comp", fig)