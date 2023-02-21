import numpy as np
import pandas as pd
import h5py

import corner
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

from postprocessing import reco_columns, gen_columns


def make_corner(reco, samples, labels, title, *args, **kwargs):
    blue_line = mlines.Line2D([], [], color="tab:blue", label="FullSim")
    red_line = mlines.Line2D([], [], color="tab:orange", label="FlashSim")
    fig = corner.corner(
        reco[labels],
        labels=labels,
        color="tab:blue",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
        *args,
        **kwargs
    )
    corner.corner(
        samples[labels],
        fig=fig,
        color="tab:orange",
        levels=[0.5, 0.9, 0.99],
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


if __name__ == "__main__":

    f = h5py.File("MElectrons_post.hdf5", "r")
    df = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
    f.close()

    # Corner plot: kinematics

    df["MElectron_pt"] = df["MElectron_ptRatio"] * df["MGenElectron_pt"]
    df["MElectron_eta"] = df["MElectron_etaMinusGen"] + df["MGenElectron_eta"]
    df["MElectron_phi"] = df["MElectron_phiMinusGen"] + df["MGenElectron_phi"]

    labels = ["MElectron_pt", "MElectron_eta", "MElectron_phi"]

    fig = corner.corner(
        df[labels],
        labels=labels,
        color="tab:blue",
        levels=(0.5, 0.9, 0.99),
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
    )
    plt.suptitle("MElectrons kinematics variables", fontsize=20)
    plt.savefig("corner_fig/corner_kinematics.pdf", format="pdf")
    plt.close()

    # Corner plot: isolation

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

    fig = corner.corner(
        df[labels],
        labels=labels,
        color="tab:blue",
        levels=(0.5, 0.9, 0.99),
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
    )
    plt.suptitle("MElectrons isolation variables", fontsize=20)
    plt.savefig("corner_fig/corner_isolation.pdf", format="pdf")
    plt.close()

    # Corner plot: ip

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

    fig = corner.corner(
        df[labels],
        range=ranges,
        labels=labels,
        color="tab:blue",
        levels=(0.5, 0.9, 0.99),
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
    )
    plt.suptitle("MElectrons ip variables", fontsize=20)
    plt.savefig("corner_fig/corner_ip.pdf", format="pdf")
    plt.close()

    # Corner plot: supercluster

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

    fig = corner.corner(
        df[labels],
        labels=labels,
        color="tab:blue",
        levels=(0.5, 0.9, 0.99),
        hist_bin_factor=3,
        scale_hist=True,
        plot_datapoints=False,
    )
    plt.suptitle("MElectrons supercluster variables", fontsize=20)
    plt.savefig("corner_fig/corner_supercluster.pdf", format="pdf")
    plt.close()
