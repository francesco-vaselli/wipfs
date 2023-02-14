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


def validate_rejets(
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

        gen = []
        reco = []
        samples = []

        for bid, data in enumerate(test_loader):

            _, y, z = data[0], data[1], data[2]
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

    gen = np.array(gen).reshape((args.test_limit, args.y_dim))
    reco = np.array(reco).reshape((args.test_limit, args.zdim))
    samples = np.array(samples).reshape((args.test_limit, args.zdim))

    # Samples postprocessing 

    samples[:, 15] = np.rint(samples[:, 15])
    samples[:, 16] = np.rint(samples[:, 16])
    samples[:, 16] = np.where(samples[:, 16]==1, 0, samples[:, 16])
    samples[:, 16] = np.where(samples[:, 16]==3, 2, samples[:, 16])
    samples[:, 16] = np.where(samples[:, 16]==4, 2, samples[:, 16])
    samples[:, 16] = np.where(samples[:, 16]==5, 6, samples[:, 16])
    samples[:, 15] = np.rint(samples[:, 15])
    samples[:, 16] = np.rint(samples[:, 16])

    jet_cond = ["MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour",]
    
    df = pd.DataFrame(data=gen, columns=jet_cond)

    samples[:, 7] = samples[:, 7] + df['GenJet_eta'].values
    samples[:, 9] = samples[:, 9] * df['GenJet_mass'].values
    samples[:, 11] = samples[:, 11] +  df['GenJet_phi'].values
    samples[:, 11]= np.where(samples[:, 11]< -np.pi, samples[:, 11] + 2*np.pi, samples[:, 11])
    samples[:, 11]= np.where(samples[:, 11]> np.pi, samples[:, 11] - 2*np.pi, samples[:, 11])
    samples[:, 12] = samples[:, 12] * df['GenJet_pt'].values

    # Reco postprocessing

    reco[:, 7] = reco[:, 7] + df['GenJet_eta'].values
    reco[:, 9] = reco[:, 9] * df['GenJet_mass'].values
    reco[:, 11] = reco[:, 11] +  df['GenJet_phi'].values
    reco[:, 11]= np.where( reco[:, 11]< -np.pi, reco[:, 11] + 2*np.pi, reco[:, 11])
    reco[:, 11]= np.where( reco[:, 11]> np.pi, reco[:, 11] - 2*np.pi, reco[:, 11])
    reco[:, 12] = reco[:, 12] * df['GenJet_pt'].values

    # Plots

    names = ["Jet_area", "Jet_btagCMVA", "Jet_btagCSVV2", "Jet_btagDeepB", "Jet_btagDeepC", "Jet_btagDeepFlavB", "Jet_btagDeepFlavC",
        "Jet_etaMinusGen", "Jet_bRegCorr", "Jet_massRatio", "Jet_nConstituents", "Jet_phiMinusGen",
        "Jet_ptRatio","Jet_qgl", "Jet_muEF", "Jet_puId", "Jet_jetId"]

    for i in range(0, args.zdim):
        ws = wasserstein_distance(reco[:, i], samples[:, i])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(reco[:, i], histtype='step', label='FullSim', lw=1, bins=100)
        samples[:, i] = np.where(samples[:, i] < rangeR.min(), rangeR.min(), samples[:, i])
        samples[:, i] = np.where(samples[:, i] > rangeR.max(), rangeR.max(), samples[:, i])
        ax1.hist(samples[:, i], bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()], label=f'FlashSim, ws={round(ws, 4)}')
        fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
        ax1.legend(frameon=False, loc='upper right')

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(reco[:, i], histtype='step', lw=1, bins=100)
        ax2.hist(samples[:, i], bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()])
        writer.add_figure(f"{names[i]}", fig, global_step=epoch)
    plt.close()

    # Corner plot

    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    fig = corner.corner(reco[:, [1, 2, 3, 4, 5]], labels=['Jet_btagCMVA', 'Jet_btagCSVV2', 'Jet_btagDeepB', 'Jet_btagDeepC', 'Jet_btagDeepFlavB'], color='tab:blue',
                        levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)

    corner.corner(samples[:, [1, 2, 3, 4, 5]], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=24, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 4.0), loc='upper right')
    plt.suptitle('Jet tagging distributions correlations', fontsize=20)
    writer.add_figure("Jet tagging correlations", fig, global_step=epoch)
    plt.close()