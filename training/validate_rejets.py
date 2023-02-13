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

        tot_sample = []
        dff_test_reco = []
        dff_test_gen = []

        for bid, data in enumerate(test_loader):

            _, y, z = data[0], data[1], data[2]
            inputs_y = y.to(device)

            z_sampled = model.sample(
                    num_samples=1, context=inputs_y.view(-1, args.y_dim)
                )
            z_sampled = z_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            print(z_sampled.shape, inputs_y.shape)
            z = z.cpu().detach().numpy()
            dff_test_reco.append(z)
            z_sampled = z_sampled.reshape(-1, args.zdim)
            tot_sample.append(z_sampled)
            dff_test_gen.append(y)


    tot_sample = np.array(tot_sample)
    generated_samples = np.reshape(tot_sample, (100000, 17))
    dff_test_reco = pd.Dataframe(data=np.array(dff_test_reco).reshape((100000, 17)))
    dff_test_gen = pd.Dataframe(data=np.array(dff_test_gen).reshape((100000, 14)))


    generated_samples[:, 15] = np.rint(generated_samples[:, 15])
    generated_samples[:, 16] = np.rint(generated_samples[:, 16])
    generated_samples[:, 16] = np.where(generated_samples[:, 16]==1, 0, generated_samples[:, 16])
    generated_samples[:, 16] = np.where(generated_samples[:, 16]==3, 2, generated_samples[:, 16])
    generated_samples[:, 16] = np.where(generated_samples[:, 16]==4, 2, generated_samples[:, 16])
    generated_samples[:, 16] = np.where(generated_samples[:, 16]==5, 6, generated_samples[:, 16])
    generated_samples[:, 15] = np.rint(generated_samples[:, 15])
    generated_samples[:, 16] = np.rint(generated_samples[:, 16])

    rescaled_generated_samples=[]
    wss = []
    for i in range(0, 17):
        generated_sample = generated_samples[:, i]
        ws = wasserstein_distance(dff_test_reco.values[:, i], generated_sample)
        wss.append(ws)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(dff_test_reco.values[:, i], histtype='step', label='FullSim', lw=1, bins=100)
        generated_sample = np.where(generated_sample < rangeR.min(), rangeR.min(), generated_sample)
        generated_sample = np.where(generated_sample > rangeR.max(), rangeR.max(), generated_sample)
        rescaled_generated_samples.append(generated_sample)
        ax1.hist(generated_sample, bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()], label=f'FlashSim, ws={round(ws, 4)}')
        fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
        ax1.legend(frameon=False, loc='upper right')

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_yscale("log")
        ax2.hist(dff_test_reco.values[:, i], histtype='step', lw=1, bins=100)
        ax2.hist(generated_sample, bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()])
        writer.add_figure(fig, global_step=epoch)
    plt.close()

    jet_cond = ["MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour",]

    df = pd.DataFrame(data=dff_test_gen, columns=jet_cond)

    rescaled_generated_samples = np.array(rescaled_generated_samples)
    rescaled_generated_samples = np.swapaxes(rescaled_generated_samples, 0, 1)
    totalj = rescaled_generated_samples

    totalj[:, 7] = totalj[:, 7] + df['GenJet_eta'].values
    totalj[:, 9] = totalj[:, 9] * df['GenJet_mass'].values
    totalj[:, 11] = totalj[:, 11] +  df['GenJet_phi'].values
    totalj[:, 11]= np.where( totalj[:, 11]< -np.pi, totalj[:, 11] + 2*np.pi, totalj[:, 11])
    totalj[:, 11]= np.where( totalj[:, 11]> np.pi, totalj[:, 11] - 2*np.pi, totalj[:, 11])
    totalj[:, 12] = totalj[:, 12] * df['GenJet_pt'].values

    total = dff_test_reco.values
    total[:, 7] = total[:, 7] + df['GenJet_eta'].values
    total[:, 9] = total[:, 9] * df['GenJet_mass'].values
    total[:, 11] = total[:, 11] +  df['GenJet_phi'].values
    total[:, 11]= np.where( total[:, 11]< -np.pi, total[:, 11] + 2*np.pi, total[:, 11])
    total[:, 11]= np.where( total[:, 11]> np.pi, total[:, 11] - 2*np.pi, total[:, 11])
    total[:, 12] = total[:, 12] * df['GenJet_pt'].values


    for i in range(0, 17):
        generated_sample = totalj[:, i]
        ws = wasserstein_distance(total[:, i], generated_sample)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

        _, rangeR, _ = ax1.hist(total[:, i], histtype='step', lw=1, bins=100, label='FullSim')
        
        ax1.hist(generated_sample, bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()], label=f'FlashSim, ws = {ws}')
        fig.suptitle(f"Comparison of {names[i]}", fontsize=16)
        ax1.legend(fontsize=16, frameon=False)

        ax2.set_yscale("log")
        ax2.hist(total[:, i], histtype='step', lw=1, bins=100)
        ax2.hist(generated_sample, bins=100,  histtype='step', lw=1,
                range=[rangeR.min(), rangeR.max()], label=f'ws = {ws}')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        writer.add_figure(fig, global_step=epoch)
    plt.close()


    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    fig = corner.corner(total[:, [1, 2, 3, 4, 5]], labels=['Jet_btagCMVA', 'Jet_btagCSVV2', 'Jet_btagDeepB', 'Jet_btagDeepC', 'Jet_btagDeepFlavB'], color='tab:blue',
                        levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    corner.corner(rescaled_generated_samples[:, [1, 2, 3, 4, 5]], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=24, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 4.0), loc='upper right')
    plt.suptitle('Jet tagging distributions correlations', fontsize=20)
    writer.add_figure(fig, global_step=epoch)
    plt.close()


    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    fig = corner.corner(total[:, [13, 10]], bins=40, labels=['Jet_qgl', 'Jet_nConstituents'], color='tab:blue', smooth1d=0.5,
                        levels=(0.5,0.9, 0.99), hist_bin_factor=1, scale_hist=True, plot_datapoints=False)
    corner.corner(rescaled_generated_samples[:, [13, 10]], bins=40, levels=[0.5, 0.9, 0.99], hist_bin_factor=1, color='tab:orange', smooth1d=0.5,
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    plt.suptitle('qgl and nConstituens correlations', fontsize=16
                )
    writer.add_figure(fig, global_step=epoch)
    plt.close()



    limited_pt = total[:, 12]
    limited_ptj = totalj[:, 12]
    gen = df.loc[:, 'GenJet_pt'].values
    limited = np.vstack([gen[:len(limited_pt)], limited_pt]).T
    limitedj = np.vstack([gen[:len(limited_ptj)], limited_ptj]).T



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)

    _, rangeR, _ = ax1.hist(limited_pt, histtype='step', lw=1, bins=20, range=[0, 100])

    ax1.hist(limited_ptj, bins=20,  histtype='step', lw=1,
            range=[rangeR.min(), rangeR.max()], label=f'ws = {ws}')
    fig.suptitle(f"Comparison of {names[i]}")

    ax2.set_yscale("log")
    ax2.hist(limited_pt, histtype='step', lw=1, bins=20, range=[0, 100])
    ax2.hist(limited_ptj, bins=20,  histtype='step', lw=1,
            range=[rangeR.min(), rangeR.max()], label=f'ws = {ws}')
    writer.add_figure(fig, global_step=epoch)
    plt.close()



    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    fig = corner.corner(total[:, [12, 9]], range=[(0, 100), (0,40)], labels=['Jet_pt [GeV]', 'Jet_mass [GeV]'], color='tab:blue',
                        levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    corner.corner(totalj[:, [12, 9]], range=[(0, 100), (0,40)], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    plt.suptitle(r'p$_T$ and mass correlations', fontsize=16
                )
    writer.add_figure(fig, global_step=epoch)
    plt.close()


    blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    fig = corner.corner(limited, range=[(0,100), (0,100)], labels=['GenJet_pt [GeV]', 'Jet_pt [GeV]'], color='tab:blue',
                        levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    corner.corner(limitedj, range=[(0,100), (0,100)], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
                scale_hist=True, plot_datapoints=False, fig=fig)
    plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    plt.suptitle(r'GenJet_p$_T$ and p$_T$ correlations', fontsize=16
             )
    writer.add_figure(fig, global_step=epoch)
    plt.close()


    n = 30
    y = rescaled_generated_samples[:, 12].flatten()
    x = gen.flatten()
    x_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n)
    y_bins = np.logspace(np.log10(y.min()), np.log10(y.max()), n)
    H, xen, yen = np.histogram2d(x, y, bins=[x_bins, y_bins])

    xbinwn = xen[1]-xen[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    x_slice_mean, x_slice_rms = [], []
    for i,b in enumerate(xen[:-1]):
        x_slice_mean.append( y[ (x>xen[i]) & (x<=xen[i+1]) ].mean())
        x_slice_rms.append( y[ (x>xen[i]) & (x<=xen[i+1]) ].std())
        
    x_slice_mean_nano = np.array(x_slice_mean)
    x_slice_rms_nano = np.array(x_slice_rms)

    n = 30
    y = dff_test_reco.values[:, 12].flatten()
    x = gen.flatten()
    x_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n)
    y_bins = np.logspace(np.log10(y.min()), np.log10(y.max()), n)
    H, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])

    xbinw = xe[1]-xe[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    x_slice_mean, x_slice_rms = [], []
    for i,b in enumerate(xe[:-1]):
        x_slice_mean.append( y[ (x>xe[i]) & (x<=xe[i+1]) ].mean())
        x_slice_rms.append( y[ (x>xe[i]) & (x<=xe[i+1]) ].std())
        
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
    ax1.errorbar(xen[:-1]+ xbinwn/2, x_slice_mean, x_slice_rms_nano, marker='o', fmt='_',  label='FullSim')
    ax1.errorbar(xe[:-1]+ xbinw/2, x_slice_mean, x_slice_rms, marker='o', fmt='_', label="FlashSim")
    ax1.set_title(r"Profile histogram for GenJet_p$_T$ vs p$_T$Ratio", fontsize=16)
    ax1.set_xscale('log')
    ax1.set_ylabel(r"p$_T$Ratio", fontsize=16)
    ax1.set_xlim([10,1000])
    ax2.errorbar(xen[:-1]+ xbinwn/2, x_slice_rms_nano)
    ax2.errorbar(xe[:-1]+ xbinw/2, x_slice_rms)
    ax1.legend(fontsize=16, frameon=False)
    ax2.set_xscale('log')
    ax2.set_xlim([10,1000])
    ax2.set_xlabel(r"GenJet_p$_T$ [GeV]", fontsize=16)
    ax2.set_ylabel("RMS", fontsize=16)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    writer.add_figure(fig, global_step=epoch)
    plt.close()


    def histANDroc(gen, gen_df):
        truth = np.abs(gen_df.values)
        mask_b = np.where(truth[:, 12]==5)
        mask_uds = np.where((truth[:, 12]==1) | (truth[:, 12]==2) | (truth[:, 12]==3))
        print(truth[:, 12])
        bs = gen[mask_b, 2].flatten()
        nbs = gen[mask_uds, 2].flatten()
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
        plt.title("FlashSim Jet_btagDeepB for b ground truth", fontsize=16)
        plt.legend(fontsize=16, frameon=False, loc='upper left')

        y_bs = np.ones(len(bs))
        y_nbs = np.zeros(len(nbs))
        y_t = np.concatenate((y_bs, y_nbs))
        y_s = np.concatenate((bs, nbs))

        fpr, tpr, _ = roc_curve(y_t.ravel(), y_s.ravel())
        roc_auc = auc(fpr, tpr)

        return figure, fpr, tpr, roc_auc, bs, nbs


    fig, fpr, tpr, roc_auc, bs, nbs = histANDroc(generated_samples, dff_test_gen)
    writer.global_step(fig, global_step=epoch)
    plt.close()

    cfig, cfpr, ctpr, croc_auc, cbs, cnbs  = histANDroc(dff_test_reco.values, dff_test_gen)

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

    plt.yscale("log")
    plt.ylim([0.0005, 1.05])
    plt.xlabel("Efficency for b-jet (TP)", fontsize=16)
    plt.ylabel("Mistagging prob (FP)", fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Receiver operating characteristic", fontsize=16)
    plt.legend(fontsize=16, frameon=False,loc="best")
    writer.add_figure(fig, global_step=epoch)



