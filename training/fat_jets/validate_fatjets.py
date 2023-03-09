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

            z_sampled = model.sample(
                    num_samples=1, context=inputs_y.view(-1, args.y_dim)
                )
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
                "Mfatjet_particleNetMD_XbbvsQCD",]
    
    for i in range(0, args.x_dim):
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
        plt.savefig(f"{save_dir}/{names[i]}.png")
                    
    plt.close()

    # Samples postprocessing 


    jet_cond = ["MgenjetAK8_pt",
                "MgenjetAK8_phi",
                "MgenjetAK8_eta",
                "MgenjetAK8_hadronFlavour",
                "MgenjetAK8_partonFlavour",
                "MgenjetAK8_mass",
                "MgenjetAK8_ncFlavour",
                "MgenjetAK8_nbFlavour",]
    
    df = pd.DataFrame(data=gen, columns=jet_cond)

    samples[:, 1] = samples[:, 1] + df["MgenjetAK8_eta"].values
    # samples[:, 9] = samples[:, 9] * df['GenJet_mass'].values
    samples[:, 2] = samples[:, 2] +  df["MgenjetAK8_phi"].values
    samples[:, 2]= np.where(samples[:, 2]< -np.pi, samples[:, 2] + 2*np.pi, samples[:, 2])
    samples[:, 2]= np.where(samples[:, 2]> np.pi, samples[:, 2] - 2*np.pi, samples[:, 2])
    samples[:, 0] = samples[:, 0] * df["MgenjetAK8_pt"].values

    # Reco postprocessing

    reco[:, 1] = reco[:, 1] + df["MgenjetAK8_eta"].values
    # reco[:, 9] = reco[:, 9] * df['GenJet_mass'].values
    reco[:, 2] = reco[:, 2] +  df["MgenjetAK8_phi"].values
    reco[:, 2]= np.where(reco[:, 2]< -np.pi, reco[:, 2] + 2*np.pi, reco[:, 2])
    reco[:, 2]= np.where(reco[:, 2]> np.pi, reco[:, 2] - 2*np.pi, reco[:, 2])
    reco[:, 0] = reco[:, 0] * df["MgenjetAK8_pt"].values
    # reco[:, 7] = reco[:, 7] + df['GenJet_eta'].values
    # reco[:, 9] = reco[:, 9] * df['GenJet_mass'].values
    # reco[:, 11] = reco[:, 11] +  df['GenJet_phi'].values
    # reco[:, 11]= np.where( reco[:, 11]< -np.pi, reco[:, 11] + 2*np.pi, reco[:, 11])
    # reco[:, 11]= np.where( reco[:, 11]> np.pi, reco[:, 11] - 2*np.pi, reco[:, 11])
    # reco[:, 12] = reco[:, 12] * df['GenJet_pt'].values

    # Plots


    for i in range(0, args.x_dim):
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
        plt.savefig(f"{save_dir}/{names[i]}.png")
                    
    plt.close()

    # # Corner plot

    # blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    # red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    # fig = corner.corner(reco[:, [1, 2, 3, 4, 5]], labels=['Jet_btagCMVA', 'Jet_btagCSVV2', 'Jet_btagDeepB', 'Jet_btagDeepC', 'Jet_btagDeepFlavB'], color='tab:blue',
    #                     levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)

    # corner.corner(samples[:, [1, 2, 3, 4, 5]], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
    #             scale_hist=True, plot_datapoints=False, fig=fig)
    # plt.legend(fontsize=24, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 4.0), loc='upper right')
    # plt.suptitle('Jet tagging distributions correlations', fontsize=20)
    # writer.add_figure("Jet tagging correlations", fig, global_step=epoch)
    # plt.savefig(f"{save_dir}/Jet_tagging_correlations.png")
    # plt.close()

    # blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    # red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    # fig = corner.corner(reco[:, [13, 10]], range=[(0, 1), (0,90)], bins=40, labels=['Jet_qgl', 'Jet_nConstituents'], color='tab:blue', smooth1d=0.5,
    #                     levels=(0.5,0.9, 0.99), hist_bin_factor=1, scale_hist=True, plot_datapoints=False)
    # corner.corner(samples[:, [13, 10]], range=[(0, 1), (0,90)], bins=40, levels=[0.5, 0.9, 0.99], hist_bin_factor=1, color='tab:orange', smooth1d=0.5,
    #             scale_hist=True, plot_datapoints=False, fig=fig)
    # plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    # plt.suptitle('qgl and nConstituens correlations', fontsize=16)
    # writer.add_figure('qgl and nConstituens correlations', fig, global_step=epoch)
    # plt.savefig(f"{save_dir}/qgl_nConstituens_correlations.png")
    # plt.close()

    # blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    # red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    # fig = corner.corner(reco[:, [12, 9]], range=[(0, 100), (0,40)], labels=['Jet_pt [GeV]', 'Jet_mass [GeV]'], color='tab:blue',
    #                     levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    # corner.corner(samples[:, [12, 9]], range=[(0, 100), (0,40)], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
    #             scale_hist=True, plot_datapoints=False, fig=fig)
    # plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    # plt.suptitle(r'p$_T$ and mass correlations', fontsize=16
    #             )
    # writer.add_figure(r'p$_T$ and mass correlations', fig, global_step=epoch)
    # plt.savefig(f"{save_dir}/pt_mass_correlations.png")
    # plt.close()

    # limited_pt = reco[:, 12]
    # limited_ptj = samples[:, 12]
    # x = df.loc[:, 'GenJet_pt'].values
    # limited = np.vstack([x[:len(limited_pt)], limited_pt]).T
    # limitedj = np.vstack([x[:len(limited_ptj)], limited_ptj]).T

    # blue_line = mlines.Line2D([], [], color='tab:blue', label='FullSim')
    # red_line = mlines.Line2D([], [], color='tab:orange', label='FlashSim')
    # fig = corner.corner(limited, range=[(0,100), (0,100)], labels=['GenJet_pt [GeV]', 'Jet_pt [GeV]'], color='tab:blue',
    #                     levels=(0.5,0.9, 0.99), hist_bin_factor=3, scale_hist=True, plot_datapoints=False)
    # corner.corner(limitedj, range=[(0,100), (0,100)], levels=[0.5, 0.9, 0.99], hist_bin_factor=3, color='tab:orange',
    #             scale_hist=True, plot_datapoints=False, fig=fig)
    # plt.legend(fontsize=16, frameon=False, handles=[blue_line,red_line], bbox_to_anchor=(0., 1.0, 1., 1.0), loc='upper right')
    # plt.suptitle(r'GenJet_p$_T$ and p$_T$ correlations', fontsize=16
    #             )
    # writer.add_figure(r'GenJet_p$_T$ and p$_T$ correlations', fig, global_step=epoch)
    # plt.savefig(f"{save_dir}/GenJet_pt_pt_correlations.png")
    # plt.close()    

    # # b-tagging FOM

    # def histANDroc(gen, gen_df):
    #     truth = np.abs(gen_df)
    #     mask_b = np.where(truth[:, 12]==5)
    #     mask_uds = np.where((truth[:, 12]==1) | (truth[:, 12]==2) | (truth[:, 12]==3))
    #     print(truth[:, 12])
    #     bs = gen[mask_b, 2].flatten()
    #     nbs = gen[mask_uds, 2].flatten()
    #     # nbs = nbs[0:len(bs)]

    #     bs = bs[bs >=-0.05]
    #     nbs = nbs[nbs >=-0.05]

    #     bs = np.where(bs<0, 0, bs)
    #     nbs = np.where(nbs<0, 0, nbs)

    #     bs = np.where(bs>1, 1, bs)
    #     nbs = np.where(nbs>1, 1, nbs)

    #     # bs = bs[0:len(nbs)]

    #     figure = plt.figure(figsize=(9, 6.5))
    #     ax = plt.gca()
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     plt.hist(bs.flatten(), bins=50, label="b",  histtype='step', lw=2, color='C2')
    #     plt.hist(nbs.flatten(), bins=50, label="uds",  histtype='step', lw=2, color='C3')
    #     plt.title("FlashSim Jet_btagDeepB for b ground truth", fontsize=16)
    #     plt.legend(fontsize=16, frameon=False, loc='upper left')

    #     y_bs = np.ones(len(bs))
    #     y_nbs = np.zeros(len(nbs))
    #     y_t = np.concatenate((y_bs, y_nbs))
    #     y_s = np.concatenate((bs, nbs))

    #     fpr, tpr, _ = roc_curve(y_t.ravel(), y_s.ravel())
    #     roc_auc = auc(fpr, tpr)

    #     return figure, fpr, tpr, roc_auc, bs, nbs

    # print(gen.shape)
    # fig, fpr, tpr, roc_auc, bs, nbs = histANDroc(samples, gen)
    # cfig, cfpr, ctpr, croc_auc, cbs, cnbs  = histANDroc(reco, gen)

    # fig = plt.figure(figsize=(9, 6.5))
    # lw = 2
    # plt.plot(
    #     tpr,
    #     fpr,
    #     color="C1",
    #     lw=lw,
    #     label=f"ROC curve (area = %0.2f) FlashSim" % roc_auc,
    # )
    
    # plt.plot(
    #     ctpr,
    #     cfpr,
    #     color="C0",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f) FullSim" % croc_auc,
    # )

    # #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # #plt.xlim([0.0, 1.0])
    # plt.yscale("log")
    # plt.ylim([0.0005, 1.05])
    # plt.xlabel("Efficency for b-jet (TP)", fontsize=16)
    # plt.ylabel("Mistagging prob (FP)", fontsize=16)
    # ax = plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.title("Receiver operating characteristic", fontsize=16)
    # plt.legend(fontsize=16, frameon=False,loc="best")
    # writer.add_figure("ROC", fig, global_step=epoch)
    # plt.savefig(f"{save_dir}/ROC.png")
    # plt.close()
