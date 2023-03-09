# the NanoBuilder function, takings as input a NanoAOD file, extracting gen-level information for conditionig,
# generating a new event with the same topology and saving results in a NanoAOD-like .root file

import ROOT
import uproot
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time
import awkward as ak
import os
import sys

dirpath = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(dirpath, "..", "..", "models"))
sys.path.insert(0, os.path.join(dirpath, "..", "..", "utils"))
sys.path.insert(0, os.path.join(dirpath, "..", "..", "training", "electrons"))


from postprocessing import postprocessing, reco_columns
from post_actions import target_dictionary


class GenDS(Dataset):
    """A dumb dataset for storing gen-conditioning for generation

    Args:
            Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, df, cond_vars):

        y = df.loc[:, cond_vars].values
        self.y_train = torch.tensor(y, dtype=torch.float32)  # .to(device)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.y_train[idx]


# really important step: use ROOT interpreter to call c++ code directly from python
# execute only selection of Gen objects (no longer requires matching as we are not training)
ROOT.gInterpreter.ProcessLine('#include "gens.h"')


def nbd(ele_model, root, file_path, new_root):
    """The NanoBuilder function

    Args:
            ele_model (pytorch model): the trained NF net for jet generation
            root (string): old root for file providing the gen conditioning and event topology
            file_path (string): gen inputs file name
            new_root (string): new root for saving output file
    """
    # select nano aod, process and save intermmediate files to disk
    s = str(os.path.join(root, file_path))
    ROOT.gens(s)
    print("done saving intermidiate file")

    # define list of names for conditioning

    ele_cond = [
        "GenElectron_eta",
        "GenElectron_phi",
        "GenElectron_pt",
        "GenElectron_charge",
        "GenElectron_statusFlag0",
        "GenElectron_statusFlag1",
        "GenElectron_statusFlag2",
        "GenElectron_statusFlag3",
        "GenElectron_statusFlag4",
        "GenElectron_statusFlag5",
        "GenElectron_statusFlag6",
        "GenElectron_statusFlag7",
        "GenElectron_statusFlag8",
        "GenElectron_statusFlag9",
        "GenElectron_statusFlag10",
        "GenElectron_statusFlag11",
        "GenElectron_statusFlag12",
        "GenElectron_statusFlag13",
        "GenElectron_statusFlag14",
        "ClosestJet_dr",
        "ClosestJet_dphi",
        "ClosestJet_deta",
        "ClosestJet_pt",
        "ClosestJet_mass",
        "ClosestJet_EncodedPartonFlavour_light",
        "ClosestJet_EncodedPartonFlavour_gluon",
        "ClosestJet_EncodedPartonFlavour_c",
        "ClosestJet_EncodedPartonFlavour_b",
        "ClosestJet_EncodedPartonFlavour_undefined",
        "ClosestJet_EncodedHadronFlavour_b",
        "ClosestJet_EncodedHadronFlavour_c",
        "ClosestJet_EncodedHadronFlavour_light",
        "Pileup_gpudensity",
        "Pileup_nPU",
        "Pileup_nTrueInt",
        "Pileup_pudensity",
        "Pileup_sumEOOT",
        "Pileup_sumLOOT",
        "event",
        "run",
    ]

    # read processed files for jets and save event structure
    tree = uproot.open("testGens.root:Gens", num_workers=20)

    # read jet data to df
    df = tree.arrays(ele_cond, library="pd").astype("float32").dropna()
    print(df)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # crucial step: save original multiindex structure to restructure outputs later
    ele_ev_index = np.unique(df.index.get_level_values(0).values)
    events_structure_ele = (
        df.reset_index(level=1).index.value_counts().sort_index().values
    )
    print(events_structure_ele)
    print(len(events_structure_ele))
    print(sum(events_structure_ele))

    # reset dataframe index for performing 1to1 generation
    df.reset_index(drop=True)

    # save gen-level charges for matching them later to the event
    charges = np.reshape(
        df["GenElectron_charge"].values, (len(df["GenElectron_charge"].values), 1)
    )

    # read global event info to df
    dfe = tree.arrays(["event", "run"], library="pd").astype(np.longlong).dropna()
    print(dfe)
    print(f"Total number of events is {len(dfe)}")
    dfe = dfe[~dfe.isin([np.nan, np.inf, -np.inf]).any(1)]
    # in this case we are directly saving the values (only 1 value per event)
    events_structure = dfe.values
    print(events_structure.shape, events_structure.shape)

    # if some event is missing an object, we must set the missing entry to 0 manually
    # to keep a consistent structure
    # adjust structure if some events have no jets

    zeros = np.zeros(len(dfe), dtype=int)
    print(len(ele_ev_index), len(events_structure_ele))
    np.put(zeros, ele_ev_index, events_structure_ele, mode="rise")
    events_structure_ele = zeros
    print(events_structure_ele.shape, events_structure_ele)
    print(sum(events_structure_ele))

    # define datasets
    ele_dataset = GenDS(df, ele_cond)

    # start electrons 1to1 generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 10000
    ele_loader = DataLoader(
        ele_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=20,
    )
    flow = ele_model

    tot_sample = []
    leftover_sample = []
    times = []

    for batch_idx, y in enumerate(ele_loader):

        y = y.float().to(device, non_blocking=True)
        # Prints y device
        print(y.device)
        # Compute log prob
        # print(y.shape)
        if len(y) == batch_size:
            start = time.time()
            sample = flow.sample(1, context=y)
            taken = time.time() - start
            print(f"Done {batch_size} data in {taken}s")
            times.append(taken)
            sample = sample.detach().cpu().numpy()
            sample = np.squeeze(sample, axis=1)
            # print(sample.shape)
            tot_sample.append(sample)

        else:
            leftover_shape = len(y)
            sample = flow.sample(1, context=y)
            sample = sample.detach().cpu().numpy()
            sample = np.squeeze(sample, axis=1)
            # print(sample.shape)
            leftover_sample.append(sample)

    print(np.mean(times))
    tot_sample = np.array(tot_sample)
    tot_sample = np.reshape(tot_sample, ((len(ele_loader) - 1) * batch_size, 48))
    leftover_sample = np.array(leftover_sample)
    leftover_sample = np.reshape(leftover_sample, (leftover_shape, 48))
    total = np.concatenate((tot_sample, leftover_sample), axis=0)

    total = pd.DataFrame(total, columns=reco_columns)

    total = postprocessing(total, target_dictionary)

    total["MElectron_pt"] = total["MElectron_ptRatio"] * df["GenElectron_pt"]
    total["MElectron_eta"] = total["MElectron_etaMinusGen"] + df["GenElectron_eta"]
    total["MElectron_phi"] = total["MElectron_phiMinusGen"] + df["GenElectron_phi"]

    # Charge: in this branch charge is also a target variable, so we already have it in total dataframe
    # For future: I should use the following code

    # total = np.concatenate((total, charges), axis=1)

    # convert to akw arrays for saving to file with correct event structure
    ele_names = [
        "charge",
        "convVeto",
        "deltaEtaSC",
        "dr03EcalRecHitSumEt",
        "dr03HcalDepth1TowerSumEt",
        "dr03TkSumPt",
        "dr03TkSumPtHEEP",
        "dxy",
        "dxyErr",
        "dz",
        "dzErr",
        "eInvMinusPInv",
        "energyErr",
        "etaMinusGen",
        "hoe",
        "ip3d",
        "isPFcand",
        "jetPtRelv2",
        "jetRelIso",
        "lostHits",
        "miniPFRelIso_all",
        "miniPFRelIso_chg",
        "mvaFall17V1Iso",
        "mvaFall17V1Iso_WP80",
        "mvaFall17V1Iso_WP90",
        "mvaFall17V1Iso_WPL",
        "mvaFall17V1noIso",
        "mvaFall17V1noIso_WP80",
        "mvaFall17V1noIso_WP90",
        "mvaFall17V1noIso_WPL",
        "mvaFall17V2Iso",
        "mvaFall17V2Iso_WP80",
        "mvaFall17V2Iso_WP90",
        "mvaFall17V2Iso_WPL",
        "mvaFall17V2noIso",
        "mvaFall17V2noIso_WP80",
        "mvaFall17V2noIso_WP90",
        "mvaFall17V2noIso_WPL",
        "mvaTTH",
        "pfRelIso03_all",
        "pfRelIso03_chg",
        "phiMinusGen",
        "ptRatio",
        "r9",
        "seedGain",
        "sieie",
        "sip3d",
        "tightCharge",
    ]
    to_ttree = dict(zip(ele_names, total.T))
    to_ttree = ak.unflatten(ak.Array(to_ttreej), events_structure_ele)

    to_ttreee = dict(zip(["event", "run"], events_structure.T))
    to_ttreee = ak.Array(to_ttreee)

    # use uproot recreate to save directly akw arrays to .root file
    new_path = str(os.path.join(new_root, file_path))
    new_path = os.path.splitext(new_path)[0]
    with uproot.recreate(f"{new_path}_synth.root") as file:
        file["Events"] = {
            "Electron": to_ttree,
            "event": to_ttreee.event,
            "run": to_ttreee.run,
        }

    return
