import json
import numpy as np
import pandas as pd
import h5py

from matplotlib import pyplot as plt

from post_actions import target_dictionary


def restore_range(column_name, scale_dict, df):
    """
    Restore data range to the original value before dividing by max
    """
    scale = scale_dict[column_name]
    df[column_name] = df[column_name] * scale
    return df[column_name]


def inverse_transform(df, column_name, function, p):

    df[column_name] = df[column_name].apply(lambda x: (function(x) - p[1]) / p[0])
    return df[column_name]


def unsmearing(df, column_name, interval):
    """Unsmearing for in variables. We have gaussian and uniform smearing.
    If we have interval, that means that we built a fake gaussian dataset 
    in the selected interval, and then we just have to compute the sample mean
    in this range.
    """
    val = df[column_name].values
    if interval != None:
        mask_condition = np.logical_and(val >= interval[0], val <= interval[1])

        # Assuming that the value to be unsmeared is always np.log(1e-3),
        # which corresponds to an int value of 0 after a np.log(x + 1e-3)
        # transformation

        val[mask_condition] = np.log(1e-3) 
    else:
        df[column_name] = np.rint(df[column_name].values)
    return df[column_name]

def cut_unsmearing(df, column_name, cut, x1, x2):

    val = df[column_name].values
    df[column_name] = np.where(val < cut, x1, x2)
    return df[column_name]


def process_column_var(column_name, operations, df):

    for op in operations:

        if op[0] == "d":
            mask_condition = op[1]
            df[column_name] = unsmearing(df, column_name, mask_condition)

        elif op[0] == "c":
            cut = op[1]
            vals = op[2]
            df[column_name] = cut_unsmearing(df, column_name, cut, *vals)

        elif op[0] == "i":
            function = op[1]
            p = op[2]
            df[column_name] = inverse_transform(df, column_name, function, p)

        else:
            return df[column_name]
    return df[column_name]


def postprocessing(df, vars_dictionary):
    """
    Postprocessing general function given any dataframe and its dictionary
    """

    with open("scale_factors.json") as scale_file:
        scale_dict = json.load(scale_file)

    for column_name, operation in vars_dictionary.items():
        df[column_name] = restore_range(column_name, scale_dict, df)
        df[column_name] = process_column_var(column_name, operation, df)

    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    return df


def postprocessing_test(df, vars_dictionary):
    """
    Preprocessing general function given any dataframe and its dictionary
    """
    with open("scale_factors.json") as scale_file:
        scale_dict = json.load(scale_file)

    for column_name, operation in vars_dictionary.items():
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"{column_name}")
        axs[0].hist(df[column_name], bins=30, histtype="step")
        df[column_name] = restore_range(column_name, scale_dict, df)        
        df[column_name] = process_column_var(column_name, operation, df)
        axs[1].hist(df[column_name], bins=30, histtype="step")
        plt.savefig(f"figures_post/{column_name}.pdf", format="pdf")
        plt.close()  # produces MatplotlibDeprecationWarning. It is a bug (https://github.com/matplotlib/matplotlib/issues/23921)

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    return df


gen_columns = [
    "MGenElectron_eta",
    "MGenElectron_phi",
    "MGenElectron_pt",
    "MGenElectron_charge",
    "MGenPartMother_pdgId",
    "MGenPartMother_pt",
    "MGenPartMother_deta",
    "MGenPartMother_dphi",
    "MGenElectron_statusFlag0",
    "MGenElectron_statusFlag1",
    "MGenElectron_statusFlag2",
    "MGenElectron_statusFlag3",
    "MGenElectron_statusFlag4",
    "MGenElectron_statusFlag5",
    "MGenElectron_statusFlag6",
    "MGenElectron_statusFlag7",
    "MGenElectron_statusFlag8",
    "MGenElectron_statusFlag9",
    "MGenElectron_statusFlag10",
    "MGenElectron_statusFlag11",
    "MGenElectron_statusFlag12",
    "MGenElectron_statusFlag13",
    "MGenElectron_statusFlag14",
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
]

reco_columns = [
    "MElectron_charge",
    "MElectron_convVeto",
    "MElectron_deltaEtaSC",
    "MElectron_dr03EcalRecHitSumEt",
    "MElectron_dr03HcalDepth1TowerSumEt",
    "MElectron_dr03TkSumPt",
    "MElectron_dr03TkSumPtHEEP",
    "MElectron_dxy",
    "MElectron_dxyErr",
    "MElectron_dz",
    "MElectron_dzErr",
    "MElectron_eInvMinusPInv",
    "MElectron_energyErr",
    "MElectron_etaMinusGen",
    "MElectron_hoe",
    "MElectron_ip3d",
    "MElectron_isPFcand",
    "MElectron_jetPtRelv2",
    "MElectron_jetRelIso",
    "MElectron_lostHits",
    "MElectron_miniPFRelIso_all",
    "MElectron_miniPFRelIso_chg",
    "MElectron_mvaFall17V1Iso",
    "MElectron_mvaFall17V1Iso_WP80",
    "MElectron_mvaFall17V1Iso_WP90",
    "MElectron_mvaFall17V1Iso_WPL",
    "MElectron_mvaFall17V1noIso",
    "MElectron_mvaFall17V1noIso_WP80",
    "MElectron_mvaFall17V1noIso_WP90",
    "MElectron_mvaFall17V1noIso_WPL",
    "MElectron_mvaFall17V2Iso",
    "MElectron_mvaFall17V2Iso_WP80",
    "MElectron_mvaFall17V2Iso_WP90",
    "MElectron_mvaFall17V2Iso_WPL",
    "MElectron_mvaFall17V2noIso",
    "MElectron_mvaFall17V2noIso_WP80",
    "MElectron_mvaFall17V2noIso_WP90",
    "MElectron_mvaFall17V2noIso_WPL",
    "MElectron_mvaTTH",
    "MElectron_pfRelIso03_all",
    "MElectron_pfRelIso03_chg",
    "MElectron_phiMinusGen",
    "MElectron_ptRatio",
    "MElectron_r9",
    "MElectron_seedGain",
    "MElectron_sieie",
    "MElectron_sip3d",
    "MElectron_tightCharge"
]

if __name__ == "__main__":

    f = h5py.File("MElectrons.hdf5", "r")
    df = pd.DataFrame(data=f.get("data"), columns=gen_columns+reco_columns)
    f.close()

    df = postprocessing_test(df, target_dictionary)

    file = h5py.File(f"MElectrons_post.hdf5", "w")
    dset = file.create_dataset("data", data=df.values, dtype="f4")
    file.close()