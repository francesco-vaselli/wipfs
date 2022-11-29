import sys
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import uproot

dict = {
    "MGenElectron_eta": (),
    "MGenElectron_phi": (),
    "MGenElectron_pt": (),
    "MGenElectron_charge": (),
    "MGenPartMother_pdgId": (),
    "MGenPartMother_pt": (),
    "MGenPartMother_deta": (),
    "MGenPartMother_dphi": (),
    "MGenElectron_statusFlag0": (),
    "MGenElectron_statusFlag1": (),
    "MGenElectron_statusFlag2": (),
    "MGenElectron_statusFlag3": (),
    "MGenElectron_statusFlag4": (),
    "MGenElectron_statusFlag5": (),
    "MGenElectron_statusFlag6": (),
    "MGenElectron_statusFlag7": (),
    "MGenElectron_statusFlag8": (),
    "MGenElectron_statusFlag9": (),
    "MGenElectron_statusFlag10": (),
    "MGenElectron_statusFlag11": (),
    "MGenElectron_statusFlag12": (),
    "MGenElectron_statusFlag13": (),
    "MGenElectron_statusFlag14": (),
    "ClosestJet_dr": (),
    "ClosestJet_dphi": (),
    "ClosestJet_deta": (),
    "ClosestJet_pt": (),
    "ClosestJet_mass": (),
    "ClosestJet_EncodedPartonFlavour_light": (),
    "ClosestJet_EncodedPartonFlavour_gluon": (),
    "ClosestJet_EncodedPartonFlavour_c": (),
    "ClosestJet_EncodedPartonFlavour_b": (),
    "ClosestJet_EncodedPartonFlavour_undefined": (),
    "ClosestJet_EncodedHadronFlavour_b": (),
    "ClosestJet_EncodedHadronFlavour_c": (),
    "ClosestJet_EncodedHadronFlavour_light": (),
    "MElectron_charge": (),
    "MElectron_convVeto": (),
    "MElectron_cutBased": (),
    "MElectron_cutBased_Fall17_V1": (),
    "MElectron_dr03TkSumPt": (),
    "MElectron_dr03TkSumPtHEEP": (),
    "MElectron_dxy": (),
    "MElectron_dxyErr": (),
    "MElectron_dz": (),
    "MElectron_dzErr": (),
    "MElectron_eCorr": (),
    "MElectron_eInvMinusPInv": (),
    "MElectron_energyErr": (),
    "MElectron_etaMinusGen": (),
    "MElectron_hoe": (),
    "MElectron_ip3d": (),
    "MElectron_isPFcand": (),
    "MElectron_jetPtRelv2": (),
    "MElectron_jetRelIso": (),
    "MElectron_lostHits": (),
    "MElectron_miniPFRelIso_all": (),
    "MElectron_miniPFRelIso_chg": (),
    "MElectron_mvaFall17V1Iso": (),
    "MElectron_mvaFall17V1Iso_WP80": (),
    "MElectron_mvaFall17V1Iso_WP90": (),
    "MElectron_mvaFall17V1Iso_WPL": (),
    "MElectron_mvaFall17V1noIso": (),
    "MElectron_mvaFall17V1noIso_WP80": (),
    "MElectron_mvaFall17V1noIso_WP90": (),
    "MElectron_mvaFall17V1noIso_WPL": (),
    "MElectron_mvaFall17V2Iso": (),
    "MElectron_mvaFall17V2Iso_WP80": (),
    "MElectron_mvaFall17V2Iso_WP90": (),
    "MElectron_mvaFall17V2Iso_WPL": (),
    "MElectron_mvaFall17V2noIso": (),
    "MElectron_mvaFall17V2noIso_WP80": (),
    "MElectron_mvaFall17V2noIso_WP90": (),
    "MElectron_mvaFall17V2noIso_WPL": (),
    "MElectron_mvaTTH": (),
    "MElectron_pfRelIso03_all": (),
    "MElectron_pfRelIso03_chg": (),
    "MElectron_phiMinusGen": (),
    "MElectron_ptRatio": (),
    "MElectron_r9": (),
    "MElectron_seedGain": (),
    "MElectron_sieie": (),
    "MElectron_sip3d": (),
    "MElectron_tightCharge": (),
    "MElectron_vidNestedWPBitmap0": (),
    "MElectron_vidNestedWPBitmap1": (),
    "MElectron_vidNestedWPBitmap2": (),
    "MElectron_vidNestedWPBitmap3": (),
    "MElectron_vidNestedWPBitmap4": (),
    "MElectron_vidNestedWPBitmap5": (),
    "MElectron_vidNestedWPBitmap6": (),
    "MElectron_vidNestedWPBitmap7": (),
    "MElectron_vidNestedWPBitmap8": (),
    "MElectron_vidNestedWPBitmap9": (),
    "MElectron_vidNestedWPBitmapHEEP0": (),
    "MElectron_vidNestedWPBitmapHEEP1": (),
    "MElectron_vidNestedWPBitmapHEEP2": (),
    "MElectron_vidNestedWPBitmapHEEP3": (),
    "MElectron_vidNestedWPBitmapHEEP4": (),
    "MElectron_vidNestedWPBitmapHEEP5": (),
    "MElectron_vidNestedWPBitmapHEEP6": (),
    "MElectron_vidNestedWPBitmapHEEP7": (),
    "MElectron_vidNestedWPBitmapHEEP8": (),
    "MElectron_vidNestedWPBitmapHEEP9": (),
    "MElectron_vidNestedWPBitmapHEEP10": (),
    "MElectron_vidNestedWPBitmapHEEP11": (),
    "Pileup_gpudensity": (),
    "Pileup_nPU": (),
    "Pileup_nTrueInt": (),
    "Pileup_pudensity": (),
    "Pileup_sumEOOT": (),
    "Pileup_sumLOOT": (),
}


def print_dictionary(tree):

    vars_to_save = tree.keys()

    dict = {name: () for name in vars_to_save}

    for column_name, operation in dict.items():
        print(f"'{column_name}': {operation},")


def preprocessing(operation_dict, df):

    # for column_name, operations in operation_dict.items():

    return


def one_hot_maker(df, column_names, new_name):

    y = LabelBinarizer().fit_transform(df[column_names])
    df = df.drop(column_names, axis=1)
    df[new_name] = y

    return df


if __name__ == "__main__":

    f = sys.argv[1]
    tree = uproot.open(f"MElectrons_v{f}.root:MElectrons")

    # print_dictionary(tree)

    df = (
        tree.arrays(entry_stop=10, library="pd")
        .reset_index(drop=True)
        .astype("float32")
        .dropna()
    )

    df = one_hot_maker(df,[
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
        ],
        "MGenElectron_statusFlags_one_hot"
    )

    print(df["MGenElectron_statusFlags_one_hot"])

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
