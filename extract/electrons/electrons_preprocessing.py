import sys
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import uproot

np.random.seed(0)

dict = {
    "MGenElectron_eta": ([-0.5, 0.5], False, False),
    "MGenElectron_phi": (False, False, False),
    "MGenElectron_pt": (False, False, False),
    "MGenElectron_charge": (False, False, False),
    "MGenPartMother_pdgId": (False, False, False),
    "MGenPartMother_pt": (False, False, False),
    "MGenPartMother_deta": (False, False, False),
    "MGenPartMother_dphi": (False, False, False),
    "MGenElectron_statusFlag0": (False, 0.1, False),
    "MGenElectron_statusFlag1": (False, False, False),
    "MGenElectron_statusFlag2": (False, False, False),
    "MGenElectron_statusFlag3": (False, False, False),
    "MGenElectron_statusFlag4": (False, False, False),
    "MGenElectron_statusFlag5": (False, False, False),
    "MGenElectron_statusFlag6": (False, False, False),
    "MGenElectron_statusFlag7": (False, False, False),
    "MGenElectron_statusFlag8": (False, False, False),
    "MGenElectron_statusFlag9": (False, False, False),
    "MGenElectron_statusFlag10": (False, False, False),
    "MGenElectron_statusFlag11": (False, False, False),
    "MGenElectron_statusFlag12": (False, False, False),
    "MGenElectron_statusFlag13": (False, False, False),
    "MGenElectron_statusFlag14": (False, False, False),
    "ClosestJet_dr": (False, False, False),
    "ClosestJet_dphi": (False, False, False),
    "ClosestJet_deta": (False, False, False),
    "ClosestJet_pt": (False, False, False),
    "ClosestJet_mass": (False, False, False),
    "ClosestJet_EncodedPartonFlavour_light": (False, False, False),
    "ClosestJet_EncodedPartonFlavour_gluon": (False, False, False),
    "ClosestJet_EncodedPartonFlavour_c": (False, False, False),
    "ClosestJet_EncodedPartonFlavour_b": (False, False, False),
    "ClosestJet_EncodedPartonFlavour_undefined": (False, False, False),
    "ClosestJet_EncodedHadronFlavour_b": (False, False, False),
    "ClosestJet_EncodedHadronFlavour_c": (False, False, False),
    "ClosestJet_EncodedHadronFlavour_light": (False, False, False),
    "MElectron_charge": (False, False, False),
    "MElectron_convVeto": (False, False, False),
    "MElectron_cutBased": (False, False, False),
    "MElectron_cutBased_Fall17_V1": (False, False, False),
    "MElectron_dr03TkSumPt": (False, False, False),
    "MElectron_dr03TkSumPtHEEP": (False, False, False),
    "MElectron_dxy": (False, False, False),
    "MElectron_dxyErr": (False, False, False),
    "MElectron_dz": (False, False, False),
    "MElectron_dzErr": (False, False, False),
    "MElectron_eCorr": (False, False, False),
    "MElectron_eInvMinusPInv": (False, False, False),
    "MElectron_energyErr": (False, False, False),
    "MElectron_etaMinusGen": (False, False, False),
    "MElectron_hoe": (False, False, False),
    "MElectron_ip3d": (False, False, False),
    "MElectron_isPFcand": (False, False, False),
    "MElectron_jetPtRelv2": (False, False, False),
    "MElectron_jetRelIso": (False, False, False),
    "MElectron_lostHits": (False, False, False),
    "MElectron_miniPFRelIso_all": (False, False, False),
    "MElectron_miniPFRelIso_chg": (False, False, False),
    "MElectron_mvaFall17V1Iso": (False, False, False),
    "MElectron_mvaFall17V1Iso_WP80": (False, False, False),
    "MElectron_mvaFall17V1Iso_WP90": (False, False, False),
    "MElectron_mvaFall17V1Iso_WPL": (False, False, False),
    "MElectron_mvaFall17V1noIso": (False, False, False),
    "MElectron_mvaFall17V1noIso_WP80": (False, False, False),
    "MElectron_mvaFall17V1noIso_WP90": (False, False, False),
    "MElectron_mvaFall17V1noIso_WPL": (False, False, False),
    "MElectron_mvaFall17V2Iso": (False, False, False),
    "MElectron_mvaFall17V2Iso_WP80": (False, False, False),
    "MElectron_mvaFall17V2Iso_WP90": (False, False, False),
    "MElectron_mvaFall17V2Iso_WPL": (False, False, False),
    "MElectron_mvaFall17V2noIso": (False, False, False),
    "MElectron_mvaFall17V2noIso_WP80": (False, False, False),
    "MElectron_mvaFall17V2noIso_WP90": (False, False, False),
    "MElectron_mvaFall17V2noIso_WPL": (False, False, False),
    "MElectron_mvaTTH": (False, False, False),
    "MElectron_pfRelIso03_all": (False, False, False),
    "MElectron_pfRelIso03_chg": (False, False, False),
    "MElectron_phiMinusGen": (False, False, False),
    "MElectron_ptRatio": (False, False, False),
    "MElectron_r9": (False, False, False),
    "MElectron_seedGain": (False, False, False),
    "MElectron_sieie": (False, False, False),
    "MElectron_sip3d": (False, False, False),
    "MElectron_tightCharge": (False, False, False),
    "MElectron_vidNestedWPBitmap0": (False, False, False),
    "MElectron_vidNestedWPBitmap1": (False, False, False),
    "MElectron_vidNestedWPBitmap2": (False, False, False),
    "MElectron_vidNestedWPBitmap3": (False, False, False),
    "MElectron_vidNestedWPBitmap4": (False, False, False),
    "MElectron_vidNestedWPBitmap5": (False, False, False),
    "MElectron_vidNestedWPBitmap6": (False, False, False),
    "MElectron_vidNestedWPBitmap7": (False, False, False),
    "MElectron_vidNestedWPBitmap8": (False, False, False),
    "MElectron_vidNestedWPBitmap9": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP0": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP1": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP2": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP3": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP4": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP5": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP6": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP7": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP8": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP9": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP10": (False, False, False),
    "MElectron_vidNestedWPBitmapHEEP11": (False, False, False),
    "Pileup_gpudensity": (False, False, False),
    "Pileup_nPU": (False, False, False),
    "Pileup_nTrueInt": (False, False, False),
    "Pileup_pudensity": (False, False, False),
    "Pileup_sumEOOT": (False, False, False),
    "Pileup_sumLOOT": (False, False, False),
}


def print_dictionary(tree):

    vars_to_save = tree.keys()

    dict = {name: (False, False, False) for name in vars_to_save}

    for column_name, operation in dict.items():
        print(f"'{column_name}': {operation},")


def preprocessing(column_name, operation, df):

    print(f"{column_name}:")
    if operation[0]:
        range = operation[0]
        print(f"Saturating in range {range}...")
        val = df[column_name].values
        df[column_name] = np.where(val < range[0], range[0], val)
        val = df[column_name].values
        df[column_name] = np.where(val > range[1], range[1], val)
        print("Done")

    if operation[1]:
        sigma = operation[1]
        print(f"Gaussian smearing with sigma {sigma}")
        df[column_name] = df[column_name].apply(
            lambda x: x + sigma * np.random.normal()
        )
        print("Done")

    if operation[2]:
        func = operation[2][0]
        p = operation[2][1]
        print(f"Applying {func}(x * {p[0]} + {p[1]})")
        df[column_name] = df[column_name].apply(lambda x: func(x * p[0] + p[1]))
        print("Done")

    return df[column_name]

if __name__ == "__main__":

    f = sys.argv[1]
    tree = uproot.open(f"MElectrons_v{f}.root:MElectrons")

    print_dictionary(tree)

    df = (
        tree.arrays(entry_stop=1000, library="pd")
        .reset_index(drop=True)
        .astype("float32")
        .dropna()
    )

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    for column_name, operation in dict.items():
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"{column_name}")
        axs[0].hist(df[column_name], bins=50, histtype='step')
        df[column_name] = preprocessing(column_name, operation, df)
        axs[1].hist(df[column_name], bins=50, histtype='step')
        plt.savefig(f"preprocessing_fig/{column_name}.pdf", format="pdf")
        plt.close(fig)
