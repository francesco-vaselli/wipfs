import numpy as np
import pandas as pd


def transform(df, column_name, function, p):

    df[column_name] = df[column_name].apply(lambda x: (function(x) - p[1]) / p[1])
    return df[column_name]


def gaus_to_int(df, column_name, interval):

    val = df[column_name].values
    if interval != None:
        mask_condition = np.logical_and(val >= interval[0], val <= interval[1])
        loc = np.mean(val[mask_condition])
        val[mask_condition] = np.ones_like(val[mask_condition]) * loc
    else:
        df[column_name] = np.rint(df[column_name].values)
    return df[column_name]


def process_column_var(column_name, operations, df):

    for op in operations:

        if op[0] == "g":
            mask_condition = op[1]
            df[column_name] = gaus_to_int(df, column_name, mask_condition)

        if op[0] == "t":
            function = op[1]
            p = op[2]
            df[column_name] = transform(df, column_name, function, p)

        else:
            return df[column_name]
    return df[column_name]


def preprocessing(df, vars_dictionary):
    """
    Preprocessing general function given any dataframe and its dictionary
    """

    for column_name, operation in vars_dictionary.items():
        df[column_name] = process_column_var(column_name, operation, df)

    return df


gen_columnns = [
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
    "MElectron_tightCharge",
]
