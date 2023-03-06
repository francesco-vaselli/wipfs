import numpy as np

"""
Dictionary of postprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

unsmearing: ["d", [inf, sup]]
transformation: ["i", func, [a, b]]  # func(x - b) / a

In the case of multiple operations, order follows the operation list indexing.
"""

context_dictionary = {
    "MGenElectron_eta": [],
    "MGenElectron_phi": [],
    "MGenElectron_pt": [],
    "MGenElectron_charge": [],
   # "MGenPartMother_pdgId": [],
   # "MGenPartMother_pt": [],
   # "MGenPartMother_deta": [],
   # "MGenPartMother_dphi": [],
    "MGenElectron_statusFlag0": [],
    "MGenElectron_statusFlag1": [],
    "MGenElectron_statusFlag2": [],
    "MGenElectron_statusFlag3": [],
    "MGenElectron_statusFlag4": [],
    "MGenElectron_statusFlag5": [],
    "MGenElectron_statusFlag6": [],
    "MGenElectron_statusFlag7": [],
    "MGenElectron_statusFlag8": [],
    "MGenElectron_statusFlag9": [],
    "MGenElectron_statusFlag10": [],
    "MGenElectron_statusFlag11": [],
    "MGenElectron_statusFlag12": [],
    "MGenElectron_statusFlag13": [],
    "MGenElectron_statusFlag14": [],
    "ClosestJet_dr": [["i", np.expm1, [1, 0]]],
    "ClosestJet_dphi": [["i", np.tan, [100, 0]]],
    "ClosestJet_deta": [["i", np.tan, [100, 0]]],
    "ClosestJet_pt": [["i", np.expm1, [1, 0]]],
    "ClosestJet_mass": [["i", np.expm1, [1, 0]]],
    "ClosestJet_EncodedPartonFlavour_light": [],
    "ClosestJet_EncodedPartonFlavour_gluon": [],
    "ClosestJet_EncodedPartonFlavour_c": [],
    "ClosestJet_EncodedPartonFlavour_b": [],
    "ClosestJet_EncodedPartonFlavour_undefined": [],
    "ClosestJet_EncodedHadronFlavour_b": [],
    "ClosestJet_EncodedHadronFlavour_c": [],
    "ClosestJet_EncodedHadronFlavour_light": [],
    "Pileup_gpudensity": [],
    "Pileup_nPU": [],
    "Pileup_nTrueInt": [],
    "Pileup_pudensity": [],
    "Pileup_sumEOOT": [],
    "Pileup_sumLOOT": [],
}

target_dictionary = {
    "MElectron_charge": [["c", 0, [-1, 1]]],
    "MElectron_convVeto": [["d", None]],
    #    "MElectron_cutBased": [["d", 0.1, None]],
    #    "MElectron_cutBased_Fall17_V1": [["d", 0.1, None]],
    #    "MElectron_cutBased_HEEP": [["d", 0.1, None]],
    "MElectron_deltaEtaSC": [],
    "MElectron_dr03EcalRecHitSumEt": [["d", [-np.inf, -2]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_dr03HcalDepth1TowerSumEt": [
        ["d", [-np.inf, -2]],
        ["i", np.exp, [1, 1e-3]],
    ],
    "MElectron_dr03TkSumPt": [["d", [-np.inf, -2]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_dr03TkSumPtHEEP": [["d", [-np.inf, -2]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_dxy": [["i", np.tan, [150, 0]]],
    "MElectron_dxyErr": [["i", np.exp, [1, 1e-3]]],
    "MElectron_dz": [["i", np.tan, [50, 0]]],
    "MElectron_dzErr": [["i", np.exp, [1, 1e-3]]],
    # "MElectron_eCorr": [["i", np.arctan, [1e-1, -1e-1]]],
    "MElectron_eInvMinusPInv": [["i", np.tan, [150, 0]]],
    "MElectron_energyErr": [["i", np.expm1, [1, 0]]],
    "MElectron_etaMinusGen": [["i", np.tan, [100, 0]]],
    "MElectron_hoe": [["d", [-np.inf, -6]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_ip3d": [["i", np.exp, [1, 1e-3]]],
    "MElectron_isPFcand": [["d", None]],
    "MElectron_jetPtRelv2": [["i", np.expm1, [1, 0]]],
    "MElectron_jetRelIso": [["i", np.exp, [10, 1e-2]]],
    "MElectron_lostHits": [["d", None]],
    "MElectron_miniPFRelIso_all": [["d", [-np.inf, -5.5]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_miniPFRelIso_chg": [["d", [-np.inf, -5.5]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_mvaFall17V1Iso": [],
    "MElectron_mvaFall17V1Iso_WP80": [["d", None]],
    "MElectron_mvaFall17V1Iso_WP90": [["d", None]],
    "MElectron_mvaFall17V1Iso_WPL": [["d", None]],
    "MElectron_mvaFall17V1noIso": [],
    "MElectron_mvaFall17V1noIso_WP80": [["d", None]],
    "MElectron_mvaFall17V1noIso_WP90": [["d", None]],
    "MElectron_mvaFall17V1noIso_WPL": [["d", None]],
    "MElectron_mvaFall17V2Iso": [],
    "MElectron_mvaFall17V2Iso_WP80": [["d", None]],
    "MElectron_mvaFall17V2Iso_WP90": [["d", None]],
    "MElectron_mvaFall17V2Iso_WPL": [["d", None]],
    "MElectron_mvaFall17V2noIso": [],
    "MElectron_mvaFall17V2noIso_WP80": [["d", None]],
    "MElectron_mvaFall17V2noIso_WP90": [["d", None]],
    "MElectron_mvaFall17V2noIso_WPL": [["d", None]],
    "MElectron_mvaTTH": [],
    "MElectron_pfRelIso03_all": [
        ["d", [-np.inf, -5.5]],
        ["i", np.exp, [1, 1e-3]],
    ],
    "MElectron_pfRelIso03_chg": [["d", [-np.inf, -5.5]], ["i", np.exp, [1, 1e-3]]],
    "MElectron_phiMinusGen": [["i", np.tan, [80, 0]]],
    "MElectron_ptRatio": [["i", np.tan, [10, -10]]],
    "MElectron_r9": [["i", np.tan, [10, -0.15]], ["i", np.exp, [1, 1e-2]]],
    "MElectron_seedGain": [["d", None]],
    "MElectron_sieie": [["i", np.tan, [1, -1.25]], ["i", np.exp, [10, 1e-1]]],
    "MElectron_sip3d": [["i", np.expm1, [1, 0]]],
    "MElectron_tightCharge": [["d", None]],
    #    "MElectron_vidNestedWPBitmap0": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap1": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap2": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap3": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap4": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap5": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap6": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap7": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap8": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmap9": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP0": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP1": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP2": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP3": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP4": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP5": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP6": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP7": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP8": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP9": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP10": [["d", 0.1, None]],
    #    "MElectron_vidNestedWPBitmapHEEP11": [["d", 0.1, None]],
}
