import pandas as pd
import numpy as np
import uproot
import awkward as ak
import h5py
import sys

STOP = None


if __name__ == '__main__':

    root_files = [
            "~/wipfs/extract/fake_jets/extracted_files/FJets1.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets2.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets3.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets4.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets5.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets6.root:FJets",
        ]
    
    tree = uproot.open(root_files[0], num_workers=20)
    # define pandas df for fast manipulation
    dfgl = tree.arrays(
        [
            "Pileup_gpudensity",
            "Pileup_nPU",
            "Pileup_nTrueInt",
            "Pileup_pudensity",
            "Pileup_sumEOOT",
            "Pileup_sumLOOT",
        ],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    print(dfgl)

    # define pandas df for fast manipulation
    dfft = tree.arrays(
        ["FJet_pt", "FJet_eta", "FJet_phi"],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")

    for root_file in root_files[1:]:
        tree = uproot.open(root_file, num_workers=20)
        df1 = tree.arrays(
            [
                "Pileup_gpudensity",
                "Pileup_nPU",
                "Pileup_nTrueInt",
                "Pileup_pudensity",
                "Pileup_sumEOOT",
                "Pileup_sumLOOT",
            ],
            library="pd",
            entry_stop=STOP,
        ).astype("float32")

        df2 = tree.arrays(
            ["FJet_pt", "FJet_eta", "FJet_phi"],
            library="pd",
            entry_stop=STOP,
        ).astype("float32")

        dfgl = pd.concat([dfgl, df1], axis=0)
        dfft = pd.concat([dfft, df2], axis=0)
        dfgl = dfgl.reset_index(drop=True)
        # dfft = dfft.reset_index(drop=True)

    print(dfgl)
    print(dfft)
    out_idx = dfft.index.get_level_values(0)
    inn_idx = dfft.index.get_level_values(1)
    dfft = dfft.reset_index(drop=True).reindex(pd.MultiIndex.from_arrays([np.arange(len(out_idx)), inn_idx]))
    print(dfft)
    num_fakes = dfft.reset_index(level=1).index.value_counts(sort=False).reindex(np.arange(len(dfgl)), fill_value=0).values
    # fill missing fakes with 0s. seems to be cutting excess fakes per event
    dfft = dfft.reindex(pd.MultiIndex.from_product([np.arange(len(dfgl)), np.arange(10)]), fill_value=0) 

    # get all fake in one event on the same row
    # NOTE: we now have all pts, then all etas, then all phis
    dfft = dfft.unstack(level=-1).T.reset_index(drop=True).T
    print(dfft)

    df = pd.concat([dfft, dfgl, pd.DataFrame(num_fakes, columns=['num_fakes'])], axis=1)
    df = df[(df.T != 0).any()]
    # print(df)

    # TODO: add Ht, phi calculation?

    df["num_fakes"] = df["num_fakes"].apply(
        lambda x: x + np.random.uniform(low=-0.5, high=0.5) # if x > 0 else 0 WE SHOULDN'T HAVE ANY 0s
    )
    print(df)

    # save_file = h5py.File(f"../../training/datasets/fake_jets{file_num}.hdf5", "w")

    # dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    # save_file.close()
