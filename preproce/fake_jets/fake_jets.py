import pandas as pd
import numpy as np
import uproot
import awkward as ak
import h5py
import sys

STOP = None


if __name__ == '__main__':

    file_num = sys.argv[1]
    
    tree = uproot.open(f"~/wipfs/extract/fake_jets/extracted_files/FJets{file_num}.root:FJets", num_workers=20)
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

    num_fakes = dfft.reset_index(level=1).index.value_counts(sort=False).reindex(np.arange(len(dfgl)), fill_value=0).values
    # fill missing fakes with 0s. seems to be cutting excess fakes per event
    dfft = dfft.reindex(pd.MultiIndex.from_product([np.arange(len(dfgl)), np.arange(10)]), fill_value=0) 

    # fill missing fakes with nonphysical values
    dfft['FJet_pt'] = [i if i != 0 else np.random.normal(-10, 1) for i in dfft['FJet_pt'].values]
    dfft['FJet_eta'] = [i if i != 0 else np.random.normal(-10, 1) for i in dfft['FJet_eta'].values]
    dfft['FJet_phi'] = [i if i != 0 else np.random.normal(-10, 1) for i in dfft['FJet_phi'].values]

    # get all fake in one event on the same row
    # NOTE: we now have all pts, then all etas, then all phis
    dfft = dfft.unstack(level=-1).T.reset_index(drop=True).T
    print(dfft)

    df = pd.concat([dfft, dfgl, pd.DataFrame(num_fakes, columns=['num_fakes'])], axis=1)
    df["num_fakes"] = df["num_fakes"].apply(
        lambda x: x + 0.1 * np.random.normal()
    )
    print(df)

    save_file = h5py.File(f"../../training/datasets/fake_jets{file_num}.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()
