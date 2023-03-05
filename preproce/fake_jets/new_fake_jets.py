import pandas as pd
import numpy as np
import uproot
import awkward as ak
import h5py


def mod_sum_pt(pts):
    """module sum of pts

    Args:
        pts (np.array): pts array shape [n events, 10 fake jets], 0 for empty jets
    Returns:
        spt: sum of pts [n events]
    """
    spt = np.sum(pts, axis=1)
    spt.flatten()

    return spt


def vec_sum_pt(pts, phis):
    """vector sum of pts

    Args:
        pts (np.array): pts array shape [n events, 10 fake jets], 0 for empty jets
        phis (np.array): phis array shape [n events, 10 fake jets], 0 for empty jets

    Returns:
        spt: vector sum of pts [n events]
        angle: angle of the vector sum of pts [n events]
    """
    px = np.sum(pts * np.cos(phis), axis=1)
    py = np.sum(pts * np.sin(phis), axis=1)

    spt = np.sqrt(px**2 + py**2)
    spt.flatten()

    angle = np.arctan2(py, px)
    angle.flatten()

    return spt, angle

STOP = None

def single_file_preprocess(filename : str) -> pd.DataFrame:
    tree = uproot.open(filename, num_workers=20)
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

    # get all fake in one event on the same row
    # NOTE: we now have all pts, then all etas, then all phis
    dfft = dfft.unstack(level=-1).T.reset_index(drop=True).T
    print(dfft)

    df = pd.concat([dfft, dfgl, pd.DataFrame(num_fakes, columns=['num_fakes'])], axis=1)
    df = df[(df.iloc[:, :10].T != 0).any()]
    df = df[df["num_fakes"]<=10]
    df = df.reset_index(drop=True)
    print(df)

    # Ht, phi calculation
    pts = df.iloc[:, :10].values
    phis = df.iloc[:, 20:30].values

    Ht = mod_sum_pt(pts)
    pt, angle = vec_sum_pt(pts, phis)

    # saturate mod pT
    Ht = np.where(Ht > 200, 200, Ht)
    # saturate pT
    pt = np.where(pt > 200, 200, pt)

    df["Ht"] = Ht
    df["pt"] = pt
    df["angle"] = angle

    # add NMasks
    print(df["num_fakes"].values.shape)
    # create a mask of shape [len(df), 10*3] which is 1 for the first num_fakes*3 and 0 for the rest
    NMasks = np.zeros((len(df), 10*3))
    for i in range(1, 11):
        NMasks[df["num_fakes"].values == i, :(i)*3] = 1
    NMasks = NMasks.astype("float32")
    #print(mask, df["num_fakes"].values[100],mask[100], df["num_fakes"].values[107], mask[107] )
    dfnm = pd.DataFrame(NMasks)
    print(dfnm)
    df = pd.concat([df, dfnm], axis=1)
    print(df)

    df["num_fakes"] = df["num_fakes"].apply(
        lambda x: x + np.random.uniform(low=-0.5, high=0.5) # if x > 0 else 0 WE SHOULDN'T HAVE ANY 0s
    )
    print(df)

    return df

if __name__ == '__main__':

    root_files = [
            "~/wipfs/extract/fake_jets/extracted_files/FJets1.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets2.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets3.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets4.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets5.root:FJets",
            "~/wipfs/extract/fake_jets/extracted_files/FJets6.root:FJets",
        ]
    df = single_file_preprocess(root_files[0])
    for root_file in root_files[1:]:
        df1 = single_file_preprocess(root_file)
        df = pd.concat([df, df1], axis=0)
        df = df.reset_index(drop=True)

    print(df)

    save_file = h5py.File(f"../../training/datasets/full_fake_with_mask.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()