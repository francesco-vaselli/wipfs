# preprocess all the muons variables for use in training

import uproot
import pandas as pd
import h5py
import numpy as np
import sys


if __name__ == "__main__":

    # use uproot to read .root file directly in python
    # f = sys.argv[1]
    tree = uproot.open(f"FJets.root:FJets", num_workers=20)
    vars_to_save = tree.keys()
    print(vars_to_save)

    # define pandas df for fast manipulation
    df = tree.arrays(library="pd", entry_stop=10).reset_index(drop=True).astype("float32").dropna()
    print(df)

    # numerical errors check
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]