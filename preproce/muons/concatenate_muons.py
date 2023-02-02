import h5py
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.join("../..", "utils"))

from fake_utils import mod_sum_pt, sum_px_py


if __name__=='__main__':

    h5_files = [
            "../../../muonData/amuons1.hdf5",
            "../../../muonData/amuons2.hdf5",
            "../../../muonData/amuons3.hdf5",
            "../../../muonData/amuons4.hdf5",
            "../../../muonData/amuons5.hdf5",
            "../../../muonData/amuons6.hdf5",
            "../../../muonData/amuons8.hdf5",
            "../../../muonData/amuons9.hdf5",
        ]

    data = np.array(h5py.File(h5_files[0], "r")["data"][:,:])
    df = pd.DataFrame(data=data)
    for h5_file in h5_files[1:]:
        data = np.array(h5py.File(h5_file, "r")["data"][:,:])
        df = pd.concat([df, pd.DataFrame(data=data)], axis=0)
        df = df.reset_index(drop=True)


    save_file = h5py.File(f"../../training/datasets/train_dataset_muons.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()

    # df.to_hdf('../../training/datasets/train_dataset_fake_jets.hdf5', key='data', mode='w')