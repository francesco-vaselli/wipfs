import h5py
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.join("../..", "utils"))

from fake_utils import mod_sum_pt, sum_px_py


if __name__=='__main__':

    h5_files = [
            "../../training/datasets/fake_jets1.hdf5",
            "../../training/datasets/fake_jets2.hdf5",
            "../../training/datasets/fake_jets3.hdf5",
            "../../training/datasets/fake_jets4.hdf5",
            "../../training/datasets/fake_jets5.hdf5"
        ]

    data = np.array(h5py.File(h5_files[0], "r")["data"][:,:])
    df = pd.DataFrame(data=data)
    for h5_file in h5_files[1:]:
        data = np.array(h5py.File(h5_file, "r")["data"][:,:])
        df = pd.concat([df, pd.DataFrame(data=data)], axis=0)


    df = df.sort_values(by=df.columns[32])

    print(df.iloc[:, 32])
    print(df)
    # revert N fakes to int and scale
    df.iloc[:, 36] = np.rint(df.iloc[:, 36].values)/10

    # print(df.iloc[:, 36])

    # get sum of modpt, px and py
    pts = df.iloc[:, :10].values
    # correct for negative values
    pts = np.where(pts < 0, 0, pts)
    # etas = np.reshape(dfft1['FJet_eta'], (-1, 10))
    phis = df.iloc[:, 20:30].values
    # this is not needed as the 0 module avoids taking into account the unphysical phis
    # phis = np.where(phis < -2*np.pi, 0, phis)

    mod_pt = mod_sum_pt(pts)
    px, py = sum_px_py(pts, phis)

    df["mod_sum_pt"] = mod_pt
    df["sum_px"] = px
    df["sum_py"] = py

    print(df.iloc[:, [36, 37, 38, 39]])

    save_file = h5py.File(f"../../training/datasets/train_dataset_fake_jets_only_flows.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()

    # df.to_hdf('../../training/datasets/train_dataset_fake_jets.hdf5', key='data', mode='w')