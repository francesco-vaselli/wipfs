import h5py
import pandas as pd
import numpy as np


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

    print(df)

    df = df.sort_values(by=df.columns[32])

    print(df.iloc[:, 32])

    save_file = h5py.File(f"../../training/datasets/train_dataset_fake_jets.hdf5", "w")

    dset = save_file.create_dataset("data", data=df.values, dtype="f4")

    save_file.close()

    # df.to_hdf('../../training/datasets/train_dataset_fake_jets.hdf5', key='data', mode='w')