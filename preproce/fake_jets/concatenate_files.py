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

    df = pd.read_hdf(h5_files[0], key='data')
    for h5_file in h5_files[1:]:
        df = pd.concat([df, pd.read(h5_file, key='data')], ignore_index=True)

    print(df)

    df = df.sort_values(by=['32'])

    print(df.iloc[:, 32])

    df.to_hdf('../../training/datasets/train_dataset_fake_jets.hdf5', key='data', mode='w')