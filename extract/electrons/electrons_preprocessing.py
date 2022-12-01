import sys
import json
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import uproot

np.random.seed(0)  # fixed seed for gaussian random smearing


def make_dataset(tree, version, dictionary=False, *args, **kwargs):
    """
    Print dictionary of TTree variables and their preprocessing operations and return corresponding pandas dataframe. Sets default operations to False
    """
    df = (
        tree.arrays(library="pd", *args, **kwargs)
        .reset_index(drop=True)
        .astype("float32")
        .dropna()
    )

    if dictionary:
        vars_to_save = tree.keys()
        d = {name: [False, False, False] for name in vars_to_save}

        with open(f"vars_dictionary_v{version}.txt", "w") as file:
            file.write(json.dumps(d, indent=""))

        file.close()

    return df


def read_dictionary(version):
    """
    Read operation dictionary from file
    """
    with open(f"vars_dictionary_v{version}.txt") as file:
        data = file.read()
    d = json.loads(data)
    file.close()
    return d


def process_column(column_name, operation, df):
    """
    Process the single dataframe column as specified in the operation list
    """
    print(f"{column_name}:")
    if operation[0]:
        range = operation[0]
        print(f"Saturating in range {range}...")
        val = df[column_name].values
        df[column_name] = np.where(val < range[0], range[0], val)
        val = df[column_name].values
        df[column_name] = np.where(val > range[1], range[1], val)
        print("Done")

    if operation[1]:
        sigma = operation[1]
        print(f"Gaussian smearing with sigma {sigma}")
        df[column_name] = df[column_name].apply(
            lambda x: x + sigma * np.random.normal()
        )
        print("Done")

    if operation[2]:
        func = operation[2][0]
        p = operation[2][1]
        print(f"Applying {func}(x * {p[0]} + {p[1]})")
        df[column_name] = df[column_name].apply(lambda x: func(x * p[0] + p[1]))
        print("Done")

    return df[column_name]


def preprocessing(df, vars_dictionary):
    """
    Preprocessing general function given any dataframe and its dictionary
    """
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    for column_name, operation in vars_dictionary.items():
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"{column_name}")
        axs[0].hist(df[column_name], bins=50, histtype="step")
        df[column_name] = process_column(column_name, operation, df)
        axs[1].hist(df[column_name], bins=50, histtype="step")
        plt.savefig(f"preprocessing_fig/{column_name}.pdf", format="pdf")
        plt.close()  # produces MatplotlibDeprecationWarning. It is a bug (https://github.com/matplotlib/matplotlib/issues/23921)

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    return df


if __name__ == "__main__":

    f = sys.argv[1]
    tree = uproot.open(f"MElectrons_v{f}.root:MElectrons")

    df = make_dataset(tree, version=f, dictionary=True, entry_stop=10000)

    vars_dictionary = read_dictionary(version=f)

    df = preprocessing(df, vars_dictionary)

    file = h5py.File(f"aelectrons_v{f}.hdf5", "w")

    dset = file.create_dataset("data", data=df.values, dtype="f4")

    file.close()
