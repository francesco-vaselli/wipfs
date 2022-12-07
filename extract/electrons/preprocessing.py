import sys
import json
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import uproot

import warnings
warnings.filterwarnings("ignore") # temporary for MatPlotLibDeprecationWarning bug

from dictionary import vars_dictionary  # operation dictionary

np.random.seed(0)  # fixed seed for gaussian random smearing


def make_dataset(tree, version, dictionary=False, *args, **kwargs):
    """
    Given the TTree, returns the corresponding pandas dataframe.
    If dictionary is True, an empty dictionary of TTree variables is dumped on .txt file (to be copied on dictionary.py).
    """
    df = (
        tree.arrays(library="pd", *args, **kwargs)
        .reset_index(drop=True)
        .astype("float32")
        .dropna()
    )

    if dictionary:
        val = input("Are you sure to make a new empty vars_dictionary? (y/n)\n")
        if val == "y":
            print("Rewriting...")
            vars_to_save = tree.keys()
            d = {name: [] for name in vars_to_save}

            with open(f"vars_dictionary_v{version}.txt", "w") as file:
                for key, value in d.items():
                    file.write(f'"{key}": {value},\n')

            file.close()
        else:
            print("Aborting...")
            return df
        print("Done.")
    return df


def saturation(df, column_name, interval):
    """
    Performs saturation on given column.
    """
    print(f"Saturating in range {interval}...")
    val = df[column_name].values
    df[column_name] = np.where(val < interval[0], interval[0], val)
    val = df[column_name].values
    df[column_name] = np.where(val > interval[1], interval[1], val)
    return df[column_name]


def gaus_smearing(df, column_name, sigma, interval):
    """
    Performs gaussian smearing on given column. If interval is specified, random gaussian data are asseigned to column in interval.
    """
    val = df[column_name].values
    if interval != None:
        mask_condition = np.logical_and(val >= interval[0], val <= interval[1])
        loc = np.mean(val[mask_condition])
        print(
            f"Creating gaussian data (loc={loc}, scale={sigma}) in range {interval}..."
        )
        val[mask_condition] = np.random.normal(
            loc=loc, scale=sigma, size=val[mask_condition].shape
        )
    else:
        print(f"Smearing with sigma={sigma}...")
        df[column_name] = df[column_name].apply(
            lambda x: x + sigma * np.random.normal()
        )
    return df[column_name]


def transform(df, column_name, function, p):
    """
    Performs a function tranformation on column
    """
    print(f"Applying {function} with parameters {p}...")
    df[column_name] = df[column_name].apply(lambda x: function(x * p[0] + p[1]))
    return df[column_name]


def process_column_var(column_name, operations, df):
    """
    Processes single dataframe column. Operation type is specified by string.
    """
    print(f"Processing {column_name}...")
    for op in operations:
        if op[0] == "s":
            interval = op[1]
            df[column_name] = saturation(df, column_name, interval)

        if op[0] == "g":
            sigma = op[1]
            mask_condition = op[2]
            df[column_name] = gaus_smearing(df, column_name, sigma, mask_condition)

        if op[0] == "t":
            function = op[1]
            p = op[2]
            df[column_name] = transform(df, column_name, function, p)

        else:
            return df[column_name]
    print("Done.")
    return df[column_name]


def preprocessing(df, vars_dictionary):
    """
    Preprocessing general function given any dataframe and its dictionary
    """
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    for column_name, operation in vars_dictionary.items():
       fig, axs = plt.subplots(1, 2)
       plt.suptitle(f"{column_name}")
       axs[0].hist(df[column_name], bins=30, histtype="step")
       df[column_name] = process_column_var(column_name, operation, df)
       axs[1].hist(df[column_name], bins=30, histtype="step")
       plt.savefig(f"figures/{column_name}.pdf", format="pdf")
       plt.close()  # produces MatplotlibDeprecationWarning. It is a bug (https://github.com/matplotlib/matplotlib/issues/23921)

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis="columns")]

    return df


if __name__ == "__main__":

    f = sys.argv[1]
    tree = uproot.open(f"MElectrons_v{f}.root:MElectrons", num_workers=20)

    df = make_dataset(tree, version=f, dictionary=False)

    df = preprocessing(df, vars_dictionary)

    file = h5py.File(f"electrons_v{f}.hdf5", "w")

    dset = file.create_dataset("data", data=df.values, dtype="f4")

    file.close()
