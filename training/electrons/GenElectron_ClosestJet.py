import numpy as np
from matplotlib import pyplot as plt
import h5py
import pandas as pd

from postprocessing import gen_columns, reco_columns

f = h5py.File("MElectrons.hdf5", "r")
df = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

mask = (
    df["ClosestJet_EncodedPartonFlavour_g"].values
    + df["ClosestJet_EncodedPartonFlavour_light"].values
).astype(bool)

ele_pt = df["GenElectron_pt"].values
jet_pt = df["ClosestJet_pt"].values

ele_pt = ele_pt[mask]
jet_pt = jet_pt[mask]

plt.hist(np.abs(ele_pt - jet_pt), histtype="step", bins=100, range=[0, 10])
plt.show()
