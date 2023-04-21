import numpy as np
from matplotlib import pyplot as plt
import h5py
import pandas as pd

from postprocessing import gen_columns, reco_columns

f = h5py.File("MElectrons.hdf5", "r")
df = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

#mask = (
#    df["ClosestJet_EncodedPartonFlavour_gluon"].values
#    + df["ClosestJet_EncodedPartonFlavour_light"].values
#).astype(bool)

mask = df["ClosestJet_EncodedPartonFlavour_b"].values.astype(bool)

total = df["ClosestJet_EncodedPartonFlavour_b"].values + df["ClosestJet_EncodedPartonFlavour_c"].values+df["ClosestJet_EncodedPartonFlavour_gluon"].values + df["ClosestJet_EncodedPartonFlavour_light"].values + df["ClosestJet_EncodedPartonFlavour_undefined"].values

print(total.sum())

plt.hist(mask.astype(int), histtype="step", bins=2)
plt.savefig("mask.pdf", format="pdf")
plt.close()

ele_pt = df["MGenElectron_pt"].values
jet_pt = df["ClosestJet_pt"].values

ele_pt = ele_pt[mask]
jet_pt = jet_pt[mask]

plt.hist(np.abs(ele_pt - jet_pt), histtype="step", bins=100, range=[0, 150])
plt.savefig("ele_jet.pdf", format="pdf")
plt.close()

dr = df["ClosestJet_dr"].values
dr = dr[mask]

plt.hist(dr, histtype="step", bins=100, range=[0, 0.2])
plt.savefig("dr.pdf", format="pdf")
plt.close()
