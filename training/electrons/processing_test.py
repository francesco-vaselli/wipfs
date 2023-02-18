import json
import pandas as pd
import numpy as np
import h5py

from scipy.stats import wasserstein_distance, ks_2samp

from postprocessing import gen_columns, reco_columns

f = h5py.File("MElectrons_pre.hdf5", "r")
df_pre = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

f = h5py.File("MElectrons_post.hdf5", "r")
df_post = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

ws_dict = {}
ks_dict = {}

for column_name in reco_columns:
    ws_dict[column_name] = wasserstein_distance(
        df_pre[column_name].values, df_post[column_name].values
    )
    ks_pvalue = ks_2samp(df_pre[column_name].values, df_post[column_name].values)[1]
    ks_dict[column_name] = ks_pvalue

f = open("ws.json", "w")
f.write(json.dumps(ws_dict))
f.close()

f = open("ks.json", "w")
f.write(json.dumps(ks_dict))
f.close()

