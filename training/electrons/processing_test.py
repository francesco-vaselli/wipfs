import json
import pandas as pd
import numpy as np
import h5py

from scipy.stats import wasserstein_distance

from postprocessing import gen_columns, reco_columns

f = h5py.File("MElectrons_pre.hdf5", "r")
df_pre = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

f = h5py.File("MElectrons_post.hdf5", "r")
df_post = pd.DataFrame(data=f.get("data"), columns=gen_columns + reco_columns)
f.close()

ws_dict = {}

for column_name in reco_columns:
    ws_dict[column_name] = wasserstein_distance(
        df_pre[column_name].values, df_post[column_name].values
    )

f = open("ws.json", "w")
f.write(json.dumps(ws_dict))
f.close()