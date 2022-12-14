# preprocess all the muons variables for use in training
import uproot
import pandas as pd
import h5py
import numpy as np
import sys
import torch  # we use pytorch
import torch_geometric
from torch_geometric.data import (
    Data,
)  # and the "geometric" package to help handling GNNs
from torch_cluster import (
    knn_graph,
)  # utility to search for  nearest neighbor on a graph
import networkx as nx  # tool to draw graphs
import matplotlib.pyplot as plt


STOP = 10
KNN = 3


def graph_builder(gen_jets, gen_parts, global_f, fake_target, k):

    data = Data(globf=torch.tensor(global_f))
    data.x = torch.vstack((torch.tensor(gen_jets), torch.tensor(gen_parts)))
    data.y = torch.tensor(fake_target)

    data.edge_index = knn_graph(data.x[:, [1, 2]], k=k)

    return data


if __name__ == "__main__":

    # use uproot to read .root file directly in python
    # f = sys.argv[1]
    tree = uproot.open(f"FJets.root:FJets", num_workers=20)
    vars_to_save = tree.keys()
    print(vars_to_save)

    # define pandas df for fast manipulation
    dfgj = tree.arrays(
        [
            "GenJet_mass",
            "GenJet_eta",
            "GenJet_phi",
            "GenJet_pt",
            "GenJet_partonFlavour",
            "GenJet_hadronFlavour",
        ],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    dfgj["GenJet_pdgId"] = 100 * np.abs(dfgj.GenJet_hadronFlavour.values) + np.abs(
        dfgj.GenJet_partonFlavour.values
    )
    dfgj["GenJet_status"] = np.ones(len(dfgj))
    dfgj["GenJet_statusFlags"] = np.zeros(len(dfgj))
    dfgj = dfgj.drop(
        columns=[
            "GenJet_partonFlavour",
            "GenJet_hadronFlavour",
        ]
    )
    print(dfgj)

    dfgj1 = dfgj.iloc[0]
    # a pandas df equal to dfgj but without stats
    dfgj2 = dfgj.drop(columns=["GenJet_status", "GenJet_statusFlags"])

    # define pandas df for fast manipulation
    dfgp = tree.arrays(
        [
            "GenPart_mass",
            "GenPart_eta",
            # "GenPart_genPartIdxMother",
            "GenPart_phi",
            "GenPart_pdgId",
            "GenPart_pt",
            "GenPart_status",
            "GenPart_statusFlags",
        ],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    dfgp = dfgp[np.abs(dfgp.GenPart_eta) <= 5]
    print(dfgp)

    # define pandas df for fast manipulation
    dfgl = tree.arrays(
        [
            "Pileup_gpudensity",
            "Pileup_nPU",
            "Pileup_nTrueInt",
            "Pileup_pudensity",
            "Pileup_sumEOOT",
            "Pileup_sumLOOT",
        ],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    print(dfgl)

    # define pandas df for fast manipulation
    dfft = tree.arrays(
        ["FJet_phi", "FJet_eta", "FJet_pt"],
        library="pd",
        entry_stop=STOP,
    ).astype("float32")
    dfft = dfft.reindex(
        pd.MultiIndex.from_product([np.arange(len(dfgl)), np.arange(10)]), fill_value=0
    )
    print(dfft)

    for i in range(len(dfgj)):

        graph = graph_builder(
            dfgj.loc[[i]].reset_index(drop=True).values,
            dfgp.loc[[i]].reset_index(drop=True).values,
            dfgl.loc[[i]].reset_index(drop=True).values,
            dfft.loc[[i]].reset_index(drop=True).values,
            k=KNN,
        )

        g = torch_geometric.utils.to_networkx(graph)
        x = graph.x
        # for j in (enumerate(x)):
        #   print(j)
        fig, ax = plt.subplots()
        nx.draw(g, {i: [p[1], p[2]] for i, p in enumerate(x)}, node_size=(25), ax=ax)
        limits = plt.axis("on")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.scatter(
            graph.y[:, 1][
                ((np.array(graph.y[:, 1]) != 0) | (np.array(graph.y[:, 0]) != 0))
            ],
            graph.y[:, 0][
                ((np.array(graph.y[:, 1]) != 0) | (np.array(graph.y[:, 0]) != 0))
            ],
            color="red",
        )
        plt.savefig("graph.png")


