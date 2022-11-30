# preprocess all the muons variables for use in training
import uproot
import pandas as pd
import h5py
import numpy as np
import sys
import torch  # we use pytorch
import torch_geometric
from torch_geometric.data import HeteroData  # and the "geometric" package to help handling GNNs
from torch_cluster import (
    knn_graph,
)  # utility to search for  nearest neighbor on a graph
import networkx as nx  # tool to draw graphs
import matplotlib.pyplot as plt


STOP = 10
KNN = 3

def graph_builder(gen_jets, gen_parts, global_f, fake_target, k):

    data = HeteroData()
    # Create two node types "gen part" and "gen jet" holding a feature matrix:
    data['gen_jet'].x = torch.tensor(gen_jets)
    data['gen_part'].x = torch.tensor(gen_parts)
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
    print(dfgj)

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
    print(dfft)

    for i in range(len(dfgj)):

        graph = graph_builder(
            dfgj.loc[[i]].reset_index(drop=True).values,
            dfgp.loc[[i]].reset_index(drop=True).values,
            dfgl.loc[[i]].reset_index(drop=True).values,
            dfft.loc[[i]].reset_index(drop=True).values,
            k=KNN,
        )

        g = torch_geometric.utils.to_networkx(graph.to_homogeneous())
        nx.draw(g)
        plt.savefig('graph.png')
