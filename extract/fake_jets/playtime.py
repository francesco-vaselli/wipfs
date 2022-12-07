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


# a pytorch neural network to predict the fake jet pt, eta, phi
# taking as input the graph of the gen objects
# and the global features, passing them through a pytorch geometric GNN
# and then through a fully connected network
# and finally to an lstm
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(5, 64)
        self.conv2 = torch_geometric.nn.GCNConv(64, 64)
        # dense layers
        self.fc1 = torch.nn.Linear(64 + 3, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        # lstm
        self.lstm = torch.nn.LSTM(64, 64, 3)
        # lstm output of shape (batch, 10, 3)
        self.lstm = torch.nn.Linear(64, 30)

    def forward(self, data):
        x, edge_index, globf = data.x, data.edge_index, data.globf
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.cat((x, globf), 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.lstm(x)
        return x


# a pytorch geometric dataset to load the data
# and to split it into train, validation and test
# applying the graph_builder function to each event
class FakeJetsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, files, transform=None, pre_transform=None):
        super(FakeJetsDataset, self).__init__(root, files, transform, pre_transform)
        self.data, self.slices = self.process()
        self.raw_file_names = files

    @property
    def raw_file_names(self):
        return ["FJets.root"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        # use uproot to read .root file directly in python
        # f = sys.argv[1]
        tree = uproot.open(f"{self.raw_file_names}:FJets", num_workers=20)
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

        # define pandas df for fast manipulation
        dfft = tree.arrays(
            ["FJet_phi", "FJet_eta", "FJet_pt"],
            library="pd",
            entry_stop=STOP,
        ).astype("float32")
        dfft = dfft.reindex(
            pd.MultiIndex.from_product([np.arange(len(dfgl)), np.arange(10)]), fill_value=0
        )

        data_list = []
        for i in range(len(dfgj)):
            data =  graph_builder(
                dfgj.loc[[i]].reset_index(drop=True).values,
                dfgp.loc[[i]].reset_index(drop=True).values,
                dfgl.loc[[i]].reset_index(drop=True).values,
                dfft.loc[[i]].reset_index(drop=True).values,
                k=KNN,
            )
            data_list.append(data)
        
        return self.collate(data_list)


    def len(self):
        return len(self.data)


# traing function
def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

# validation function
def test(loader):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        loss += F.mse_loss(pred, data.y, reduction="sum").item()
    return loss / len(loader.dataset)





if __name__ == "__main__":

    # load the data
    dataset = FakeJetsDataset(root=".")

    # define model and optimizer
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # split the data into train, validation and test
    train_dataset = dataset[: int(0.8 * len(dataset))]
    val_dataset = dataset[int(0.8 * len(dataset)) : int(0.9 * len(dataset))]
    test_dataset = dataset[int(0.9 * len(dataset)) :]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # train the model
    for epoch in range(1, 201):
        loss = train(epoch)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    

    
    
