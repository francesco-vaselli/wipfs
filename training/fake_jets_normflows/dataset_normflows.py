import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SimpleFakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, y_dim, z_dim, args, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, 30:30 +6]
        z = self.archives[0]["data"][start:limit, (6 + 30) : (6 + 1 + 30 + z_dim)]
        z = z/200
        y = y[z[:, 1] > 0]
        z = z[z[:, 1] > 0]
        print(z.shape)
        z3 = z[:, [1, 2, 3]]
        print(z3.shape)
        if args.y_dim is not None:
            self.y_train = torch.tensor(y, dtype=torch.float32)
        else:
            self.y_train = None
        self.z_train = torch.tensor(z3, dtype=torch.float32) 
        print(self.z_train.size())

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.y_train[idx], self.z_train[idx]


def get_datasets(args):

    path = "../datasets/train_dataset_fake_jets_only_flows_no_rint.hdf5"

    tr_dataset = SimpleFakesDataset(
        [path],
        y_dim=args.y_dim,
        z_dim=args.z_dim,
        args = args,
        start=args.train_start,
        limit=args.train_limit,
    )

    te_dataset = SimpleFakesDataset(
        [path],
        y_dim=args.y_dim,
        z_dim=args.z_dim,
        args = args,
        start=args.test_start,
        limit=args.test_limit,
    )

    return tr_dataset, te_dataset