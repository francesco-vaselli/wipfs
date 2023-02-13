import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    """Very simple Dataset for reading hdf5 data
        This is way simpler than muons as we heve enough jets in a single file
        Still, dataloading is a bottleneck even here
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, limit):

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][:limit, 0:14]
        x = self.archives[0]["data"][:limit, 14:31]
        self.x_train = torch.tensor(x, dtype=torch.float32)  # .to(device)
        self.y_train = torch.tensor(y, dtype=torch.float32)  # .to(device)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class FakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, x_dim : (x_dim + y_dim)]
        x = self.archives[0]["data"][start:limit, 0:x_dim]
        N = self.archives[0]["data"][start:limit, (y_dim + x_dim) : (y_dim + x_dim + 1)]
        self.x_train = torch.tensor(x, dtype=torch.float32).view(
            -1, 1, x_dim
        )  # reshape needed for CONV1D
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.N_train = torch.tensor(N, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.N_train[idx]


class NewFakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, x_dim : (x_dim + y_dim)]
        x = self.archives[0]["data"][start:limit, 0:x_dim]
        z = self.archives[0]["data"][
            start:limit, (y_dim + x_dim) : (y_dim + x_dim + z_dim)
        ]
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        self.y_train = torch.tensor(y, dtype=torch.float32)
        z[:, [1, 2, 3]] = z[:, [1, 2, 3]] / 200.0
        self.z_train = torch.tensor(z, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class NewVarsDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices
        z_dim = z_dim+1 # this is done to preprocess the last two variables into one
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        x = self.archives[0]["data"][start:limit, 0:x_dim]
        y = self.archives[0]["data"][start:limit, x_dim : (x_dim + y_dim)]
        z = self.archives[0]["data"][
            start:limit, (y_dim + x_dim) : (y_dim + x_dim + z_dim)
        ]
        angle = torch.tensor(np.arctan2(z[:, 3], z[:, 2]))
        z[:, [1, 2, 3]] = z[:, [1, 2, 3]] / 200.0
        z1 = torch.tensor(z[:, [0, 1]])
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.z_train = torch.tensor(torch.cat((z1, angle), dim=1), dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class NoZeroFakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, x_dim : (x_dim + y_dim)]
        x = self.archives[0]["data"][start:limit, 0:x_dim]
        z = self.archives[0]["data"][
            start:limit, (y_dim + x_dim) : (y_dim + x_dim + z_dim)
        ]
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        z[:, [1, 2, 3]] = z[:, [1, 2, 3]] / 200.0
        y = y[z[:, 1] > 0]
        z = z[z[:, 1] > 0]
        # print(f"y shape: {y.shape}, z shape: {z.shape}")
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.z_train = torch.tensor(z, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class SortedNoZeroFakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, x_dim : (x_dim + y_dim)]
        x = self.archives[0]["data"][start:limit, 0:x_dim]
        z = self.archives[0]["data"][
            start:limit, (y_dim + x_dim) : (y_dim + x_dim + z_dim)
        ]
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        z[:, [1, 2, 3]] = z[:, [1, 2, 3]] / 200.0
        y = y[z[:, 1] > 0]
        z = z[z[:, 1] > 0]
        _, indx = torch.sort(torch.tensor(y[:, 2]), dim=0, descending=True)
        y = y[indx.numpy()]
        z = z[indx.numpy()]
        # print(f"y shape: {y.shape}, z shape: {z.shape}")
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.z_train = torch.tensor(z, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class SimpleFakesDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, 32]
        x = self.archives[0]["data"][start:limit, 0:x_dim]
        z = self.archives[0]["data"][start:limit, (6 + x_dim) : (6 + x_dim + z_dim)]
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        self.y_train = torch.tensor(y, dtype=torch.float32)
        # z[:, 1] = z[:, 1] / 200.0
        self.z_train = torch.tensor(z, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class SimpleMuonsDataset(Dataset):
    """Very simple Dataset for reading hdf5 data for fakes
        divides each row into 3 parts: reco (x), gen (y), N of fakes (N)
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, x_dim, y_dim, z_dim, start=0, limit=-1):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start:limit, 2]
        x = self.archives[0]["data"][start:limit, 30:52]
        z = self.archives[0]["data"][start:limit, [34, 48]]
        self.x_train = torch.tensor(
            x, dtype=torch.float32
        )  # .view(-1, 1, x_dim) no reshape because no conv1d
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.z_train = torch.tensor(z, dtype=torch.float32)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.z_train[idx]


class ReadDataset(Dataset):
    """Very simple Dataset for reading hdf5 data
        This is way simpler than muons as we heve enough jets in a single file
        Still, dataloading is a bottleneck even here
    Args:
        Dataset (Pytorch Dataset): Pytorch Dataset class
    """

    def __init__(self, h5_paths, limit, x_dim, y_dim):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][:limit, 0:y_dim]
        x = self.archives[0]["data"][:limit, y_dim : (y_dim + x_dim)]
        self.x_train = torch.tensor(x, dtype=torch.float32)  # .to(device)
        self.y_train = torch.tensor(y, dtype=torch.float32)  # .to(device)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class H5Dataset(Dataset):
    def __init__(self, h5_paths, x_dim, y_dim, limit=-1):
        """Initialize the class, set indexes across datasets and define lazy loading
        Args:
            h5_paths (strings): paths to the various hdf5 files to include in the final Dataset
            limit (int, optional): optionally limit dataset length to specified values, if negative
                returns the full length as inferred from files. Defaults to -1.
        """
        max_events = int(5e9)
        self.limit = max_events if limit == -1 else int(limit)
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]

        self.strides = []
        for archive in self.archives:
            with archive as f:
                self.strides.append(len(f["data"]))

        self.len_in_files = self.strides[1:]
        self.strides = np.cumsum(self.strides)
        self._archives = None

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.strides, index, side="right")
        idx_in_file = index - self.strides[max(0, file_idx - 1)]
        y = self.archives[file_idx]["data"][idx_in_file, 0:y_dim]
        x = self.archives[file_idx]["data"][idx_in_file, y_dim : (x_dim + y_dim)]
        y = torch.from_numpy(y)
        x = torch.from_numpy(x)
        # x = x.float()
        # y = y.float()

        return x, y

    def __len__(self):
        # return self.strides[-1] #this will process all files
        if self.limit <= self.strides[-1]:
            return self.limit
        else:
            return self.strides[-1]


class H5FakesDataset(Dataset):
    def __init__(self, h5_paths, x_dim, y_dim, limit=-1):
        """Initialize the class, set indexes across datasets and define lazy loading
        Args:
            h5_paths (strings): paths to the various hdf5 files to include in the final Dataset
            limit (int, optional): optionally limit dataset length to specified values, if negative
                returns the full length as inferred from files. Defaults to -1.
        """
        max_events = int(5e9)
        self.limit = max_events if limit == -1 else int(limit)
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]

        self.strides = []
        for archive in self.archives:
            with archive as f:
                self.strides.append(len(f["data"]))

        self.len_in_files = self.strides[1:]
        self.strides = np.cumsum(self.strides)
        self._archives = None

        self.x_dim = x_dim
        self.y_dim = y_dim

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.strides, index, side="right")
        idx_in_file = index - self.strides[max(0, file_idx - 1)]
        y = self.archives[file_idx]["data"][
            idx_in_file, self.x_dim : (self.x_dim + self.y_dim)
        ]
        x = self.archives[file_idx]["data"][idx_in_file, 0 : self.x_dim]
        N = self.archives[file_idx]["data"][
            idx_in_file, (self.y_dim + self.x_dim) : (self.y_dim + self.x_dim + 1)
        ]
        x = torch.tensor(x, dtype=torch.float32).view(
            -1, self.x_dim
        )  # differently from FakesDataset now we are getting the single item to be batched
        y = torch.tensor(y, dtype=torch.float32)
        N = torch.tensor(N, dtype=torch.float32)
        # x = x.float()
        # y = y.float()

        return x, y, N

    def __len__(self):
        # return self.strides[-1] #this will process all files
        if self.limit <= self.strides[-1]:
            return self.limit
        else:
            return self.strides[-1]
