import h5py
import torch
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

    def __init__(self, h5_paths, limit, x_dim, y_dim):

        # we must fix a convention for parametrizing slices

        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][:limit, x_dim : (x_dim + y_dim + 1)]
        x = self.archives[0]["data"][:limit, 0:x_dim]
        N = self.archives[0]["data"][:limit, (y_dim + x_dim + 1) : (y_dim + x_dim + 2)]
        self.x_train = torch.tensor(x, dtype=torch.float32)  # .to(device)
        self.y_train = torch.tensor(y, dtype=torch.float32)  # .to(device)
        self.N_train = torch.tensor(N, dtype=torch.float32)  # .to(device)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.N_train[idx]


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
