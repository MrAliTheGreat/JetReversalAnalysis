import torch
import zarr, fsspec
import numpy as np



class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.store = zarr.open(dataset_path, mode = "r")
        self.inputs = self.store["input"]
        self.labels = self.store["label"]
        self.num_datapoints = self.inputs.shape[0]

    def __len__(self):
        return self.num_datapoints
    
    def __getitem__(self, index):
        x = np.asarray(self.inputs[index])
        y = np.asarray(self.labels[index])

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y



class InferenceAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, windowed_dataset_path, full_dataset_path, index_full_update_len):
        super().__init__()

        self.store_windowed = zarr.open(windowed_dataset_path, mode = "r")
        self.inputs_windowed = self.store_windowed["input"]
        self.labels_windowed = self.store_windowed["label"]

        self.store_full = zarr.open(full_dataset_path, mode = "r")
        self.labels_full = self.store_full["label"]
        self.extras_full = self.store_full["extra"]

        self.index_full_update_len = index_full_update_len

        self.num_datapoints = min(self.labels_windowed.shape[0], self.labels_full.shape[0] * index_full_update_len)

    def __len__(self):
        return self.num_datapoints
    
    def __getitem__(self, index):
        index_windowed = index
        index_full = index // self.index_full_update_len

        input_window = np.asarray(self.inputs_windowed[index_windowed])
        label_window = np.asarray(self.labels_windowed[index_windowed])
        input_window = torch.from_numpy(input_window).float()
        label_window = torch.from_numpy(label_window).float()

        label_full = np.asarray(self.labels_full[index_full])
        extra_full = np.asarray(self.extras_full[index_full])
        label_full = torch.from_numpy(label_full).float()
        extra_full = torch.from_numpy(extra_full).float()

        return input_window, label_window, label_full, extra_full

