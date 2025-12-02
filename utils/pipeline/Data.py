import torch
import zarr
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
