import torch
import polars as pl
import numpy as np



#################### METHODS ####################

def get_mean_std(dataset_path, cols):
    '''
        input:
            dataset_path: path to csv dataset
            cols: target columns in the dataset for getting the stats
        output:
            {
                mean: {col: mean_val},
                std: {col: std_val}
            }
    '''
    res = (
        pl.scan_csv(
            dataset_path
        )
        .select(
            cols
        )
        .select(
            pl.col("*").str.json_decode(dtype = pl.List(pl.Float32))
        )
        .explode("*")
        .select([
            pl.col("*").mean().name.suffix("_mean"),
            pl.col("*").std().name.suffix("_std")
        ])
    ).collect(engine = "streaming")

    stats = {
        "mean": {},
        "std": {}
    }

    for col in cols:
        stats["mean"][col] = res[f"{col}_mean"][0]
        stats["std"][col] = res[f"{col}_std"][0]
        
    return stats


#################### CLASSES ####################

class WindowedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset_path,
            stats,
            input_features,
            label_features,
            num_single_sample_timesteps,
            stride,
            input_window_length,
            label_window_length,
            chunk_size = 512
        ):
        '''
            input_df & label_df
                Type: Numpy
                Shape: Number of time-series, Number of time-steps, Number of input features
            
            input_features & label_features
                Type: List

            multi worker setup can cause problems here!
            There is a fix though! Assign unique data iteration based on worker info.
        '''

        super().__init__()
        
        self.dataset_path = dataset_path
        self.input_features = input_features
        self.label_features = label_features
        self.num_single_sample_timesteps = num_single_sample_timesteps
        self.stride = stride
        self.input_window_length = input_window_length
        self.label_window_length = label_window_length
        self.valid_length = self.input_window_length + self.label_window_length
        self.chunk_size = chunk_size
        self.means = self.dict2numpy(stats["mean"], input_features)
        self.stds = self.dict2numpy(stats["std"], input_features)
        self.stds[self.stds == 0] = 10 ** -8

    @staticmethod
    def dict2numpy(d, cols):
        vals = []
        for col in cols:
            vals.append(d[col])
        return np.array(vals)
    
    def __iter__(self):
        df_reader = pl.read_csv_batched(self.dataset_path, batch_size = self.chunk_size)

        while(True):
            new_chunk = df_reader.next_batches(1)
            if(new_chunk is None):
                break
            
            data_chunk = new_chunk[0]
            data_chunk = (
                data_chunk
                .drop(["id"])    # No eps or n_0_squared!
                .select(
                    pl.col("*").str.json_decode()
                )
                # .with_columns(
                #     eta_list = pl.col("eta_list").list.eval(pl.element().flatten(), parallel = True)
                # )
            )
            input_df = data_chunk.select(self.input_features).explode("*").to_numpy().reshape(data_chunk.shape[0], self.num_single_sample_timesteps, len(self.input_features))
            label_df = data_chunk.select(self.label_features).explode("*").to_numpy().reshape(data_chunk.shape[0], self.num_single_sample_timesteps, len(self.label_features))

            # Permutation substitutes shuffle=True in Dataloader!
            for time_series_idx in np.random.permutation(data_chunk.shape[0]):
                for input_window_start_idx in range(0, self.num_single_sample_timesteps - self.valid_length + 1, self.stride):
                    label_window_start_idx = input_window_start_idx + self.input_window_length
                    
                    input_window = input_df[time_series_idx, input_window_start_idx: label_window_start_idx, :]
                    label_window = label_df[time_series_idx, label_window_start_idx: label_window_start_idx + self.label_window_length, :]

                    # input_window_mean = input_window.mean(axis = 0)
                    # input_window_std = input_window.std(axis = 0)
                    # input_window_std[input_window_std == 0] = 10 ** -8
                    # input_window = (input_window - input_window_mean) / input_window_std
                    
                    input_window = (input_window - self.means) / self.stds
                    label_window = (label_window - self.means) / self.stds

                    yield torch.tensor(input_window, dtype = torch.float), torch.tensor(label_window, dtype = torch.float)