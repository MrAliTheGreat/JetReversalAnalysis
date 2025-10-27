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
        .with_columns([
            pl.col(col).str.json_decode(dtype = pl.List(pl.Float32)) for col in cols if col != "eta_list"
        ])
        # .with_columns(
        #     pl.col("eta_list")
        #     .str.json_decode(dtype = pl.List(pl.List(pl.Float32)))
        #     .list.eval(pl.element().flatten())
        #     .alias("eta_list")
        # )
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

def get_mean_std_respected_temporal(dataset_path, cols, num_single_sample_timesteps, input_window_len, label_window_len, window_stride):
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
    if("eta_list" in cols):
        df = (
            pl.scan_csv(
                dataset_path
            )
            .select(
                cols
            )
            .with_columns([
                pl.col(col).str.json_decode(dtype = pl.List(pl.Float32)) for col in cols if col != "eta_list"
            ])
            .with_columns(    # eta dataset
                pl.col("eta_list")
                .str.json_decode(dtype = pl.List(pl.List(pl.Float32)))
                .list.eval(pl.element().flatten())
            )
            .explode("*")
            .with_columns([
                pl.arange(0, pl.count()).alias("row_idx")
            ])
            .with_columns([
                # (pl.col("row_idx") // num_single_sample_timesteps).alias("timeseries_idx"),
                (pl.col("row_idx") % num_single_sample_timesteps).alias("timestep_idx")
            ])
            .drop("row_idx")
        )
    else:
        df = (
            pl.scan_csv(
                dataset_path
            )
            .select(
                cols
            )
            .with_columns([
                pl.col("*").str.json_decode(dtype = pl.List(pl.Float32))
            ])
            .explode("*")
            .with_columns([
                pl.arange(0, pl.count()).alias("row_idx")
            ])
            .with_columns([
                # (pl.col("row_idx") // num_single_sample_timesteps).alias("timeseries_idx"),
                (pl.col("row_idx") % num_single_sample_timesteps).alias("timestep_idx")
            ])
            .drop("row_idx")
        )        

    res = []
    agg_exprs = []
    for col in cols:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))

    for window_end_idx in range(input_window_len, num_single_sample_timesteps - label_window_len + 1, window_stride):
        windowed_df = df.filter(pl.col("timestep_idx") < window_end_idx)

        windowed_df = windowed_df.select(agg_exprs).with_columns(
            pl.lit(window_end_idx).alias("window_end_idx")
        )

        res.append(windowed_df.collect(engine = "streaming"))

    return pl.concat(res)


#################### CLASSES ####################

class WindowedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset_path,
            input_stats,
            label_stats,
            input_features,
            label_features,
            num_single_sample_timesteps,
            stride,
            input_window_length,
            label_window_length,
            chunk_size = 512,
            inference = False
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
        self.inference = inference

        # self.means = self.dict2numpy(stats["mean"], input_features)
        # self.stds = self.dict2numpy(stats["std"], input_features)
        # self.stds[self.stds == 0] = 10 ** -8

        self.input_means = self.pl2numpy(input_stats, input_features, "mean")
        self.input_stds = self.pl2numpy(input_stats, input_features, "std")
        self.input_stds[self.input_stds == 0] = 10 ** -8

        self.label_means = self.pl2numpy(label_stats, label_features, "mean")
        self.label_stds = self.pl2numpy(label_stats, label_features, "std")
        self.label_stds[self.label_stds == 0] = 10 ** -8

    @staticmethod
    def dict2numpy(d, cols):
        vals = []
        for col in cols:
            vals.append(d[col])
        return np.array(vals)

    @staticmethod
    def pl2numpy(df, cols, target_stat):
        return df.select(
            [col + "_" + target_stat for col in cols]
        ).to_numpy()
    
    def __iter__(self):
        df_reader = pl.read_csv_batched(self.dataset_path, batch_size = self.chunk_size)

        while(True):
            new_chunk = df_reader.next_batches(1)
            if(new_chunk is None):
                break
            
            data_chunk = new_chunk[0]
            if("eta_list" in data_chunk.columns):
                data_chunk = (
                    data_chunk
                    .drop(["id", "eps", "n_0_squared"])    # No eps or n_0_squared!
                    .with_columns([
                        pl.col(feature)
                        .str.json_decode(dtype = pl.List(pl.Float32))
                        for feature in self.input_features if feature != "eta_list"
                    ])
                    .with_columns([
                        pl.col("eta_list")
                        .str.json_decode(dtype = pl.List(pl.List(pl.Float32)))
                        .list.eval(pl.element().flatten())
                    ])
                )
            else:
                data_chunk = (
                    data_chunk
                    .drop(["id", "eps", "n_0_squared"])    # No eps or n_0_squared!
                    .with_columns([
                        pl.col("*").str.json_decode(dtype = pl.List(pl.Float32))
                    ])
                )

            input_df = data_chunk.select(self.input_features).explode("*").to_numpy().reshape(data_chunk.shape[0], self.num_single_sample_timesteps, len(self.input_features))
            label_df = data_chunk.select(self.label_features).explode("*").to_numpy().reshape(data_chunk.shape[0], self.num_single_sample_timesteps, len(self.label_features))

            # Permutation substitutes shuffle=True in Dataloader!
            for time_series_idx in np.random.permutation(data_chunk.shape[0]):
                # If using get_mean_std no enumerate 
                for i, input_window_start_idx in enumerate(range(0, self.num_single_sample_timesteps - self.valid_length + 1, self.stride)):
                    label_window_start_idx = input_window_start_idx + self.input_window_length
                    
                    input_window = input_df[time_series_idx, input_window_start_idx: label_window_start_idx, :]
                    label_window = label_df[time_series_idx, label_window_start_idx: label_window_start_idx + self.label_window_length, :]
                    label_full = label_df[time_series_idx, :, :]
                    
                    input_window = (input_window - self.input_means[i, :]) / self.input_stds[i, :]
                    label_window = (label_window - self.label_means[i, :]) / self.label_stds[i, :]
                    # label_full = (label_full - self.means) / self.stds
    
                    input_window = torch.tensor(input_window, dtype = torch.float)
                    label_window = torch.tensor(label_window, dtype = torch.float)
                    label_full = torch.tensor(label_full, dtype = torch.float)

                    if(self.inference):
                        yield input_window, label_window, label_full
                    else:
                        yield input_window, label_window
