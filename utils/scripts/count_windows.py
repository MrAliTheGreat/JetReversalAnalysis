import polars as pl
import json
from numcodecs import Blosc
import zarr
import numpy as np
import argparse

np.random.seed(0)


# Use full path instead of ~
with open("/users/labnet5/gr5/abahari/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    num_single_sample_timesteps = data["num_single_sample_timesteps"]
    input_window_length = data["input_window_length"]
    label_window_length = data["label_window_length"]
    window_stride = data["window_stride"]
    valid_length = input_window_length + label_window_length
    input_features = data["input_features"]
    label_features = data["label_features"]

parser = argparse.ArgumentParser(description = "Dataset Information")
parser.add_argument(
    "--input-dataset",
    required = True,
    help = "Input CSV Dataset"
)
parser.add_argument(
    "--mode",
    required = True,
    help = "stream and reversal"
)
args = parser.parse_args()


df_reader = pl.read_csv_batched(args.input_dataset, batch_size = 50000)
num_selected_windows = 0
num_total_windows = 0

while(True):
    new_chunk = df_reader.next_batches(64)
    if(new_chunk is None):
        break
    
    for data_chunk in new_chunk:
        if("eta_list" in data_chunk.columns):
            data_chunk = (
                data_chunk
                .drop(["id"])    # No eps or n_0_squared!
                .with_columns([
                    pl.col(feature)
                    .str.json_decode(dtype = pl.List(pl.Float32))
                    for feature in input_features if feature != "eta_list"
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
                .drop(["id"])    # No eps or n_0_squared!
                .with_columns([
                    pl.col("*").str.json_decode(dtype = pl.List(pl.Float32))
                ])
            )

        input_df = data_chunk.select(
            input_features
        ).explode("*").to_numpy().reshape(
            data_chunk.shape[0], num_single_sample_timesteps, len(input_features)
        )
        label_df = data_chunk.select(
            label_features
        ).explode("*").to_numpy().reshape(
            data_chunk.shape[0], num_single_sample_timesteps, len(label_features)
        )

        for time_series_idx in range(data_chunk.shape[0]):
            for input_window_start_idx in range(0, num_single_sample_timesteps - valid_length + 1, window_stride):
                label_window_start_idx = input_window_start_idx + input_window_length

                input_window = input_df[time_series_idx, input_window_start_idx: label_window_start_idx, :]
                label_window = label_df[time_series_idx, label_window_start_idx: label_window_start_idx + label_window_length, :]
                num_total_windows += 1

                if((args.mode == "stream") and (input_window_start_idx <= 5000)):
                    continue

                if((args.mode == "reversal") and not (input_window_start_idx <= num_single_sample_timesteps // 2 and input_window_start_idx + valid_length > num_single_sample_timesteps // 2)):
                    continue

                num_selected_windows += 1


print(f"Selected Windows: {num_selected_windows}, Total Windows: {num_total_windows}")
print(f"Selected/Total Ratio: {num_selected_windows / num_total_windows}")
