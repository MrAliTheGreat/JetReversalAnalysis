import polars as pl
import json
from numcodecs import Blosc
import zarr
import numpy as np
import argparse


# Use full path instead of ~
with open("/users/labnet5/gr5/abahari/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    stats_path = data["stats_path"]
    num_single_sample_timesteps = data["num_single_sample_timesteps"]
    input_window_length = data["input_window_length"]
    label_window_length = data["label_window_length"]
    window_stride = data["window_stride"]
    valid_length = input_window_length + label_window_length
    input_features = data["input_features"]
    label_features = data["label_features"]

parser = argparse.ArgumentParser(description = "Dataset Information")
parser.add_argument("--input-dataset", required = True, help = "Input CSV Dataset")
parser.add_argument("--output-zarr", required = True, help = "Output Zarr Dataset")
args = parser.parse_args()



def pl2numpy(df, cols, target_stat):
    return df.select(
        [col + "_" + target_stat for col in cols]
    ).to_numpy()

compressor = Blosc(
    cname = "zstd",
    clevel = 5,
    shuffle = Blosc.BITSHUFFLE
)
store = zarr.open(args.output_zarr, mode = "w")

inputs = store.create(
    name = "input",
    shape = (0, input_window_length, len(input_features)),
    chunks = (1024, input_window_length, len(input_features)),
    dtype = "float32",
    compressor = compressor,
    fill_value = 0,
    overwrite = True
)
labels = store.create(
    name = "label",
    shape = (0, label_window_length, len(label_features)),
    chunks = (1024, label_window_length, len(label_features)),
    dtype = "float32",
    compressor = compressor,
    fill_value = 0,
    overwrite = True
)


stats = pl.read_csv(stats_path)

input_means = pl2numpy(stats, input_features, "mean")
input_stds = pl2numpy(stats, input_features, "std")
input_stds[input_stds == 0] = 10 ** -8

label_means = pl2numpy(stats, label_features, "mean")
label_stds = pl2numpy(stats, label_features, "std")
label_stds[label_stds == 0] = 10 ** -8

buffer_size = 16384; buffer_idx = 0
buffer_inputs = np.zeros((buffer_size, input_window_length, len(input_features)))
buffer_labels = np.zeros((buffer_size, label_window_length, len(label_features)))
df_reader = pl.read_csv_batched(args.input_dataset, batch_size = 50000)
total_windows = 0

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

                input_window = (input_window - input_means) / input_stds
                label_window = (label_window - label_means) / label_stds

                buffer_inputs[buffer_idx, :, :] = input_window
                buffer_labels[buffer_idx, :, :] = label_window
                buffer_idx += 1
                total_windows += 1

                if(buffer_idx >= buffer_size):
                    inputs.append(buffer_inputs)
                    labels.append(buffer_labels)
                    buffer_idx = 0

        if(buffer_idx < buffer_size):
            inputs.append(buffer_inputs[0: buffer_idx, :, :])
            labels.append(buffer_labels[0: buffer_idx, :, :])

print(f"{args.output_zarr} created!")