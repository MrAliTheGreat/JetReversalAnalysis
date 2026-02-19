import polars as pl
import json
from numcodecs import Blosc
import zarr
import numpy as np
import argparse


# Use full path instead of ~
with open("/users/labnet5/gr5/abahari/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    label_features = data["label_features"]
    extra_features = data["extra_features"]

parser = argparse.ArgumentParser(description = "Dataset Information")
parser.add_argument("--input-dataset", required = True, help = "Input CSV Test Dataset")
parser.add_argument("--output-zarr", required = True, help = "Output Zarr Dataset")
parser.add_argument("--num-timesteps", required = True, help = "num_single_sample_timesteps")
args = parser.parse_args()

num_single_sample_timesteps = int(args.num_timesteps)


compressor = Blosc(
    cname = "zstd",
    clevel = 5,
    shuffle = Blosc.BITSHUFFLE
)
store = zarr.open(args.output_zarr, mode = "w")

labels_store = store.create(
    name = "label",
    shape = (0, num_single_sample_timesteps, len(label_features)),
    chunks = (1024, num_single_sample_timesteps, len(label_features)),
    dtype = "float32",
    compressor = compressor,
    fill_value = 0,
    overwrite = True
)
extras_store = store.create(
    name = "extra",
    shape = (0, num_single_sample_timesteps, len(extra_features)),
    chunks = (1024, num_single_sample_timesteps, len(extra_features)),
    dtype = "float32",
    compressor = compressor,
    fill_value = 0,
    overwrite = True
)

buffer_size = 128
buffer_labels = np.zeros((buffer_size, num_single_sample_timesteps, len(label_features)))
buffer_extras = np.zeros((buffer_size, num_single_sample_timesteps, len(extra_features)))
df_reader = pl.read_csv_batched(args.input_dataset, batch_size = 50000)

while(True):
    new_chunk = df_reader.next_batches(64)
    if(new_chunk is None):
        break
    
    for data_chunk in new_chunk:
        data_chunk = (
            data_chunk
            .drop(["id"])    # No eps or n_0_squared!
            .with_columns([
                pl.col(feature)
                .str.json_decode(dtype = pl.List(pl.Float32))
                for feature in label_features + extra_features
            ])
            .with_columns([
                pl.col("eta_list")
                .str.json_decode(dtype = pl.List(pl.List(pl.Float32)))
                .list.eval(pl.element().flatten())
            ])
        )

        label_df = data_chunk.select(
            label_features
        ).explode("*").to_numpy().reshape(
            data_chunk.shape[0], num_single_sample_timesteps, len(label_features)
        )
        extra_df = data_chunk.select(
            extra_features
        ).explode("*").to_numpy().reshape(
            data_chunk.shape[0], num_single_sample_timesteps, len(extra_features)
        )

        buffer_idx = 0

        for time_series_idx in range(data_chunk.shape[0]):
            labels = label_df[time_series_idx, :, :]
            extras = extra_df[time_series_idx, :, :]

            buffer_labels[buffer_idx, :, :] = labels
            buffer_extras[buffer_idx, :, :] = extras
            buffer_idx += 1

            if(buffer_idx >= buffer_size):
                labels_store.append(buffer_labels)
                extras_store.append(buffer_extras)
                buffer_idx = 0

        if(buffer_idx < buffer_size):
            labels_store.append(buffer_labels[:buffer_idx, :, :])
            extras_store.append(buffer_extras[:buffer_idx, :, :])

print(f"{args.output_zarr} created!")

