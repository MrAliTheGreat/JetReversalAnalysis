######################## DISCLAIMER ########################
# THIS CODE ONLY WORKS IF ZARR FILES CAN ALL FIT IN RAM    #
# FOR LARGER ZARR FILES CHUNKED MERGING MUST BE CONSIDERED #
############################################################

import zarr
import numpy as np
from numcodecs import Blosc
import argparse

np.random.seed(7)

parser = argparse.ArgumentParser(description = "Merge two similar zarr datasets")
parser.add_argument("--datasets", nargs = 2, required = True, help = "Dataset paths")
parser.add_argument("--output-zarr", required = True, help = "Output merged zarr dataset path")
args = parser.parse_args()

store1 = zarr.open(args.datasets[0], mode = "r")
store2 = zarr.open(args.datasets[1], mode = "r")

input1 = store1["input"][:]
label1 = store1["label"][:]

input2 = store2["input"][:]
label2 = store2["label"][:]

inputs = np.concatenate([input1, input2], axis = 0)
labels = np.concatenate([label1, label2], axis = 0)

num_samples = inputs.shape[0]
perm = np.random.permutation(num_samples)

inputs_shuffled = inputs[perm]
labels_shuffled = labels[perm]

compressor = Blosc(
    cname = "zstd",
    clevel = 5,
    shuffle = Blosc.BITSHUFFLE
)

store = zarr.open(args.output_zarr, mode = "w")

store.create_dataset(
    name = "input",
    data = inputs_shuffled,
    chunks = (1024, inputs_shuffled.shape[1], inputs_shuffled.shape[2]),
    dtype = "float32",
    compressor = compressor,
)
store.create_dataset(
    name = "label",
    data = labels_shuffled,
    chunks = (1024, labels_shuffled.shape[1], labels_shuffled.shape[2]),
    dtype = "float32",
    compressor = compressor,
)

print(f"Merged and randomly shuffled zarr file created at {args.output_zarr}")
