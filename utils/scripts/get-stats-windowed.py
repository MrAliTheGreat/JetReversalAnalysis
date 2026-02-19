import zarr
import polars as pl
import argparse
import json
import dask.array as da
import numpy as np

# Use full path instead of ~
with open("/users/labnet5/gr5/abahari/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    input_features = data["input_features"]


parser = argparse.ArgumentParser(description = "Dataset Information")
parser.add_argument(
    "--input-dataset",
    required = True,
    help = "Input CSV Dataset"
)
parser.add_argument(
    "--output-csv",
    required = True,
    help = "Output CSV Mean and Standard Deviation Dataset"
)
args = parser.parse_args()


store = zarr.open(args.input_dataset, mode = "r")
inputs = store["input"]
dx = da.from_zarr(inputs)

mean = dx.mean(axis = (0, 1), dtype = np.float64).compute()
std = dx.std(axis = (0, 1), dtype = np.float64).compute()

stats = {}

for i, name in enumerate(input_features):
    stats[f"{name}_mean"] = [mean[i]]

for i, name in enumerate(input_features):
    stats[f"{name}_std"] = [std[i]]

df = pl.DataFrame(stats)
df.write_csv(args.output_csv)

print(f"Stats calculated and saved at {args.output_csv}")

