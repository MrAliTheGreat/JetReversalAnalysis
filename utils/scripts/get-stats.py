import polars as pl
import json

# Use full path instead of ~
with open("~/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    stats_dataset_path = data["dataset_path"]["stats"]
    input_features = data["input_features"]
    final_stats_path = "/mnt/abahari/stats.csv"


def get_mean_std(dataset_path, cols):
    '''
        input:
            dataset_path: path to csv dataset
            cols: target columns in the dataset for getting the stats
        output:
            pl.DataFrame with each feature having _mean and _std column
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
            .select([
                pl.col("*").mean().name.suffix("_mean"),
                pl.col("*").std().name.suffix("_std")
            ])
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
            .select([
                pl.col("*").mean().name.suffix("_mean"),
                pl.col("*").std().name.suffix("_std")
            ])
        )        

    return df.collect(engine = "streaming")



stats = get_mean_std(
    dataset_path = stats_dataset_path,
    cols = input_features,
)

stats.write_csv(final_stats_path)
print(f"Stats calculated save to path: {final_stats_path}")
