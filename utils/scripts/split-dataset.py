import polars as pl
import numpy as np
import json

# Use full path instead of ~
with open("~/Documents/Thesis/src/params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    seed_val = data["seed_val"]
    dataset_path = "~/Documents/Thesis/src/dataset/reversalData_minor.csv"
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

np.random.seed(seed_val)


def get_num_data_points(dataset_path):
    print(f"Getting the number of data points in {dataset_path}")
    return pl.scan_csv(dataset_path).select(pl.len()).collect().item()

def split_dataset(dataset_path, train_ratio, val_ratio, test_ratio):
    if(not np.isclose(train_ratio + val_ratio + test_ratio, 1.0)):
        print("Ratios do not sum up to 1!")
        return

    print("Splitting the dataset...")
    df = (
        pl.scan_csv(dataset_path)
        .with_columns(
            (pl.col("id").hash(seed = seed_val) / pl.datatypes.UInt64.max()).alias("rand_id")
        )
    )

    df_train = df.filter(
        pl.col("rand_id") < train_ratio
    ).drop("rand_id")
    df_val = df.filter(
        (pl.col("rand_id") >= train_ratio) & (pl.col("rand_id") < train_ratio + val_ratio)
    ).drop("rand_id")
    df_test = df.filter(
        pl.col("rand_id") >= train_ratio + val_ratio
    ).drop("rand_id")

    df_train.sink_csv(f"{dataset_path[:-4]}-train.csv")
    print("Train Dataset Saved!")
    df_val.sink_csv(f"{dataset_path[:-4]}-val.csv")
    print("Validation Dataset Saved!")
    df_test.sink_csv(f"{dataset_path[:-4]}-test.csv")
    print("Test Dataset Saved!")


# split_dataset(
#     dataset_path,
#     train_ratio,
#     val_ratio,
#     test_ratio
# )

# print(f"Number of train data points: {get_num_data_points(dataset_path)}")
# print(f"Number of train data points: {get_num_data_points(f'{dataset_path[:-4]}-train.csv')}")
# print(f"Number of validation data points: {get_num_data_points(f'{dataset_path[:-4]}-val.csv')}")
# print(f"Number of test data points: {get_num_data_points(f'{dataset_path[:-4]}-test.csv')}")
