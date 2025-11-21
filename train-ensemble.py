#!/usr/bin/env python
# coding: utf-8

import json

with open("./params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    seed_vals = data["seed_vals"]
    ensemble_root_path = data["ensemble_root_path"]
    dataset_path_train = data["dataset_path"]["train"]
    dataset_path_val = data["dataset_path"]["validation"]
    dataset_path_test = data["dataset_path"]["test"]
    num_single_sample_timesteps = data["num_single_sample_timesteps"]
    window_stride = data["window_stride"]
    input_window_length = data["input_window_length"]
    label_window_length = data["label_window_length"]
    input_features = data["input_features"]
    label_features = data["label_features"]
    extra_features = data["extra_features"]
    relative_attention_num_buckets = data["relative_attention_num_buckets"]
    embedding_dim = data["embedding_dim"]
    num_attention_head = data["num_attention_head"]
    num_encoder_layers = data["num_encoder_layers"]
    num_decoder_layers = data["num_decoder_layers"]
    position_wise_nn_dim = data["position_wise_nn_dim"]
    dropout = data["dropout"]
    batch_size = data["batch_size"]
    epochs = data["epochs"]
    learning_rate = data["learning_rate"]

import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score, PearsonCorrCoef
from datetime import datetime
import os

from utils.pipeline.Data import get_mean_std_respected_temporal, WindowedIterableDataset
from utils.pipeline.Model import TimeSeriesHuggingFaceTransformer
from utils.pipeline.Run import train, validate
from utils.pipeline.Monitor import Overfit





setup = f'''
dataset: random new dataset 10k
bos_projector: non-linear (1 LeakyReLU)
bos_input: encoder hidden state of last input time-step
positional encoding: sin, cos
Loss: MSE
num_single_sample_timesteps: {num_single_sample_timesteps}
input_window_len: {input_window_length}
label_window_len: {label_window_length}
window_stride: {window_stride}
relative_attention_num_buckets: {relative_attention_num_buckets}
embedding_dim: {embedding_dim}
num_attention_head: {num_attention_head}
num_encoder_layers: {num_encoder_layers}
num_decoder_layers: {num_decoder_layers}
position_wise_nn_dim: {position_wise_nn_dim}
dropout: {dropout}
batch_size: {batch_size}
epochs: {epochs}
learning_rate: {learning_rate}
{input_features} -> {label_features}
extra features: {extra_features}
___________________________________________________________________________________________________________

'''

num_log = len(os.listdir("./ensemble")) - 1
os.mkdir(f"./ensemble/{num_log}")
with open(f"./ensemble/logs/{num_log}.log", mode = "a") as f:
    f.write(datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    f.write(setup)

input_stats = get_mean_std_respected_temporal(
    dataset_path = dataset_path_train,
    cols = input_features,
    num_single_sample_timesteps = num_single_sample_timesteps,
    input_window_len = input_window_length,
    label_window_len = label_window_length,
    window_stride = window_stride
)
if(set(input_features) == set(label_features)):
    output_stats = input_stats
else:
    output_stats = get_mean_std_respected_temporal(
        dataset_path = dataset_path_train,
        cols = label_features,
        num_single_sample_timesteps = num_single_sample_timesteps,
        input_window_len = input_window_length,
        label_window_len = label_window_length,
        window_stride = window_stride
    )

tic = datetime.now()
for seed_val in seed_vals:
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)

    df_train = WindowedIterableDataset(
        dataset_path = dataset_path_train,
        input_stats = input_stats,
        label_stats = output_stats,
        input_features = input_features,
        label_features = label_features,
        extra_features = extra_features,
        num_single_sample_timesteps = num_single_sample_timesteps,
        stride = window_stride,
        input_window_length = input_window_length,
        label_window_length = label_window_length
    )
    data_loader_train = DataLoader(
        df_train,
        batch_size = batch_size,
        pin_memory = True
    )

    df_val = WindowedIterableDataset(
        dataset_path = dataset_path_val,
        input_stats = input_stats,
        label_stats = output_stats,
        input_features = input_features,
        label_features = label_features,
        extra_features = extra_features,
        num_single_sample_timesteps = num_single_sample_timesteps,
        stride = window_stride,
        input_window_length = input_window_length,
        label_window_length = label_window_length
    )
    data_loader_val = DataLoader(
        df_val,
        batch_size = batch_size,
        pin_memory = True
    )

    model = TimeSeriesHuggingFaceTransformer(
        input_window_len = input_window_length,
        output_window_len = label_window_length,
        input_dim = len(input_features),
        output_dim = len(label_features),
        d_model = embedding_dim,
        num_head = num_attention_head,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_encoder_layers,
        position_wise_ffn_dim = position_wise_nn_dim,
        relative_attention_num_buckets = relative_attention_num_buckets,
        dropout = dropout
    ).to(device)

    # overfit_monitor = Overfit()
    # overfit_count = 0

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate
    )

    train_r2 = R2Score(multioutput = "uniform_average").to(device)
    val_r2 = R2Score(multioutput = "uniform_average").to(device)

    train_per_feature_pearson = PearsonCorrCoef(num_outputs = len(label_features)).to(device)
    val_per_feature_pearson = PearsonCorrCoef(num_outputs = len(label_features)).to(device)

    train_per_timestep_r2 = [R2Score(multioutput = "uniform_average").to(device) for _ in range(label_window_length)]
    val_per_timestep_r2 = [R2Score(multioutput = "uniform_average").to(device) for _ in range(label_window_length)]

    train_per_feature_r2 = R2Score(multioutput = "raw_values").to(device)
    val_per_feature_r2 = R2Score(multioutput = "raw_values").to(device)


    for epoch in range(epochs):
        train_loss, train_r2_value, train_ft_r2s, train_ts_r2s, train_feature_pearsons = train(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            r2 = train_r2,
            per_timestep_r2 = train_per_timestep_r2,
            per_feature_r2 = train_per_feature_r2,
            per_feature_pearson = train_per_feature_pearson,
            data_loader = data_loader_train,
            device = device,
            epoch = epoch,
            total_epochs = epochs
        )

        val_loss, val_r2_value, val_ft_r2s, val_ts_r2s, val_feature_pearsons = validate(
            model = model,
            criterion = criterion,
            r2 = val_r2,
            per_timestep_r2 = val_per_timestep_r2,
            per_feature_r2 = val_per_feature_r2,
            per_feature_pearson = val_per_feature_pearson,
            data_loader = data_loader_val,
            device = device,
            epoch = epoch,
            total_epochs = epochs
        )

    # if(overfit_monitor.check(epoch = epoch, train_loss = train_loss, val_loss = val_loss)):
    #     break

    model_path = f"{ensemble_root_path}/{num_log}/T5-{input_window_length}-{label_window_length}-{window_stride}-{seed_val}.pt"

    torch.save(model, model_path)

    with open(f"./ensemble/logs/{num_log}.log", mode = "a") as f:
        f.write(f"seed: {seed_val}\n")
        f.write(f"Num trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        f.write(f"{model_path}\n\n")
        f.write(f"Train Loss: {train_loss}, Train R2: {train_r2_value}\n")
        f.write(f"Val Loss: {val_loss}, Val R2: {val_r2_value}\n\n")
        
        f.write(f"Train Per Feature R2:\n")
        for i in range(len(label_features)):
            f.write(f"    {label_features[i]}: {train_ft_r2s[i]:.6f}\n")
        f.write(f"Val Per Feature R2:\n")
        for i in range(len(label_features)):
            f.write(f"    {label_features[i]}: {val_ft_r2s[i]:.6f}\n")

        f.write("\nTrain Per Feature Pearson:\n")
        f.write(f"    {[f'{f_p:.6f}' for f_p in train_feature_pearsons]}\n")
        f.write("Val Per Feature Pearson:\n")
        f.write(f"    {[f'{f_p:.6f}' for f_p in val_feature_pearsons]}\n")

        # sorted_idxs = np.argsort(train_ts_r2s)
        # f.write("\nTrain worst 10 Time-Steps R2:\n")
        # for idx in sorted_idxs[:10]:
        #     f.write(f"    {idx + 1}: {train_ts_r2s[idx]:.6f}\n")
        # f.write("Train best 10 Time-Steps R2:\n")
        # for idx in sorted_idxs[-10:][::-1]:
        #     f.write(f"    {idx + 1}: {train_ts_r2s[idx]:.6f}\n")

        # sorted_idxs = np.argsort(val_ts_r2s)
        # f.write("Val worst 10 Time-Steps R2:\n")
        # for idx in sorted_idxs[:10]:
        #     f.write(f"    {idx + 1}: {val_ts_r2s[idx]:.6f}\n")
        # f.write("Val best 10 Time-Steps R2:\n")
        # for idx in sorted_idxs[-10:][::-1]:
        #     f.write(f"    {idx + 1}: {val_ts_r2s[idx]:.6f}\n")

        f.write("\n==========================================\n")

    with open(f"./ensemble/logs/{num_log}-timestepR2.log", mode = "a") as f:
        f.write(f"seed: {seed_val}\n")
        f.write("Train R2\n")
        f.write(str(train_ts_r2s))
        f.write("\n\n")
        f.write("Val R2\n")
        f.write(str(val_ts_r2s))
        f.write("\n\n")
        f.write("==========================================\n")


    del model
    torch.cuda.empty_cache()

    print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: {model_path} saved!\n")

toc = datetime.now()
print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: Ensemble training done!")
print(f"Total execution time: {toc - tic}\n")
