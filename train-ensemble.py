import json

with open("./params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    seed_vals = data["seed_vals"]
    ensemble_root_path = data["ensemble_root_path"]
    dataset_path_train = data["dataset_path"]["train"]
    dataset_path_val = data["dataset_path"]["validation"]
    dataset_finetune_path_train = data["finetune_reversals_dataset_path"]["train"]
    dataset_finetune_path_val = data["finetune_reversals_dataset_path"]["validation"]
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

from utils.pipeline.Data import WindowedDataset
from utils.pipeline.Model import TimeSeriesHuggingFaceTransformer
from utils.pipeline.Run import train, validate
from utils.pipeline.Monitor import Overfit



def create_data_loaders(dataset_path_train, dataset_path_val):
    df_train = WindowedDataset(dataset_path = dataset_path_train)
    data_loader_train = DataLoader(
        df_train,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 8,
        prefetch_factor = 4,
        pin_memory = True,
        persistent_workers = True
    )

    df_val = WindowedDataset(dataset_path = dataset_path_val)
    data_loader_val = DataLoader(
        df_val,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 8,
        prefetch_factor = 4,
        pin_memory = True,
        persistent_workers = True
    )

    return data_loader_train, data_loader_val


def run_model_process(model, device, data_loader_train, data_loader_val, optimizer, label_features, label_window_length, epochs, model_path, mode = "pretrain"):
    # overfit_monitor = Overfit()
    # overfit_count = 0

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

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

    with open(f"./ensemble/logs/{num_log}-{mode}.log", mode = "a") as f:
        f.write(f"seed: {seed_val}\n")
        f.write(f"mode: {mode}\n")
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

        f.write("\n==========================================\n")

    with open(f"./ensemble/logs/{num_log}-{mode}-timestepR2.log", mode = "a") as f:
        f.write(f"seed: {seed_val}\n")
        f.write(f"mode: {mode}\n")
        f.write("Train R2\n")
        f.write(str(train_ts_r2s))
        f.write("\n\n")
        f.write("Val R2\n")
        f.write(str(val_ts_r2s))
        f.write("\n\n")
        f.write("==========================================\n")

    print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: {model_path} saved!\n")

    return model

def prepare_model_finetuning(model):
    # Freeze nothing
    for p in model.parameters():
        p.requires_grad = True

    # # Freeze everything
    # for p in model.parameters():
    #     p.requires_grad = False

    # # Unfreeze decoder, lm_head, bos_projector, and last encoder block
    # for p in model.decoder.parameters():
    #     p.requires_grad = True

    # for p in model.lm_head.parameters():
    #     p.requires_grad = True

    # for p in model.bos_projector.parameters():
    #     p.requires_grad = True

    # for p in model.encoder.block[-1].parameters():
    #     p.requires_grad = True

    return model





setup = f'''
dataset: pretrain stream dataset + finetune both direction reversal dataset (Nothing frozen)
bos_projector: non-linear (1 LeakyReLU)
bos_input: encoder hidden state of last input time-step
positional encoding: sin, cos
Loss: MAE + 0.1 * sum(Symmetry MAE(u, e, plus))
num_single_sample_timesteps: {num_single_sample_timesteps}
input_window_len: {input_window_length}
label_window_len: {label_window_length}
window_stride: {window_stride}, 10 reversals
relative_attention_num_buckets: {relative_attention_num_buckets}
embedding_dim: {embedding_dim}
num_attention_head: {num_attention_head}
num_encoder_layers: {num_encoder_layers}
num_decoder_layers: {num_decoder_layers}
position_wise_nn_dim: {position_wise_nn_dim}
dropout: {dropout}
batch_size: {batch_size}
epochs: {epochs}, /2 finetuning
learning_rate: {learning_rate}, *0.05 finetuning
{input_features} -> {label_features}
extra features: {extra_features}
___________________________________________________________________________________________________________

'''

num_log = len(os.listdir("./ensemble")) - 1
os.mkdir(f"./ensemble/{num_log}")
with open(f"./ensemble/logs/{num_log}-info.log", mode = "a") as f:
    f.write(datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    f.write(setup)

tic = datetime.now()
for seed_val in seed_vals:
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)

    data_loader_train, data_loader_val = create_data_loaders(
        dataset_path_train = dataset_path_train,
        dataset_path_val = dataset_path_val
    )

    # Same input and output window! Just shorter stride (10 vs 35) for more detailed data
    data_loader_train_finetune, data_loader_val_finetune = create_data_loaders(
        dataset_path_train = dataset_finetune_path_train,
        dataset_path_val = dataset_finetune_path_val
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

    # model = torch.compile(model)    # nvcc not found!

    # model_path = f"{ensemble_root_path}/{num_log}/T5-{input_window_length}-{label_window_length}-{window_stride}-{seed_val}.pt"

    model_path = f"{ensemble_root_path}/{num_log}/T5-{input_window_length}-{label_window_length}-p{window_stride}-f10-{seed_val}.pt"


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate
    )
    model = run_model_process(
        model = model,
        device = device,
        data_loader_train = data_loader_train,
        data_loader_val = data_loader_val,
        optimizer = optimizer,
        label_features = label_features,
        label_window_length = label_window_length,
        epochs = epochs,
        model_path = model_path,
        mode = "pretrain"
    )

    model = prepare_model_finetuning(model = model)

    optimizer_finetune = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate * 0.05
    )
    model = run_model_process(
        model = model,
        device = device,
        data_loader_train = data_loader_train_finetune,
        data_loader_val = data_loader_val_finetune,
        optimizer = optimizer_finetune,
        label_features = label_features,
        label_window_length = label_window_length,
        epochs = int(epochs / 2),
        model_path = model_path,
        mode = "finetune"
    )

    torch.save(model, model_path)

    del model
    torch.cuda.empty_cache()



    ### BEFORE MODULARIZATION ###
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr = learning_rate
    # )

    # train_r2 = R2Score(multioutput = "uniform_average").to(device)
    # val_r2 = R2Score(multioutput = "uniform_average").to(device)

    # train_per_feature_pearson = PearsonCorrCoef(num_outputs = len(label_features)).to(device)
    # val_per_feature_pearson = PearsonCorrCoef(num_outputs = len(label_features)).to(device)

    # train_per_timestep_r2 = [R2Score(multioutput = "uniform_average").to(device) for _ in range(label_window_length)]
    # val_per_timestep_r2 = [R2Score(multioutput = "uniform_average").to(device) for _ in range(label_window_length)]

    # train_per_feature_r2 = R2Score(multioutput = "raw_values").to(device)
    # val_per_feature_r2 = R2Score(multioutput = "raw_values").to(device)


    # for epoch in range(epochs):
    #     train_loss, train_r2_value, train_ft_r2s, train_ts_r2s, train_feature_pearsons = train(
    #         model = model,
    #         optimizer = optimizer,
    #         criterion = criterion,
    #         r2 = train_r2,
    #         per_timestep_r2 = train_per_timestep_r2,
    #         per_feature_r2 = train_per_feature_r2,
    #         per_feature_pearson = train_per_feature_pearson,
    #         data_loader = data_loader_train,
    #         device = device,
    #         epoch = epoch,
    #         total_epochs = epochs
    #     )

    #     val_loss, val_r2_value, val_ft_r2s, val_ts_r2s, val_feature_pearsons = validate(
    #         model = model,
    #         criterion = criterion,
    #         r2 = val_r2,
    #         per_timestep_r2 = val_per_timestep_r2,
    #         per_feature_r2 = val_per_feature_r2,
    #         per_feature_pearson = val_per_feature_pearson,
    #         data_loader = data_loader_val,
    #         device = device,
    #         epoch = epoch,
    #         total_epochs = epochs
    #     )

    # # if(overfit_monitor.check(epoch = epoch, train_loss = train_loss, val_loss = val_loss)):
    # #     break

    # model_path = f"{ensemble_root_path}/{num_log}/T5-{input_window_length}-{label_window_length}-{window_stride}-{seed_val}.pt"

    # torch.save(model, model_path)

    # with open(f"./ensemble/logs/{num_log}.log", mode = "a") as f:
    #     f.write(f"seed: {seed_val}\n")
    #     f.write(f"Num trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    #     f.write(f"{model_path}\n\n")
    #     f.write(f"Train Loss: {train_loss}, Train R2: {train_r2_value}\n")
    #     f.write(f"Val Loss: {val_loss}, Val R2: {val_r2_value}\n\n")
        
    #     f.write(f"Train Per Feature R2:\n")
    #     for i in range(len(label_features)):
    #         f.write(f"    {label_features[i]}: {train_ft_r2s[i]:.6f}\n")
    #     f.write(f"Val Per Feature R2:\n")
    #     for i in range(len(label_features)):
    #         f.write(f"    {label_features[i]}: {val_ft_r2s[i]:.6f}\n")

    #     f.write("\nTrain Per Feature Pearson:\n")
    #     f.write(f"    {[f'{f_p:.6f}' for f_p in train_feature_pearsons]}\n")
    #     f.write("Val Per Feature Pearson:\n")
    #     f.write(f"    {[f'{f_p:.6f}' for f_p in val_feature_pearsons]}\n")

    #     f.write("\n==========================================\n")

    # with open(f"./ensemble/logs/{num_log}-timestepR2.log", mode = "a") as f:
    #     f.write(f"seed: {seed_val}\n")
    #     f.write("Train R2\n")
    #     f.write(str(train_ts_r2s))
    #     f.write("\n\n")
    #     f.write("Val R2\n")
    #     f.write(str(val_ts_r2s))
    #     f.write("\n\n")
    #     f.write("==========================================\n")


    # del model
    # torch.cuda.empty_cache()

    # print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: {model_path} saved!\n")

toc = datetime.now()
print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: Ensemble training done!")
print(f"Total execution time: {toc - tic}\n")
