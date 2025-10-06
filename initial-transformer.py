#!/usr/bin/env python
# coding: utf-8

# **With VS Code Jupyter you don't need to add the kernel to Jupyter Notebook like before to have it as an environment**

# In[1]:


import polars as pl
import json, gc
from tqdm.auto import tqdm


# In[2]:


with open("./params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    model_path = data["model_path"]
    num_single_sample_timesteps = data["num_single_sample_timesteps"]
    window_stride = data["window_stride"]
    input_window_length = data["input_window_length"]
    label_window_length = data["label_window_length"]
    input_features = data["input_features"]
    label_features = data["label_features"]
    positional_encoding_max_len = data["positional_encoding_max_len"]
    embedding_dim = data["embedding_dim"]
    num_attention_head = data["num_attention_head"]
    num_encoder_layers = data["num_encoder_layers"]
    num_decoder_layers = data["num_decoder_layers"]
    position_wise_nn_dim = data["position_wise_nn_dim"]
    dropout = data["dropout"]
    batch_size = data["batch_size"]
    epochs = data["epochs"]
    learning_rate = data["learning_rate"]


# In[3]:


df = pl.read_csv("./reversalData_minor.csv")
df


# In[4]:


df.shape


# In[5]:


df = df.drop(["id", "eps", "n_0_squared"])
df.head()


# In[6]:


df = df.select(
    pl.col("*").str.json_decode()
)
df.head()


# In[7]:


df = df.with_columns(
    eta_list = pl.col("eta_list").list.eval(pl.element().flatten(), parallel = True)
)
df.head()


# In[8]:


import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.regression import R2Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_val = 0

torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)


# In[9]:


df = df.sample(n = len(df), with_replacement = False, shuffle = True, seed = seed_val)
df.head()


# ### Train-Val-Test Split (70-20-10)
# 
# 1. Split based on 1500 time-series samples (Vertically)
# 2. Split each sample to train-val-test (Horizontally)
# 
# Vertically for now!
# 
# #### How about 5-Fold Cross-Validation? Stratified 5F?
# 
# Just a single split for now!

# In[10]:


df_train = df[:int(len(df) * 0.7)]
df_val = df[int(len(df) * 0.7):int(len(df) * 0.9)]
df_test = df[int(len(df) * 0.9):]


# In[11]:


print(f"df_train shape: {df_train.shape}")
print(f"df_val shape: {df_val.shape}")
print(f"df_test shape: {df_test.shape}")


# ## On-Demand Data Loading (No RAM Issue)

# In[12]:


class WindowedDataset(Dataset):
    def __init__(self, input_df, label_df, num_single_sample_timesteps, stride, input_window_length, label_window_length):
        super().__init__()
        
        self.input_df = input_df  # Type: Numpy, Shape: Number of time-series, Number of time-steps, Number of input features
        self.label_df = label_df  # Type: Numpy, Shape: Number of time-series, Number of time-steps, Number of label features
        self.num_single_sample_timesteps = num_single_sample_timesteps
        self.stride = stride
        self.input_window_length = input_window_length
        self.label_window_length = label_window_length

        self.valid_length = self.input_window_length + self.label_window_length
        
        self.window_indices = []
        for time_series_idx in range(self.input_df.shape[0]):
            for input_window_start_idx in range(0, self.input_df.shape[1] - self.valid_length + 1, self.stride):
                self.window_indices.append((time_series_idx, input_window_start_idx))

    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, index):
        time_series_idx, input_window_start_idx = self.window_indices[index]

        label_window_start_idx = input_window_start_idx + self.input_window_length
        input_window = self.input_df[time_series_idx, input_window_start_idx: label_window_start_idx, :]
        label_window = self.label_df[time_series_idx, label_window_start_idx: label_window_start_idx + self.label_window_length, :]

        input_window_mean = input_window.mean(axis = 0)
        input_window_std = input_window.std(axis = 0)
        input_window_std[input_window_std == 0] = 10 ** -8
        input_window = (input_window - input_window_mean) / input_window_std

        return torch.tensor(input_window, dtype = torch.float), torch.tensor(label_window, dtype = torch.float)


# In[13]:


# df_train.explode("*")

# import torch
# from torch.utils.data import Dataset
# import polars as pl
# import numpy as np


# class CausallyNormalizedTimeSeriesDataset(Dataset):
#     def __init__(self, csv_path, input_len=200, label_len=100, stride=1):
#         self.input_len = input_len
#         self.label_len = label_len
#         self.total_len = input_len + label_len
#         self.stride = stride

#         # Load dataset from CSV (assume shape: [num_samples, time_steps * features])
#         df = pl.read_csv(csv_path)
#         raw_np = df.to_numpy()

#         # Infer shape
#         num_samples, total_cols = raw_np.shape
#         if total_cols % 1000 != 0:
#             raise ValueError("Expected time series length of 1000 per sample")
#         self.sequence_len = 1000
#         self.num_features = total_cols // self.sequence_len

#         # Reshape to (N, T, F)
#         self.data = raw_np.reshape(num_samples, self.sequence_len, self.num_features)
#         self.num_samples = num_samples

#         assert self.sequence_len >= self.total_len, "Time series too short for given input + label length"

#         # Compute mean/std using only the first input_len steps of each event
#         first_input = self.data[:, :self.input_len, :]  # shape: (N, input_len, F)
#         self.row_means = first_input.mean(axis=1, keepdims=True)  # shape: (N, 1, F)
#         self.row_stds = first_input.std(axis=1, keepdims=True) + 1e-8

#         # Precompute valid window positions
#         self.window_indices = []
#         max_start = self.sequence_len - self.total_len
#         for sample_idx in range(self.num_samples):
#             for t_start in range(0, max_start + 1, self.stride):
#                 self.window_indices.append((sample_idx, t_start))

#     def __len__(self):
#         return len(self.window_indices)

#     def __getitem__(self, idx):
#         sample_idx, t_start = self.window_indices[idx]

#         full_sequence = self.data[sample_idx]  # (T, F)
#         mean = self.row_means[sample_idx]      # (1, F)
#         std = self.row_stds[sample_idx]        # (1, F)

#         norm_sequence = (full_sequence - mean) / std  # (T, F)

#         x = norm_sequence[t_start: t_start + self.input_len]  # (input_len, F)
#         y = norm_sequence[t_start + self.input_len: t_start + self.total_len]  # (label_len, F)

#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# dataset = CausallyNormalizedTimeSeriesDataset(
#     csv_path="your_multivariate_dataset.csv",
#     input_len=200,
#     label_len=100,
#     stride=1
# )

# loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# for x, y in loader:
#     print("Input shape:", x.shape)   # (32, 200, F)
#     print("Label shape:", y.shape)   # (32, 100, F)
#     break


# ## All-in-one-go Data Windowing (Possible RAM overflow)
# 
# Must be super modular since we will run with diverse set of windows!
# 
# There is a helper function in Keras but I'm going full PyTorch! (`timeseries_dataset_from_array`)

# ### Normalization
# 
# Important! Moving average must be implemented!
# 
# During training you shouldn't have access to future tokens for normalization! --> Data Leakage
# 
# For now just z-score on the whole train set for each row separately

# In[14]:


# # All-in-one-go Data Windowing
#
# df_train = df_train.select(
#     (pl.col("*") - pl.col("*").list.mean()) / pl.col("*").list.std()
# )

# df_val = df_val.select(
#     (pl.col("*") - pl.col("*").list.mean()) / pl.col("*").list.std()
# )

# df_test = df_test.select(
#     (pl.col("*") - pl.col("*").list.mean()) / pl.col("*").list.std()
# )


# What I am doing above is global normalization where I calculate mean and std across all time-series. There are other options:
# 
# 1. Calculate mean and std per series (num_single_sample_timesteps) and apply to windows --> Temporal leakage! --> Knowing about future distribution
# 2. Calculate mean and std per window (input_window_length) and apply to each window --> Locally aware might lose global context?
# 3. Calculate mean and std based on pervious steps of the current window, normalize the current window with that mean and std
# 
# I am implementing step 2 for now in the custom dataset!

# Probably a violin plot like the one here (https://www.tensorflow.org/tutorials/structured_data/time_series#normalize_the_data) is a good idea!

# #### Reading the dataset whole!

# In[15]:


# All-in-one-go Data Windowing
#
# def createWindowedDataframe(dataframe, stride, input_features, input_window_length, label_features, label_window_length):
#     """
#         Creates a windowed time-series dataset suitable for neural network (transformer) input

#         dataframe: In time-series format where columns are features and rows are time-series steps (Polars exploded format!)
#         stride: How many timesteps will each window move by. Equal for both input and label windows
#         input_features: Names of the input features in a list
#         input_window_length: Input window size in timesteps
#         label_features: Names of the label features in a list
#         label_window_length: label window size in timesteps
#     """

#     input_df = dataframe.select(
#         pl.col(input_features)
#     )
#     input_df = input_df.select(
#         numbers = pl.concat_list("*")   # List of all features
#     )
#     input_df = input_df[:-label_window_length].with_row_index("index").with_columns(    # Exluding the ending label window
#         pl.col("index").cast(pl.Int64)
#     )
#     input_df = input_df.group_by_dynamic(   # index is considered for grouping
#         index_column = "index",
#         period = f"{input_window_length}i",
#         every = f"{stride}i",
#         closed = "left"
#     ).agg(
#         pl.col("numbers").alias("X"),
#         pl.len().alias("seq_len")
#     )
#     input_df = input_df.filter(
#         pl.col("seq_len") == input_window_length
#     )
#     input_df = input_df.select(
#         pl.exclude(["index", "seq_len"])
#     )


#     label_df = dataframe.select(
#         pl.col(label_features)
#     )
#     label_df = label_df.select(
#         numbers = pl.concat_list("*")
#     )
#     label_df = label_df[input_window_length:].with_row_index("index").with_columns(
#         pl.col("index").cast(pl.Int64)
#     )
#     label_df = label_df.group_by_dynamic(
#         index_column = "index",
#         period = f"{label_window_length}i",
#         every = f"{stride}i",
#         closed = "left"
#     ).agg(
#         pl.col("numbers").alias("Y"),
#         pl.len().alias("seq_len")
#     )
#     label_df = label_df.filter(
#         (pl.col("seq_len") == label_window_length)
#     )
#     label_df = label_df.select(
#         pl.exclude(["index", "seq_len"])
#     )

#     return pl.concat([input_df, label_df], how = "horizontal")    


# In[16]:


# All-in-one-go Data Windowing
#
# df_train = df_train.explode("*")

# timeseries_df_train = pl.DataFrame()

# for i in range(0, len(df_train), num_single_sample_timesteps):
#     temp_df = createWindowedDataframe(
#         dataframe = df_train[i:i + num_single_sample_timesteps],
#         stride = window_stride,
#         input_features = input_features,
#         input_window_length = input_window_length,
#         label_features = label_features,
#         label_window_length = label_window_length
#     )

#     if(temp_df.is_empty()):
#         timeseries_df_train = temp_df
#     else:
#         timeseries_df_train = pl.concat([timeseries_df_train, temp_df], how = "vertical")

# timeseries_df_train


# In[17]:


# All-in-one-go Data Windowing
#
# df_val = df_val.explode("*")

# timeseries_df_val = pl.DataFrame()

# for i in range(0, len(df_val), num_single_sample_timesteps):
#     temp_df = createWindowedDataframe(
#         dataframe = df_val[i:i + num_single_sample_timesteps],
#         stride = window_stride,
#         input_features = input_features,
#         input_window_length = input_window_length,
#         label_features = label_features,
#         label_window_length = label_window_length
#     )

#     if(temp_df.is_empty()):
#         timeseries_df_val = temp_df
#     else:
#         timeseries_df_val = pl.concat([timeseries_df_val, temp_df], how = "vertical")

# timeseries_df_val


# In[18]:


# All-in-one-go Data Windowing
#
# df_test = df_test.explode("*")

# timeseries_df_test = pl.DataFrame()

# for i in range(0, len(df_test), num_single_sample_timesteps):
#     temp_df = createWindowedDataframe(
#         dataframe = df_test[i:i + num_single_sample_timesteps],
#         stride = window_stride,
#         input_features = input_features,
#         input_window_length = input_window_length,
#         label_features = label_features,
#         label_window_length = label_window_length
#     )

#     if(temp_df.is_empty()):
#         timeseries_df_test = temp_df
#     else:
#         timeseries_df_test = pl.concat([timeseries_df_test, temp_df], how = "vertical")

# timeseries_df_test


# **Some visualization here is good to do a sanity-check on the dataset before sending it to the model!!!!!!!!!!**

# In[19]:


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# In[20]:


class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_head, num_encoder_layers, num_decoder_layers, positional_encoding_max_len, position_wise_ffn_dim, dropout):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.output_proj = torch.nn.Linear(output_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(
            d_model = d_model,
            dropout = dropout,
            max_len = positional_encoding_max_len
        )

        self.transformer = torch.nn.Transformer(
            d_model = d_model,
            nhead = num_head,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout,
            dim_feedforward = position_wise_ffn_dim,
            batch_first = True
        )

        self.final_proj = torch.nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, tgt_mask = None):
        # encode and decode methods can be used here

        src = self.input_proj(src)
        src = self.pos_encoder(src)
        
        tgt = self.output_proj(tgt)
        tgt = self.pos_encoder(tgt)
        
        if(tgt_mask is None):
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        decoder_out = self.transformer(src, tgt, tgt_mask = tgt_mask)

        return self.final_proj(decoder_out)
    
    def encode(self, src):
        # memory is encoder output

        src = self.input_proj(src)
        src = self.pos_encoder(src)
        memory = self.transformer.encoder(src)
        return memory

    def decode(self, tgt, memory, tgt_mask = None):
        tgt = self.output_proj(tgt)
        tgt = self.pos_encoder(tgt)

        if(tgt_mask is None):
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        decoder_out = self.transformer.decoder(tgt, memory, tgt_mask = tgt_mask)

        return self.final_proj(decoder_out)


# In[21]:


# All-in-one-go Data Windowing

# df_train = timeseries_df_train
# df_val = timeseries_df_val
# df_test = timeseries_df_test

# # %reset_selective -f "^timeseries_df_train$"
# # %reset_selective -f "^timeseries_df_val$"
# # %reset_selective -f "^timeseries_df_test$"
# # %reset_selective -f "^df$"
# del timeseries_df_train
# del timeseries_df_val
# del timeseries_df_test
# del df
# gc.collect()


# In[22]:


# # All-in-one-go Data Windowing
# 
# df_train = TensorDataset(
#     torch.Tensor(df_train["X"]),
#     torch.Tensor(df_train["Y"])
# )
# data_loader_train = DataLoader(
#     df_train,
#     batch_size = batch_size,
#     shuffle = True,
#     num_workers = 10
# )

# df_val = TensorDataset(
#     torch.Tensor(df_val["X"]),
#     torch.Tensor(df_val["Y"])
# )
# data_loader_val = DataLoader(
#     df_val,
#     batch_size = batch_size,
#     shuffle = True,
#     num_workers = 10
# )

# df_test = TensorDataset(
#     torch.Tensor(df_test["X"]),
#     torch.Tensor(df_test["Y"])
# )
# data_loader_test = DataLoader(
#     df_test,
#     batch_size = batch_size,
#     shuffle = True,
#     num_workers = 10
# )


# In[23]:


##### TRAIN #####

num_train_samples = df_train.shape[0]

input_df = df_train.select(
    pl.col(input_features)
)
label_df = df_train.select(
    pl.col(label_features)
)
input_df = input_df.explode("*").to_numpy()
input_df = input_df.reshape(num_train_samples, num_single_sample_timesteps, len(input_features))
label_df = label_df.explode("*").to_numpy()
label_df = label_df.reshape(num_train_samples, num_single_sample_timesteps, len(label_features))

df_train = WindowedDataset(
    input_df = input_df,
    label_df = label_df,
    num_single_sample_timesteps = num_single_sample_timesteps,
    stride = window_stride,
    input_window_length = input_window_length,
    label_window_length = label_window_length
)

data_loader_train = DataLoader(
    df_train,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 10,
    prefetch_factor = 8,
    persistent_workers = True,
    pin_memory = True
)



##### VALIDATION #####
num_val_samples = df_val.shape[0]

input_df = df_val.select(
    pl.col(input_features)
)
label_df = df_val.select(
    pl.col(label_features)
)
input_df = input_df.explode("*").to_numpy()
input_df = input_df.reshape(num_val_samples, num_single_sample_timesteps, len(input_features))
label_df = label_df.explode("*").to_numpy()
label_df = label_df.reshape(num_val_samples, num_single_sample_timesteps, len(label_features))

df_val = WindowedDataset(
    input_df = input_df,
    label_df = label_df,
    num_single_sample_timesteps = num_single_sample_timesteps,
    stride = window_stride,
    input_window_length = input_window_length,
    label_window_length = label_window_length
)

data_loader_val = DataLoader(
    df_val,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 10,
    prefetch_factor = 8,
    persistent_workers = True,
    pin_memory = True
)



##### TEST #####
num_test_samples = df_test.shape[0]

input_df = df_test.select(
    pl.col(input_features)
)
label_df = df_test.select(
    pl.col(label_features)
)
input_df = input_df.explode("*").to_numpy()
input_df = input_df.reshape(num_test_samples, num_single_sample_timesteps, len(input_features))
label_df = label_df.explode("*").to_numpy()
label_df = label_df.reshape(num_test_samples, num_single_sample_timesteps, len(label_features))

df_test = WindowedDataset(
    input_df = input_df,
    label_df = label_df,
    num_single_sample_timesteps = num_single_sample_timesteps,
    stride = window_stride,
    input_window_length = input_window_length,
    label_window_length = label_window_length
)

data_loader_test = DataLoader(
    df_test,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 10,
    prefetch_factor = 8,
    persistent_workers = True,
    pin_memory = True
)


##### CLEANING UP! #####

del input_df
del label_df
del df
gc.collect()


# In[ ]:


model = TimeSeriesTransformer(
    input_dim = len(input_features),
    output_dim = len(label_features),
    d_model = embedding_dim,
    num_head = num_attention_head,
    num_encoder_layers = num_encoder_layers,
    num_decoder_layers = num_encoder_layers,
    positional_encoding_max_len = positional_encoding_max_len,
    position_wise_ffn_dim = position_wise_nn_dim,
    dropout = dropout
).to(device)

print(f"Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate
)

train_r2 = R2Score(multioutput = "uniform_average").to(device)
val_r2 = R2Score(multioutput = "uniform_average").to(device)

overfit_count = 0

model.forward = torch.compile(model.forward)    # Faster for training

for epoch in range(epochs):
    ################################################## TRAINING ##################################################
    ##### No Scheduled Sampling #####
    # model.train()
    # epoch_train_loss = 0.0
    # train_progress_bar = tqdm(
    #     data_loader_train,
    #     desc = f"Epoch {epoch + 1}/{epochs}"
    # )
    
    # for batch_x, batch_y in train_progress_bar:
    #     batch_x = batch_x.to(device)
    #     batch_y = batch_y.to(device)

    #     decoder_input = torch.zeros_like(batch_y)
    #     decoder_input[:, 1:] = batch_y[:, :-1]
    #     decoder_input[:, 0] = 0    # Adding bos token

    #     optimizer.zero_grad()
    #     output = model(batch_x, decoder_input)
    #     loss = criterion(output, batch_y)
    #     loss.backward()
    #     optimizer.step()

    #     epoch_train_loss += loss.item()
    #     train_r2.update(
    #         output.view(output.shape[0], -1),    # Flatten (batch_size, timestep * num_feature)
    #         batch_y.view(batch_y.shape[0], -1)
    #     )
    #     train_progress_bar.set_postfix({
    #         "train_loss": f"{loss.item():.6f}"
    #     })

    # avg_train_loss = epoch_train_loss / len(data_loader_train)
    # epoch_train_r2 = train_r2.compute()
    # train_r2.reset()
    # print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Train R2: {epoch_train_r2:.6f}")
    #################################

    model.train()
    epoch_train_loss = 0.0
    train_progress_bar = tqdm(
        data_loader_train,
        desc = f"Epoch {epoch + 1}/{epochs}"
    )

    epsilon = max(0, 1.0 - 10 * epoch / epochs)  # Linear decay for choosing between label or pred as decoder input

    for batch_x, batch_y in train_progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape

        # First pass: Just get preds --> No grads!
        decoder_input_first_pass = torch.zeros_like(batch_y)
        decoder_input_first_pass[:, 1:, :] = batch_y[:, :-1, :] # [:, 0, :] BOS = 0
        
        with torch.no_grad():
            output_first_pass = model(batch_x, decoder_input_first_pass)
            output_first_pass = output_first_pass.detach()   # detach: Not involved in model computation graph

        # Second pass: pred and label decision for decoder input and back prop
        decoder_input_second_pass = torch.zeros_like(batch_y)   # [;, 0, :] BOS = 0
        teacher_force_mask = (
            torch.rand(num_label_batch_samples, num_label_timesteps - 1, device = device) < epsilon    # Not including BOS
        ).unsqueeze(2)  # Boolean mat of size: (num_label_batch_samples, num_label_timesteps, 1) --> Bool for selecting either pred or label
        decoder_input_second_pass[:, 1:, :] = torch.where(
            teacher_force_mask,
            batch_y[:, :-1, :],
            output_first_pass[:, :-1, :]
        )
        optimizer.zero_grad()
        output_second_pass = model(batch_x, decoder_input_second_pass)
        loss = criterion(output_second_pass, batch_y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        train_r2.update(
            output_second_pass.view(output_second_pass.shape[0], -1),    # Flatten (batch_size, timestep * num_feature)
            batch_y.view(batch_y.shape[0], -1)
        )
        train_progress_bar.set_postfix({
            "train_loss": f"{loss.item():.6f}"
        })

    avg_train_loss = epoch_train_loss / len(data_loader_train)
    epoch_train_r2 = train_r2.compute()
    train_r2.reset()
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Train R2: {epoch_train_r2:.6f}")

    ################################################# VALIDATION #################################################
    model.eval()
    epoch_val_loss = 0.0
    val_progress_bar = tqdm(
        data_loader_val,
        desc = f"Epoch {epoch + 1}/{epochs}"
    )

    with torch.no_grad():
        for batch_x, batch_y in val_progress_bar:
            # No teacher forcing in inference: Output at each timestep is feeded back to the decoder as input in the next timestep
            # Attention mask grows as timesteps pass
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape    # num_label_features == len(label_features)

            outputs = torch.zeros(num_label_batch_samples, num_label_timesteps, num_label_features).to(device)
            decoder_input = torch.zeros(num_label_batch_samples, 1 + num_label_timesteps, num_label_features).to(device)    # +1 for BOS

            full_mask = model.transformer.generate_square_subsequent_mask(1 + num_label_timesteps).to(device)
            encoder_output = model.encode(batch_x)

            for t in range(num_label_timesteps):
                # tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device) --> Alternative to full mask! (Indexing is faster than mask generation!)
                tgt_mask = full_mask[:t+1, :t+1]
                out = model.decode(
                    tgt = decoder_input[:, :t+1, :],
                    memory = encoder_output,
                    tgt_mask = tgt_mask
                )
                next_step = out[:, -1, :]
                outputs[:, t, :] = next_step
                decoder_input[:, t+1, :] = next_step

            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item()
            val_r2.update(
                outputs.view(outputs.shape[0], -1),    # Flatten (batch_size, timestep * num_feature)
                batch_y.view(batch_y.shape[0], -1)
            )
            val_progress_bar.set_postfix({
                "val_loss": f"{loss.item():.6f}"
            })

        avg_val_loss = epoch_val_loss / len(data_loader_val)
        epoch_val_r2 = val_r2.compute()
        val_r2.reset()
        print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {avg_val_loss:.6f}, Val R2: {epoch_val_r2:.6f}\n")
    

    if(epoch >= 10 and avg_val_loss - avg_train_loss > 0.01):
        overfit_count += 1
        print(f"Possible Overfitting!!! {overfit_count}/3\n")
        if(overfit_count == 3):
            print("Training Stopped!!!")
            break


# In[ ]:


################################################## TESTING ##################################################

model.eval()
test_loss = 0.0
test_progress_bar = tqdm(
    data_loader_test
)

test_r2 = R2Score(multioutput = "uniform_average")

with torch.no_grad():
    for batch_x, batch_y in test_progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape    # num_label_features == len(label_features)

        outputs = torch.zeros(num_label_batch_samples, num_label_timesteps, num_label_features).to(device)
        decoder_input = torch.zeros(num_label_batch_samples, 1 + num_label_timesteps, num_label_features).to(device)    # +1 for BOS

        full_mask = model.transformer.generate_square_subsequent_mask(1 + num_label_timesteps).to(device)
        encoder_output = model.encode(batch_x)

        for t in range(num_label_timesteps):
            tgt_mask = full_mask[:t+1, :t+1]
            out = model.decode(
                tgt = decoder_input[:, :t+1, :],
                memory = encoder_output,
                tgt_mask = tgt_mask
            )
            next_step = out[:, -1, :]
            outputs[:, t, :] = next_step
            decoder_input[:, t+1, :] = next_step

        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        test_progress_bar.set_postfix({
            "batch_test_loss": f"{loss.item():.6f}"
        })
        test_r2.update(
            outputs.view(outputs.shape[0], -1),
            batch_y.view(batch_y.shape[0], -1)
        )

    final_test_loss = test_loss / len(data_loader_test)
    final_test_r2 = test_r2.compute()
    test_r2.reset()
    print(f"Final Test Loss: {final_test_loss:.6f}, Final Test R2: {final_test_r2:.6f}")


# In[ ]:


torch.save(model, model_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




