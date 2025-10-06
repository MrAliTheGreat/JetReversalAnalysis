#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import json, gc
from tqdm.auto import tqdm


# In[2]:


with open("./params.json", mode = "r", encoding = "utf-8") as f:
    data = json.load(f)
    model_path = data["model_path"]
    dataset_path = data["dataset_path"]
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


# In[ ]:


df = pl.read_csv(dataset_path)

df = df.drop(["id", "eps", "n_0_squared"])

df = df.select(
    pl.col("*").str.json_decode()
)

df = df.with_columns(
    eta_list = pl.col("eta_list").list.eval(pl.element().flatten(), parallel = True)
)

df = df.sample(n = len(df), with_replacement = False, shuffle = True, seed = seed_val)

df


# In[5]:


df_train = df[:int(len(df) * 0.7)]
df_val = df[int(len(df) * 0.7):int(len(df) * 0.9)]
df_test = df[int(len(df) * 0.9):]

print(f"df_train shape: {df_train.shape}")
print(f"df_val shape: {df_val.shape}")
print(f"df_test shape: {df_test.shape}")


# In[6]:


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


# In[7]:


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


# In[8]:


from transformers import T5Config, T5ForConditionalGeneration

class TimeSeriesHuggingFaceTransformer(T5ForConditionalGeneration):
    def __init__(self, input_dim, output_dim, d_model, num_head, num_encoder_layers, num_decoder_layers, position_wise_ffn_dim, dropout):
        # batch_first = True in all huggingface models
        config = T5Config(
            vocab_size = 1, # No vocab --> = 1 is placeholder
            d_model = d_model,
            num_heads = num_head,
            num_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            d_ff = position_wise_ffn_dim,
            dropout = dropout,
            decoder_start_token_id = 0,
            tie_word_embeddings = False,
            relative_attention_num_buckets = 32, 
            d_kv = d_model // num_head
        )
        
        super().__init__(config)    # Creates model with random weights

        self.encoder.embed_tokens = torch.nn.Linear(input_dim, d_model)     # Embedding layer for input
        self.decoder.embed_tokens = torch.nn.Linear(output_dim, d_model)    # Embedding layer for output
        
        self.lm_head = torch.nn.Linear(d_model, output_dim, bias = False)   # Last linear before output
        
        self.output_dim = output_dim

    def forward(self, inputs_embeds, decoder_inputs_embeds, **kwargs):
        outputs = super().forward(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            **kwargs
        )
        return outputs


# In[9]:


model = TimeSeriesHuggingFaceTransformer(
    input_dim = len(input_features),
    output_dim = len(label_features),
    d_model = embedding_dim,
    num_head = num_attention_head,
    num_encoder_layers = num_encoder_layers,
    num_decoder_layers = num_encoder_layers,
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

for epoch in range(epochs):
    ################################################## TRAINING ##################################################
    model.train()
    epoch_train_loss = 0.0
    train_progress_bar = tqdm(
        data_loader_train,
        desc = f"Epoch {epoch + 1}/{epochs}"
    )

    for batch_x, batch_y in train_progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(
            inputs_embeds = model.encoder.embed_tokens(batch_x),
            decoder_inputs_embeds = model.decoder.embed_tokens(batch_y)
        )

        loss = criterion(outputs.logits, batch_y)   # logits is preds
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        train_r2.update(
            outputs.logits.view(outputs.logits.shape[0], -1),
            batch_y.view(batch_y.shape[0], -1)
        )
        train_progress_bar.set_postfix({"train_loss": f"{loss.item():.6f}"})

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
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape    # num_label_features == len(label_features)

            encoder_outputs = model.encoder(
                inputs_embeds = model.encoder.embed_tokens(batch_x)
            )

            bos = torch.zeros(
                num_label_batch_samples, 1, num_label_features,
                dtype = torch.float,
                device = device
            )
            
            preds = torch.zeros(
                num_label_batch_samples, num_label_timesteps, num_label_features,
                dtype = torch.float,
                device = device
            )
            
            ################### Analyze This ########################
            # This is where KV caching is critical for speed.
            past_key_values = None

            for i in range(num_label_timesteps):
                # 4. Pass the current decoder input to the decoder
                # Use KV caching to only compute attention for the new token
                decoder_outputs = model.decoder(
                    inputs_embeds = model.decoder.embed_tokens(bos),
                    encoder_hidden_states = encoder_outputs.last_hidden_state,
                    past_key_values = past_key_values,
                    use_cache = True,
                    return_dict = True
                )
                
                # 5. Extract the output for the *last* token
                # This is the new prediction
                decoder_last_hidden_state = decoder_outputs.last_hidden_state[:, -1:, :]

                # 6. Apply the final linear layer (lm_head) to get the prediction
                next_prediction = model.lm_head(decoder_last_hidden_state) # Shape: (batch_size, 1, num_label_features)

                preds[:, i, :] = next_prediction.squeeze(1)

                # 8. Update past_key_values for the next iteration
                # This is the core of KV caching
                past_key_values = decoder_outputs.past_key_values

                # 9. The prediction for the current step becomes the input for the next step
                bos = next_prediction
            ################### Analyze This ########################


            loss = criterion(preds, batch_y)
            epoch_val_loss += loss.item()
            val_r2.update(
                preds.view(preds.shape[0], -1),
                batch_y.view(batch_y.shape[0], -1)
            )
            val_progress_bar.set_postfix({"val_loss": f"{loss.item():.6f}"})

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





# In[ ]:





# In[ ]:




