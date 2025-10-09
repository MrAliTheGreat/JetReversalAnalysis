from tqdm import tqdm
import torch
import numpy as np



def train(model, optimizer, criterion, r2, per_timestep_r2, data_loader, device, epoch, total_epochs):
    '''
        Train for a single epoch
    '''
    
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        data_loader,
        desc = f"Epoch {epoch + 1}/{total_epochs}"
    )

    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        _, num_label_timesteps, _ = batch_y.shape

        # bos = model.bos_token.expand(num_label_batch_samples, -1, -1)

        encoder_outputs = model.encoder(
            inputs_embeds = model.encoder.embed_tokens(batch_x),
            return_dict = True
        )

        final_encoder_state = encoder_outputs.last_hidden_state[:, -1:, :]
        bos = model.bos_projector(final_encoder_state)

        decoder_input = torch.cat([bos, batch_y[:, :-1, :]], dim = 1)    # Shift right with one bos

        optimizer.zero_grad()
        outputs = model(
            encoder_outputs = encoder_outputs,
            decoder_inputs_embeds = model.decoder.embed_tokens(decoder_input),
        )

        loss = criterion(outputs.logits, batch_y)   # logits is preds
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1
        r2.update(
            outputs.logits.view(outputs.logits.shape[0], -1),
            batch_y.view(batch_y.shape[0], -1)
        )
        progress_bar.set_postfix({"train_loss": f"{loss.item():.6f}"})
        for t in range(num_label_timesteps):
            per_timestep_r2[t].update(
                outputs.logits[:, t, :],
                batch_y[:, t, :]
            )

    avg_loss = train_loss / num_batches
    avg_r2 = r2.compute().item()
    r2.reset()

    timestep_r2s = np.array([r2.compute().item() for r2 in per_timestep_r2])
    for r2 in per_timestep_r2:
        r2.reset()

    print(f"Epoch [{epoch + 1}/{total_epochs}], Train Loss: {avg_loss:.6f}, Train R2: {avg_r2:.6f}")

    sorted_idxs = np.argsort(timestep_r2s)
    print("\nWorst 5 Time-Steps Train R2:")
    for idx in sorted_idxs[:5]:
        print(f"    Time-step {idx + 1}: R2 = {timestep_r2s[idx]:.6f}")

    print("Best 5 Time-Steps Train R2:")
    for idx in sorted_idxs[-5:][::-1]:
        print(f"    Time-step {idx + 1}: R2 = {timestep_r2s[idx]:.6f}")

    print()

    return avg_loss, avg_r2


def autoregress(model, batch_x, batch_y, device, extract_attention = False):
    if(extract_attention):
        model.attention_weights = {
            "encoder_attention": [],
            "decoder_attention": [],
            "cross_attention": []
        }

    num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape    # num_label_features == len(label_features)

    # decoder_single_timestep_input = model.bos_token.expand(num_label_batch_samples, -1, -1)

    preds = torch.zeros(
        num_label_batch_samples, num_label_timesteps, num_label_features,
        dtype = torch.float,
        device = device
    )

    encoder_outputs = model.encoder(
        inputs_embeds = model.encoder.embed_tokens(batch_x),
        return_dict = True
    )

    final_encoder_state = encoder_outputs.last_hidden_state[:, -1:, :]
    decoder_single_timestep_input = model.bos_projector(final_encoder_state)

    past_key_values = None

    for i in range(num_label_timesteps):
        outputs = model(
            encoder_outputs = encoder_outputs,
            decoder_inputs_embeds = model.decoder.embed_tokens(decoder_single_timestep_input),
            past_key_values = past_key_values,
            use_cache = True,
            output_attentions = extract_attention,
            output_hidden_states = True,
            return_dict = True
        )

        decoder_last_hidden_state = outputs.decoder_hidden_states[-1][:, -1:, :]
        next_prediction = model.lm_head(decoder_last_hidden_state)    # Shape: (batch_size, 1, num_label_features)

        preds[:, i, :] = next_prediction.squeeze(1)

        # KV Caching
        past_key_values = outputs.past_key_values

        decoder_single_timestep_input = next_prediction

    return preds


def validate(model, criterion, r2, per_timestep_r2, data_loader, device, epoch, total_epochs):
    '''
        Validate for a single epoch
    '''

    model.eval()
    val_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(
        data_loader,
        desc = f"Epoch {epoch + 1}/{total_epochs}"
    )

    with torch.no_grad():
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, num_label_timesteps, _ = batch_y.shape

            preds = autoregress(
                model = model,
                batch_x = batch_x,
                batch_y = batch_y,
                device = device
            )

            loss = criterion(preds, batch_y)
            val_loss += loss.item()
            num_batches += 1
            r2.update(
                preds.view(preds.shape[0], -1),
                batch_y.view(batch_y.shape[0], -1)
            )
            progress_bar.set_postfix({"val_loss": f"{loss.item():.6f}"})
            for t in range(num_label_timesteps):
                per_timestep_r2[t].update(
                    preds[:, t, :],
                    batch_y[:, t, :]
                )

    avg_loss = val_loss / num_batches
    avg_r2 = r2.compute().item()
    r2.reset()

    timestep_r2s = np.array([r2.compute().item() for r2 in per_timestep_r2])
    for r2 in per_timestep_r2:
        r2.reset()

    print(f"Epoch [{epoch + 1}/{total_epochs}], Val Loss: {avg_loss:.6f}, Val R2: {avg_r2:.6f}")

    sorted_idxs = np.argsort(timestep_r2s)
    print("\nWorst 5 Time-Steps Val R2:")
    for idx in sorted_idxs[:5]:
        print(f"    Time-step {idx + 1}: R2 = {timestep_r2s[idx]:.6f}")

    print("Best 5 Time-Steps Val R2:")
    for idx in sorted_idxs[-5:][::-1]:
        print(f"    Time-step {idx + 1}: R2 = {timestep_r2s[idx]:.6f}")

    print("\n-----------------------------------------------------------------\n")

    return avg_loss, avg_r2

