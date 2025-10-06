from tqdm import tqdm
import torch



def train(model, optimizer, criterion, r2, data_loader, device, epoch, total_epochs):
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

        num_label_batch_samples, _, num_label_features = batch_y.shape

        # bos = torch.zeros(
        #     num_label_batch_samples, 1, num_label_features,
        #     dtype = torch.float,
        #     device = device
        # )

        bos = model.bos_token.expand(num_label_batch_samples, -1, -1)

        decoder_input = torch.cat([bos, batch_y[:, :-1, :]], dim = 1)    # Shifted right with one bos

        optimizer.zero_grad()
        outputs = model(
            inputs_embeds = model.encoder.embed_tokens(batch_x),
            decoder_inputs_embeds = model.decoder.embed_tokens(decoder_input)
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

    avg_loss = train_loss / num_batches
    avg_r2 = r2.compute()
    r2.reset()

    print(f"Epoch [{epoch + 1}/{total_epochs}], Train Loss: {avg_loss:.6f}, Train R2: {avg_r2:.6f}")
    return avg_loss, avg_r2


def validate(model, criterion, r2, data_loader, device, epoch, total_epochs):
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

            num_label_batch_samples, num_label_timesteps, num_label_features = batch_y.shape    # num_label_features == len(label_features)

            encoder_outputs = model.encoder(
                inputs_embeds = model.encoder.embed_tokens(batch_x)
            )

            # bos = torch.zeros(
            #     num_label_batch_samples, 1, num_label_features,
            #     dtype = torch.float,
            #     device = device
            # )

            bos = model.bos_token.expand(num_label_batch_samples, -1, -1)

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
            val_loss += loss.item()
            num_batches += 1
            r2.update(
                preds.view(preds.shape[0], -1),
                batch_y.view(batch_y.shape[0], -1)
            )
            progress_bar.set_postfix({"val_loss": f"{loss.item():.6f}"})

    avg_loss = val_loss / num_batches
    avg_r2 = r2.compute()
    r2.reset()
    print(f"Epoch [{epoch + 1}/{total_epochs}], Val Loss: {avg_loss:.6f}, Val R2: {avg_r2:.6f}\n")

    return avg_loss, avg_r2