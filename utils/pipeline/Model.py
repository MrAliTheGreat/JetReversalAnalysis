from transformers import BartConfig, BartForConditionalGeneration
import torch

class TimeSeriesHuggingFaceTransformer(BartForConditionalGeneration):
    def __init__(
            self,
            input_dim,
            output_dim,
            d_model,
            num_head,
            num_encoder_layers,
            num_decoder_layers,
            position_wise_ffn_dim,
            dropout,
            positional_encoding_max_len
        ):
        # batch_first = True in all huggingface models
        config = BartConfig(
            vocab_size = 1,                                                        # No vocab --> = 1 is placeholder
            pad_token_id = 0,                                                      # No vocab --> = 0 is placeholder
            d_model = d_model,
            encoder_layers = num_encoder_layers,
            decoder_layers = num_decoder_layers,
            encoder_attention_heads = num_head,
            decoder_attention_heads = num_head,
            d_ff = position_wise_ffn_dim,
            dropout = dropout,
            max_position_embeddings = positional_encoding_max_len,
        )
        
        super().__init__(config)                                                  # Creates model with random weights

        self.model.encoder.embed_tokens = torch.nn.Linear(input_dim, d_model)     # Embedding layer for input
        self.model.decoder.embed_tokens = torch.nn.Linear(output_dim, d_model)    # Embedding layer for output
        
        self.lm_head = torch.nn.Linear(d_model, output_dim)                       # Last linear before output
        
        self.output_dim = output_dim

        # self.bos_token = torch.nn.Parameter(torch.empty(1, 1, output_dim))
        # torch.nn.init.normal_(self.bos_token, mean = 0.0, std = 1.0)

        # Longer training with a single layer! Should TRY this!
        # self.bos_projector = torch.nn.Linear(d_model, output_dim)
        self.bos_projector = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, output_dim)
        )

        self.attention_weights = {
            "encoder_attention": [],
            "decoder_attention": [],
            "cross_attention": []
        }

    def forward(self, inputs_embeds = None, decoder_inputs_embeds = None, encoder_outputs = None, **kwargs):
        if(inputs_embeds is None and encoder_outputs is None):
            raise ValueError("input_embeds and encoder_outputs can not both be None at the same time!")
        
        if(inputs_embeds is not None and encoder_outputs is not None):
            raise ValueError("input_embeds and encoder_outputs can not both have values at the same time")

        outputs = super().forward(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            encoder_outputs = encoder_outputs,
            **kwargs
        )

        if(outputs.encoder_attentions):
            self.attention_weights["encoder_attention"].append(torch.stack(outputs.encoder_attentions, dim = 0).detach().cpu())
            
        if(outputs.decoder_attentions):
            self.attention_weights["decoder_attention"].append(torch.stack(outputs.decoder_attentions, dim = 0).detach().cpu())
            
        if(outputs.cross_attentions):
            # Encoder-Decoder Attention
            self.attention_weights["cross_attention"].append(torch.stack(outputs.cross_attentions, dim = 0).detach().cpu())

        return outputs
    
    def get_average_attention_values(self, attention_type = "cross_attention"):
        '''
            self.attention_weights[attention_type]
                Type: list
                Shape: output_window_num_timesteps * [torch.Size([num_decoder_layers, batch_size = 1, num_heads, single_input_timestep = 1, input_window_num_timesteps])]
            Returns
                Type: torch.tensor
                Shape: (output_window_num_timesteps, input_window_num_timesteps)
        '''

        return torch.stack(self.attention_weights[attention_type], dim = 0).squeeze_(dim = 2).squeeze_(dim = 3).mean(dim = 1).mean(dim = 1).cpu().numpy()
