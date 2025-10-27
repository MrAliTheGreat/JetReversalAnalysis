from transformers import T5Config, T5ForConditionalGeneration
import torch, math

class TimeSeriesHuggingFaceTransformer(T5ForConditionalGeneration):
    def __init__(
        self,
        input_window_len,
        output_window_len,
        input_dim,
        output_dim,
        d_model,
        num_head,
        num_encoder_layers,
        num_decoder_layers,
        position_wise_ffn_dim,
        relative_attention_num_buckets,
        dropout
    ):
        # batch_first = True in all huggingface models
        config = T5Config(
            vocab_size = 1,                                                 # No vocab --> = 1 is placeholder
            d_model = d_model,
            num_heads = num_head,
            num_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            d_ff = position_wise_ffn_dim,
            dropout = dropout,
            decoder_start_token_id = 0,
            tie_word_embeddings = False,
            relative_attention_num_buckets = relative_attention_num_buckets, 
            d_kv = d_model // num_head,
        )
        
        super().__init__(config)                                            # Creates model with random weights

        self.output_dim = output_dim

        self.encoder.embed_tokens = torch.nn.Linear(input_dim, d_model)     # Embedding layer for input
        self.decoder.embed_tokens = torch.nn.Linear(output_dim, d_model)    # Embedding layer for output
        
        self.lm_head = torch.nn.Linear(d_model, output_dim * 2, bias = False)   # Last linear before output

        # self.bos_token = torch.nn.Parameter(torch.empty(1, 1, output_dim))
        # torch.nn.init.normal_(self.bos_token, mean = 0.0, std = 1.0)

        # self.bos_projector = torch.nn.Linear(d_model, output_dim)
        self.bos_projector = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, output_dim)
        )

        input_pos = torch.arange(input_window_len, dtype = torch.float32).unsqueeze(1)
        output_pos = torch.arange(output_window_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        input_pe = torch.zeros(input_window_len, d_model)
        input_pe[:, 0::2] = torch.sin(input_pos * div_term)
        input_pe[:, 1::2] = torch.cos(input_pos * div_term)
        self.register_buffer("input_pe", input_pe)

        output_pe = torch.zeros(output_window_len, d_model)
        output_pe[:, 0::2] = torch.sin(output_pos * div_term)
        output_pe[:, 1::2] = torch.cos(output_pos * div_term)
        self.register_buffer("output_pe", output_pe)

        # self.encoder_pos_embedding = torch.nn.Embedding(input_window_len, d_model)
        # self.encoder_pos_embedding.weight.data.copy_(input_pe)

        # self.decoder_pos_embedding = torch.nn.Embedding(output_window_len, d_model)
        # self.decoder_pos_embedding.weight.data.copy_(output_pe)

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
    
