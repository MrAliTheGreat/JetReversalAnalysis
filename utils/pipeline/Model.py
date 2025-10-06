from transformers import T5Config, T5ForConditionalGeneration
import torch

class TimeSeriesHuggingFaceTransformer(T5ForConditionalGeneration):
    def __init__(self, input_dim, output_dim, d_model, num_head, num_encoder_layers, num_decoder_layers, position_wise_ffn_dim, dropout):
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
            relative_attention_num_buckets = 32, 
            d_kv = d_model // num_head,

            output_attention = True                                         # To get multi-head attentions (num_attention_head values each)
        )
        
        super().__init__(config)                                            # Creates model with random weights

        self.encoder.embed_tokens = torch.nn.Linear(input_dim, d_model)     # Embedding layer for input
        self.decoder.embed_tokens = torch.nn.Linear(output_dim, d_model)    # Embedding layer for output
        
        self.lm_head = torch.nn.Linear(d_model, output_dim, bias = False)   # Last linear before output
        
        self.output_dim = output_dim

        self.bos_token = torch.nn.Parameter(torch.empty(1, 1, output_dim))
        torch.nn.init.normal_(self.bos_token, mean = 0.0, std = 1.0)

        self.attention_weights = {
            "encoder_attention": [],
            "decoder_attention": [],
            "cross_attention": []
        }

    def forward(self, inputs_embeds, decoder_inputs_embeds, **kwargs):
        outputs = super().forward(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            **kwargs
        )

        if(outputs.encoder_attentions):
            self.attention_weights["encoder_attention"].extend(outputs.encoder_attentions)
            
        if(outputs.decoder_attentions):
            self.attention_weights["decoder_attention"].extend(outputs.decoder_attentions)
            
        if(outputs.cross_attentions):
            # Encoder-Decoder Attention
            self.attention_weights["cross_attention"].extend(outputs.cross_attentions)

        return outputs