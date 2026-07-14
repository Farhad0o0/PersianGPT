import torch
import torch.nn as nn

from configs.config import GPTConfig





class PersianGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token Embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
        )

        # Positional Embedding
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.embed_dim,
        )

        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=False,
        )
        
        # Weight Tying
        #self.lm_head.weight = self.token_embedding.weight
    

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        device = input_ids.device

        positions = torch.arange(
            seq_len,
            device=device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()

        x = self.transformer(
            x,
            mask=causal_mask,
        )

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits