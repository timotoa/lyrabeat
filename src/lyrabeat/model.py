import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math


class LocalAttention(nn.Module):
    def __init__(self, embed_size: int, attention_limit: int = 128):
        super().__init__()
        self.embed_size = embed_size
        self.attention_limit = attention_limit
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        pad = self.attention_limit
        extended = F.pad(x, (0, 0, pad, pad))
        query = self.query(x)
        keys = self.key(extended).unfold(1, 2 * pad + 1, 1)
        values = self.value(extended).unfold(1, 2 * pad + 1, 1)

        energy = torch.matmul(query.unsqueeze(2), keys)
        energy /= self.embed_size ** 0.5

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values.transpose(3, 2)).squeeze(2)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, ff_hidden_size: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(embed_size, ff_hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(ff_hidden_size, embed_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RoPE(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, x, seq_len):
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.arange(0, self.embed_size, 2, device=x.device)
        div_term = torch.exp(div_term * -(math.log(10000.0) / self.embed_size))

        pe = torch.zeros(seq_len, self.embed_size, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return x + pe


class TransformerLayer(nn.Module):
    def __init__(self, embed_size: int, ff_hidden_size: int, attention_limit: int, **kwargs):
        super().__init__()
        self.attention = LocalAttention(embed_size, attention_limit)
        self.feedforward = FeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.rope = RoPE(embed_size)

    def forward(self, x):
        x = self.rope(x, x.size(1))
        attention_out = self.attention(x)
        x = self.norm1(attention_out + x)
        ff_out = self.feedforward(x)
        out = self.norm2(ff_out + x)
        return out


class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        in_channels = config["n_mels"]
        kernel_size = config["encoder_kernel_size"]
        layers = []

        for out_channels in config["encoder"]:
            layers.append(nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size//2
            ))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        embed_size = encoder[-1]

        layers = []
        for i in range(len(encoder) - 1):
            layers.append(nn.ConvTranspose1d(embed_size, embed_size, 4, 2, 1))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose1d(embed_size, 1, 4, 2, 1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class AudioTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = Encoder(config)

        layers = []
        embed_size = config["encoder"][-1]
        for i in range(config["transformer_layer_count"]):
            layer = TransformerLayer(embed_size, **config)
            layers.append(layer)
        self.transformer = nn.Sequential(*layers)
        self.decoder = Decoder(config["encoder"])

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        return x
