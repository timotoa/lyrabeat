import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class LocalAttention(nn.Module):
    def __init__(self, embed_size: int, attention_limit: int = 128):
        super().__init__()
        self.embed_size = embed_size
        self.attention_limit = attention_limit
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        query = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        pad = self.attention_limit
        padded_keys = F.pad(keys, (0, 0, pad, pad))
        padded_values = F.pad(values, (0, 0, pad, pad))
        local_keys = padded_keys.unfold(1, 2 * pad + 1, 1)
        local_values = padded_values.unfold(1, 2 * pad + 1, 1)
        print(local_keys.size(), local_values.size())
        energy = torch.matmul(
            query.unsqueeze(2),
            local_keys.transpose(-2, -1)
        ).squeeze(2)
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention.unsqueeze(2), local_values).squeeze(2)
        return out

# class LocalAttention(nn.Module):
#     def __init__(self, embed_size: int):
#         super().__init__()
#         self.embed_size = embed_size
#         self.query = nn.Linear(embed_size, embed_size)
#         self.key = nn.Linear(embed_size, embed_size)
#         self.value = nn.Linear(embed_size, embed_size)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):

#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         scaled = self.embed_size ** 0.5
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / scaled
#         attention = self.softmax(scores)
#         out = torch.matmul(attention, V)
#         return out


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


class TransformerLayer(nn.Module):
    def __init__(self, embed_size: int, ff_hidden_size: int, attention_limit: int, **kwargs):
        super().__init__()
        self.attention = LocalAttention(embed_size)
        self.feedforward = FeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
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
