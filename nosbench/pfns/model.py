import math

import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class PFNModel(nn.Module):
    def __init__(
        self,
        ninp,
        nout,
        nhead,
        nhid,
        nlayers,
        num_features,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            Normalize(0.5, math.sqrt(1 / 12)), nn.Linear(num_features, ninp)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            ninp,
            nhead,
            nhid,
            dropout,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.y_encoder = nn.Linear(1, ninp)

        self.decoder_dict = nn.ModuleDict(
            {
                "standard": nn.Sequential(
                    nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, nout)
                )
            }
        )

        self.init_weights()

    def init_weights(self):
        initrange = 1.0
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = (
                layer.self_attn
                if isinstance(layer.self_attn, nn.ModuleList)
                else [layer.self_attn]
            )
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, single_eval_pos=None):
        x_src, y_src = src

        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]

        x_src = self.encoder(x_src)

        y_src = (
            self.y_encoder(
                y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src
            )
            if y_src is not None
            else None
        )

        train_x = x_src[:single_eval_pos]
        if y_src is not None:
            train_x = train_x + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        output = self.transformer_encoder(src)

        out_range_start = single_eval_pos
        output = {k: v(output[out_range_start:]) for k, v in self.decoder_dict.items()}

        return output
