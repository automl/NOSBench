import math

import torch
from torch import nn
from torch.nn.modules.transformer import (
    _get_activation_fn,
    Module,
    Tensor,
    Optional,
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
)
from torch.utils.checkpoint import checkpoint


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        recompute_attn=False,
        save_trainingset_representations=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_mask = int(src_mask.item())
        if self.save_trainingset_representations:
            assert (
                isinstance(src_mask, int) and not self.training
            ), "save_trainingset_representations is only supported in eval mode and requires src_mask to be an int"

        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src
        if isinstance(src_mask, tuple):
            # global attention setup
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask

            num_global_tokens = global_src_mask.shape[0]
            num_train_tokens = trainset_src_mask.shape[0]

            global_tokens_src = src_[:num_global_tokens]
            train_tokens_src = src_[
                num_global_tokens : num_global_tokens + num_train_tokens
            ]
            global_and_train_tokens_src = src_[: num_global_tokens + num_train_tokens]
            eval_tokens_src = src_[num_global_tokens + num_train_tokens :]

            attn = (
                partial(checkpoint, self.self_attn)
                if self.recompute_attn
                else self.self_attn
            )

            global_tokens_src2 = attn(
                global_tokens_src,
                global_and_train_tokens_src,
                global_and_train_tokens_src,
                None,
                True,
                global_src_mask,
            )[0]
            train_tokens_src2 = attn(
                train_tokens_src,
                global_tokens_src,
                global_tokens_src,
                None,
                True,
                trainset_src_mask,
            )[0]
            eval_tokens_src2 = attn(
                eval_tokens_src, src_, src_, None, True, valset_src_mask
            )[0]

            src2 = torch.cat(
                [global_tokens_src2, train_tokens_src2, eval_tokens_src2], dim=0
            )

        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_to_attend_to = src_[:single_eval_position]
            if self.save_trainingset_representations:
                if (
                    single_eval_position == src_.shape[0]
                    or single_eval_position is None
                ):
                    self.saved_src_to_attend_to = src_to_attend_to
                elif single_eval_position == 0:
                    if self.saved_src_to_attend_to is None:
                        raise ValueError(
                            "First save the trainingset representations by passing in a src_mask of None or the length of the src"
                        )
                    src_to_attend_to = self.saved_src_to_attend_to
                else:
                    raise ValueError(
                        "save_trainingset_representations only supports single_eval_position == 0 or single_eval_position == src.shape[0]"
                    )
            src_left = self.self_attn(
                src_[:single_eval_position],
                src_[:single_eval_position],
                src_[:single_eval_position],
            )[0]
            src_right = self.self_attn(
                src_[single_eval_position:], src_to_attend_to, src_to_attend_to
            )[0]
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                src2 = checkpoint(
                    self.self_attn,
                    src_,
                    src_,
                    src_,
                    src_key_padding_mask,
                    True,
                    src_mask,
                )[0]
            else:
                src2 = self.self_attn(
                    src_,
                    src_,
                    src_,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                )[0]
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src


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

        encoder_layer = TransformerEncoderLayer(
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

        output = self.transformer_encoder(src, torch.tensor(single_eval_pos).float())

        out_range_start = single_eval_pos
        output = {k: v(output[out_range_start:]) for k, v in self.decoder_dict.items()}

        return output
