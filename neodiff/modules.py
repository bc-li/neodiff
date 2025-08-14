import enum
import math
from typing import Optional

import torch
from torch import nn

from fairseq.models.nat import NATransformerDecoder
from fairseq.models.transformer  import TransformerEncoder
from fairseq.modules import PositionalEmbedding, TransformerDecoderLayer

from .utils import build_ffn, timestep_embedding


class NeoDiffEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None):
        super().__init__(args, dictionary, embed_tokens)

        self.project_in_dim = project_in_dim

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        
        x = embed = self.embed_scale * token_embedding
        x = self.project_in_dim(x)

        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


class EmbedNormPosition(enum.Enum):
    NO_EMBED_NORM = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class SelfCondPosition(enum.Enum):
    NO_SELF_COND = enum.auto()
    BEFORE_PROJ = enum.auto()
    AFTER_PROJ = enum.auto()


class NeoDiffDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, project_in_dim=None, project_out_dim=None):
        super().__init__(args, dictionary, embed_tokens)

        latent_dim = args.latent_dim
        model_dim = args.model_dim
        
        self.project_in_dim = project_in_dim
        self.project_out_dim = project_out_dim

        # embedding normalization
        if not args.embed_norm:
            args.embed_norm_position = EmbedNormPosition.NO_EMBED_NORM
        elif args.embed_norm_before_proj:
            args.embed_norm_position = EmbedNormPosition.BEFORE_PROJ
        else:
            args.embed_norm_position = EmbedNormPosition.AFTER_PROJ
        
        if args.embed_norm:
            self.embed_norm = nn.LayerNorm(
                latent_dim if args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ
                else model_dim,
                elementwise_affine=args.embed_norm_affine
            )

        # self-conditioning
        if not args.self_cond:
            args.self_cond_position = SelfCondPosition.NO_SELF_COND
        elif args.self_cond_before_proj:
            args.self_cond_position = SelfCondPosition.BEFORE_PROJ
        else:
            args.self_cond_position = SelfCondPosition.AFTER_PROJ
        
        if args.self_cond:
            self_cond_dim = (
                latent_dim if args.self_cond_position == SelfCondPosition.BEFORE_PROJ
                else model_dim
            )

            self.self_cond_proj = build_ffn(
                self_cond_dim * 2, self_cond_dim, self_cond_dim,
                args.activation_fn, args.dropout
            )

        self.embed_time = build_ffn(model_dim, model_dim * 4, model_dim, args.activation_fn)

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.latent_dim)

    def forward_embedding(self, tokens):
        embed = self.embed_scale * self.embed_tokens(tokens)

        if self.args.embed_norm_position == EmbedNormPosition.BEFORE_PROJ:
            embed = self.embed_norm(embed)
        
        return embed

    def forward(self, x_t, taus, mask, encoder_out, prev_x_0_hat=None):
        hidden = self.forward_hidden(x_t, taus, mask, prev_x_0_hat)

        # B x T x C -> T x B x C
        hidden = hidden.transpose(0, 1)
        attn = None
        inner_states = [hidden]

        # decoder layers
        for i, layer in enumerate(self.layers):
            # if encoder_out is not None:
            #     encoder_out_tensor = encoder_out["encoder_out"]
            #     encoder_padding_mask_tensor = encoder_out["encoder_padding_mask"]
            # else:
            #     encoder_out_tensor = None
            #     encoder_padding_mask_tensor = None
            # # Debugging statements
            # print(f"Layer {i}")
            # print(f"hidden type: {type(hidden)}, shape: {hidden.shape if isinstance(hidden, torch.Tensor) else 'N/A'}")
            # print(f"encoder_out_tensor type: {type(encoder_out_tensor)}, shape: {encoder_out_tensor.shape if isinstance(encoder_out_tensor, torch.Tensor) else 'N/A'}")
            # print(f"encoder_padding_mask_tensor type: {type(encoder_padding_mask_tensor)}, shape: {encoder_padding_mask_tensor.shape if isinstance(encoder_padding_mask_tensor, torch.Tensor) else 'N/A'}")
            # print(f"mask type: {type(mask)}, shape: {mask.shape if isinstance(mask, torch.Tensor) else 'N/A'}")
            # print(type(encoder_out["encoder_out"]))
            # print(len(encoder_out["encoder_out"]))
            # print(encoder_out["encoder_out"])
            # exit()
            # print(type(encoder_out["encoder_padding_mask"]))
            # print(len(encoder_out["encoder_padding_mask"]))
            # exit()
            # https://fairseq.readthedocs.io/en/latest/models.html#fairseq.models.fconv.FConvEncoder
            hidden, attn, _ = layer(
                hidden,
                encoder_out["encoder_out"][0] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"][0] if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=~mask,
            )
            inner_states.append(hidden)

        if self.layer_norm:
            hidden = self.layer_norm(hidden)

        # T x B x C -> B x T x C
        hidden = hidden.transpose(0, 1)

        hidden = self.project_out_dim(hidden)
        return hidden, {"attn": attn, "inner_states": inner_states}

    def forward_hidden(self, x_t, taus, mask, prev_x_0_hat=None):
        # self-conditioning
        if self.args.self_cond_position == SelfCondPosition.BEFORE_PROJ:
            cat_embed = torch.cat((x_t, prev_x_0_hat), -1)
            hidden = self.project_in_dim(self.self_cond_proj(cat_embed))
        
        elif self.args.self_cond_position == SelfCondPosition.AFTER_PROJ:
            z_hidden = self.project_in_dim(x_t)
            prev_hidden = self.project_in_dim(prev_x_0_hat)
            cat_hidden = torch.cat((z_hidden, prev_hidden), -1)
            hidden = self.self_cond_proj(cat_hidden)
        
        else:
            hidden = self.project_in_dim(x_t)

        # time embedding
        time_emb = self.embed_time(timestep_embedding(taus, self.args.model_dim).type_as(x_t))
        hidden = hidden + time_emb

        # position embedding
        positions = self.embed_positions(mask.long() + self.padding_idx)
        hidden = hidden + positions
        
        # embedding normalization
        if self.args.embed_norm_position == EmbedNormPosition.AFTER_PROJ:
            hidden = self.embed_norm(hidden)
        
        hidden = self.dropout_module(hidden)
        return hidden


class TauPredictor(nn.Module):
    def __init__(self, args, project_in_dim, embed_positions, embed_time):
        super().__init__()

        self.args = args
        self.project_in_dim = project_in_dim
        self.embed_positions = embed_positions
        self.embed_time = embed_time

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args)
            for _ in range(args.tau_predictor_layers)
        ])

        self.project_out_dim = nn.Linear(args.model_dim, args.num_tau)

        if args.tau_mse:
            self.mse_project_out_dim = nn.Linear(args.model_dim, 1)
        # self.dropout = nn.Dropout(args.dropout)

        # self.relu = nn.ReLU()
        # self.activation = nn.Sigmoid()
        # self.activation = nn.Tanh()

        # self.ffn = nn.Sequential(
        #     nn.Linear(in_dim, ffn_dim),
        #     nn.ReLU(),
        #     nn.Linear(ffn_dim, 1),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        # )

    def forward(self, x_0_hat, mask, ts, encoder_out):
        hidden = self.project_in_dim(x_0_hat)

        if ts is not None:
            time_emb = self.embed_time(timestep_embedding(ts, self.args.model_dim).type_as(x_0_hat))
            hidden = hidden + time_emb

        # # position embedding
        # positions = self.embed_positions(mask.long() + self.embed_positions.padding_idx)
        # hidden = hidden + positions

        if self.args.pred_tau_detach_all:
            hidden = hidden.detach()
        hidden = hidden.detach().transpose(0, 1)

        for i, layer in enumerate(self.layers):
            # hidden, attn, _ = layer(
            #     hidden,
            #     encoder_out.encoder_out if encoder_out is not None else None,
            #     encoder_out.encoder_padding_mask if encoder_out is not None else None,
            #     self_attn_mask=None,
            #     self_attn_padding_mask=~mask,
            # )
            hidden, attn, _ = layer(
                hidden,
                encoder_out["encoder_out"][0] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"][0] if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=~mask,
            )

        hidden = hidden.transpose(0, 1)

        # output = self.project_out_dim(hidden).squeeze(-1)
        output = self.project_out_dim(hidden)
        # output = self.dropout(output)

        if self.args.tau_mse:
            output_mse = self.mse_project_out_dim(hidden)[..., 0]
        else:
            output_mse = None

        # output = self.relu(output
        # output = self.activation(-output) * 2
        # output = self.activation(output) * 2 - 1
        # output = self.activation(output) / 2 + 0.5
        return output, output_mse
