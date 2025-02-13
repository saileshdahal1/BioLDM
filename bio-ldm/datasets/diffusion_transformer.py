import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from x_transformer import AbsolutePositionalEmbedding, Encoder

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        tx_dim,
        tx_depth,
        heads,
        latent_dim = None,
        max_seq_len=64,
        self_condition = False,
        dropout = 0.1,
        scale_shift = False,
        class_conditional=False,
        num_classes=0,
        class_unconditional_prob=0,
        seq2seq=False,
        seq2seq_context_dim=0,
        dual_output=False,
        num_dense_connections=0,
        dense_output_connection=False,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob
        self.seq2seq = seq2seq
        self.dense_output_connection = dense_output_connection
        self.max_seq_len = max_seq_len
        sinu_pos_emb = SinusoidalPosEmb(tx_dim)
        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )
        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        self.cross = seq2seq
        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout = dropout,    # dropout post-attention
            ff_dropout = dropout,       # feedforward dropout
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
            num_dense_connections=num_dense_connections,
        )
        if self.class_conditional:
            assert num_classes > 0
            self.class_embedding = nn.Sequential(nn.Embedding(num_classes+1, tx_dim),
                                                    nn.Linear(tx_dim, time_emb_dim))
        if self.seq2seq:
            self.null_embedding_seq2seq = nn.Embedding(1, tx_dim)
            self.seq2seq_proj = nn.Linear(seq2seq_context_dim, tx_dim)
            
        if self.self_condition:
            self.input_proj = nn.Linear(latent_dim*2, tx_dim)
            self.init_self_cond = nn.Parameter(torch.randn(1, latent_dim))
            nn.init.normal_(self.init_self_cond, std = 0.02)
        else:
            self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim*2 if dense_output_connection else tx_dim, latent_dim*2 if dual_output else latent_dim)

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, x_self_cond = None, class_id = None, seq2seq_cond = None, seq2seq_mask = None):
        time_emb = self.time_mlp(time*1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        if self.class_conditional:
            assert exists(class_id)
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, 'b d -> b 1 d')
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)
        #print(seq2seq_cond.shape)

        if self.self_condition:
            if exists(x_self_cond):
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(self.init_self_cond, '1 d -> b l d', b=x.shape[0], l=x.shape[1])
                x = torch.cat((x, repeated_x_self_cond), dim=-1)
        x_input = self.input_proj(x)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)
        if self.cross:
            context, context_mask = [], []
            if self.seq2seq:
                if seq2seq_cond is None:
                    null_context = repeat(self.null_embedding_seq2seq.weight, '1 d -> b 1 d', b=x.shape[0])
                    context.append(null_context)
                    context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
                else:
                    projected_seq2seq = self.seq2seq_proj(seq2seq_cond)
                    seq2seq_residual = projected_seq2seq + seq2seq_cond  # Residual connection
                    context.append(seq2seq_residual)
                    context_mask.append(seq2seq_mask)
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)
        x = self.norm(x)
        return self.output_proj(x)
