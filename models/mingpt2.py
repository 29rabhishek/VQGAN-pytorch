import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CrossAttention(nn.Module):
    """ Multi-head cross-attention layer """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.context_dim, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, context):
        B, T, C = x.size()
        B, context.shape
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)#Bs, nh, T, hdim
        q = self.query(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)#Bs, nh, S, hdim
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)#Bs nh, T, hdim
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))#Bs, nh, S, T
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v #Bs, nh, S, hdim
        y = y.transpose(1, 2).contiguous().view(B, -1, C) # Bs, S, nh, hdim-> Bs, S, nhxhdim

        y = self.resid_drop(self.proj(y))#Bs, T, C
        return y

class CausalSelfAttention(nn.Module):
    """ A vanilla multi-head masked self-attention layer """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # attention look like this [bs, nh, hdim, T, T], putting mask on in TxT metrics

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # Bs, nh, T, hdim
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Bs, T, nh, hdim-> Bs, T, nhxhdim

        y = self.resid_drop(self.proj(y))
        return y, present

class Block(nn.Module):
    """ Transformer block with cross-attention """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, context=None, layer_past=None, return_present=False):
        attn, present = self.self_attn(self.ln1(x), layer_past=layer_past)
        x = x + attn
        if context is not None:
            cross_attn = self.cross_attn(self.ln2(x), context)
            x = x + cross_attn
        x = x + self.mlp(self.ln3(x))
        if layer_past is not None or return_present:
            return x, present
        return x

class GPT(nn.Module):
    """ GPT with cross-attention """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, context_dim = 1088):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, context_dim = context_dim)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, context=None, embeddings=None):# we are here
        token_embeddings = self.tok_emb(idx)

        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x, context)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, None


