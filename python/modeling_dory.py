# SPDX-License-Identifier: Apache-2.0
# Pure-PyTorch autograd-compatible implementation of Dory 1.0 4B for LoRA Training & Fine-Tuning
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

EPS = 1e-6

class DoryConfig(PreTrainedConfig):
    model_type = "dory"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2560,
        num_hidden_layers=72,
        intermediate_size=7168,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=262144,
        initializer_range=0.0177,
        sliding_window=1024,
        ngpt_alpha_init=0.05,
        attention_output_gate_logit_scale_init=None,
        n_input_layers=16,
        n_recurrent_layers=40,
        n_output_layers=16,
        n_recurrent_loops=2,
        layer_types=None,
        rope_parameters=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.sliding_window = sliding_window
        self.ngpt_alpha_init = ngpt_alpha_init
        self.attention_output_gate_logit_scale_init = attention_output_gate_logit_scale_init
        self.n_input_layers = n_input_layers
        self.n_recurrent_layers = n_recurrent_layers
        self.n_output_layers = n_output_layers
        self.n_recurrent_loops = n_recurrent_loops
        self.layer_types = layer_types
        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)

def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return y.to(x.dtype)

def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)

def slerp_alpha(a, b, alpha, alpha_scale):
    return slerp_norm(a, b, (alpha.float() * alpha_scale).abs())

class DoryRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DoryAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        std = config.initializer_range

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        self.sqk = nn.Parameter(torch.ones(self.q_size))
        self.s_gate = nn.Parameter(torch.ones(self.q_size))
        self._sqk_factor = (self.head_dim ** 0.5) / std
        g0 = config.attention_output_gate_logit_scale_init
        if g0 is None:
            g0 = config.hidden_size ** 0.5
        self._s_gate_factor = (g0 ** 0.5) / std

        is_full = (config.layer_types[layer_idx] == "full_attention") if config.layer_types else False
        base = 10000000.0 if is_full else 10000.0
        self.rotary_emb = DoryRotaryEmbedding(self.head_dim, base=base)

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        # Reshape para aplicação de RoPE e GQA: [bsz, heads, q_len, head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 1. RoPE aplicado antes do qk-norm (conforme especificação oficial NVIDIA)
        cos, sin = self.rotary_emb(v, q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 2. nGPT qk-norm após RoPE: q escalonado por (sqk * f)^2 pós-justnorm, k com justnorm simples
        # self.sqk possui shape [self.q_size] -> view [1, num_heads, 1, head_dim]
        sqk = (self.sqk.float() * self._sqk_factor).view(1, self.num_heads, 1, self.head_dim)
        q = (sqk.square() * justnorm(q, dim=-1).float()).to(q.dtype)
        k = justnorm(k, dim=-1)

        # 3. GQA repeat_interleave se num_heads != num_kv_heads
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # 4. Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.q_size)

        # 5. Output gate: sigmoid(g * (s_gate * f)^2), depois justnorm na dimensão escondida
        sg = (self.s_gate.float() * self._s_gate_factor).square().view(1, 1, self.q_size)
        gate = torch.sigmoid(g.float() * sg).to(attn_output.dtype)
        attn_output = attn_output * gate
        attn_output = justnorm(attn_output, dim=-1)

        # 6. Projeção de saída
        return self.o_proj(attn_output)

class DoryMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.ffn, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.ffn, bias=False)
        self.down_proj = nn.Linear(self.ffn, config.hidden_size, bias=False)

        self.suv_gate = nn.Parameter(torch.ones(self.ffn))
        self.suv_up = nn.Parameter(torch.ones(self.ffn))
        self._suv_factor = config.hidden_size ** 0.5

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        g = (g.float() * (self.suv_gate.float() * self._suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self._suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(x.dtype)
        m = justnorm(m, dim=-1)
        return self.down_proj(m)

class DoryDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention = (config.layer_types[layer_idx] != "mlp") if config.layer_types else (layer_idx % 2 == 0)
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range

        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx)
            self.attn_alpha = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        if self.is_attention:
            out = self.mixer(hidden_states, attention_mask=attention_mask)
            return slerp_alpha(residual, out, self.attn_alpha, self._alpha_scale)
        out = self.mixer(hidden_states)
        return slerp_alpha(residual, out, self.mlp_alpha, self._alpha_scale)

class DoryBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        c = self.config
        n_in = c.n_input_layers
        n_rec = c.n_recurrent_layers

        for layer in self.layers[:n_in]:
            h = layer(h)
        input_h = h

        state = input_h
        for _ in range(c.n_recurrent_loops):
            combined = justnorm(state + input_h)
            h = combined
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(h)
            state = justnorm(h)

        h = state
        for layer in self.layers[n_in + n_rec :]:
            h = layer(h)
        return h

class DoryForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DoryConfig
    base_model_prefix = "dory"

    def __init__(self, config):
        super().__init__(config)
        self.backbone = DoryBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.logit_scale = nn.Parameter(torch.ones(config.vocab_size))
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def set_input_embeddings(self, value):
        self.backbone.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        return model_inputs

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states)

        logits = logits.float() * (self.lm_head.logit_scale.float() / self.config.initializer_range)
        logits = logits.to(hidden_states.dtype)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)
