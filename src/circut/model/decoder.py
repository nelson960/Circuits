from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from circut.config import ModelSpec


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if not return_attention and head_mask is None:
            sdpa_mask = None
            use_causal_flag = True
            if attention_mask is not None:
                if attention_mask.shape != (batch_size, seq_len):
                    raise ValueError(
                        f"Expected attention_mask shape {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}"
                    )
                key_mask = attention_mask[:, None, None, :]
                causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril()
                sdpa_mask = key_mask & causal_mask.view(1, 1, seq_len, seq_len)
                use_causal_flag = False
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=use_causal_flag,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.out_proj(attn_output)
            return output, None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, torch.finfo(attn_scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"Expected attention_mask shape {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}")
            key_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(~key_mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)

        if head_mask is not None:
            if head_mask.shape != (self.n_heads,):
                raise ValueError(f"Expected head_mask shape {(self.n_heads,)}, got {tuple(head_mask.shape)}")
            attn_output = attn_output * head_mask.view(1, self.n_heads, 1, 1)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        return output, attn_probs if return_attention else None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.fc_in = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        neuron_mask: torch.Tensor | None = None,
        return_hidden_state: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        hidden_pre = self.fc_in(x)
        hidden = self.activation(hidden_pre)
        if neuron_mask is not None:
            if neuron_mask.shape != (self.d_ff,):
                raise ValueError(f"Expected neuron_mask shape {(self.d_ff,)}, got {tuple(neuron_mask.shape)}")
            hidden = hidden * neuron_mask.view(1, 1, self.d_ff)
        output = self.fc_out(hidden)
        output = self.dropout(output)
        if not return_hidden_state:
            return output, None
        return output, {"hidden_pre": hidden_pre, "hidden": hidden}


class TransformerBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(spec.d_model)
        self.attn = CausalSelfAttention(spec.d_model, spec.n_heads, spec.dropout)
        self.ln_2 = nn.LayerNorm(spec.d_model)
        self.ff = FeedForward(spec.d_model, spec.d_ff, spec.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        mlp_mask: float | torch.Tensor | None = None,
        neuron_mask: torch.Tensor | None = None,
        post_attn_patch: torch.Tensor | None = None,
        post_mlp_patch: torch.Tensor | None = None,
        return_attention: bool = False,
        return_block_states: bool = False,
        return_mlp_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        attn_out, attn_probs = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_attention=return_attention,
        )
        x = x + attn_out
        if post_attn_patch is not None:
            if post_attn_patch.shape != x.shape:
                raise ValueError(f"Expected post_attn_patch shape {tuple(x.shape)}, got {tuple(post_attn_patch.shape)}")
            x = post_attn_patch
        post_attn = x
        ff_out, mlp_states = self.ff(
            self.ln_2(x),
            neuron_mask=neuron_mask,
            return_hidden_state=return_mlp_states,
        )
        if mlp_mask is not None:
            if not isinstance(mlp_mask, torch.Tensor):
                mlp_mask = torch.tensor(float(mlp_mask), device=ff_out.device, dtype=ff_out.dtype)
            ff_out = ff_out * mlp_mask
        x = x + ff_out
        if post_mlp_patch is not None:
            if post_mlp_patch.shape != x.shape:
                raise ValueError(f"Expected post_mlp_patch shape {tuple(x.shape)}, got {tuple(post_mlp_patch.shape)}")
            x = post_mlp_patch
        block_states = None
        if return_block_states:
            block_states = {"post_attn": post_attn, "post_mlp": x}
        return x, attn_probs, block_states, mlp_states


@dataclass(frozen=True)
class DecoderOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    attentions: list[torch.Tensor] | None
    residual_streams: dict[str, torch.Tensor] | None = None
    mlp_states: dict[str, torch.Tensor] | None = None


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, spec: ModelSpec, vocab_size: int) -> None:
        super().__init__()
        self.spec = spec
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, spec.d_model)
        self.position_embedding = nn.Embedding(spec.max_seq_len, spec.d_model)
        self.dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(spec) for _ in range(spec.n_layers)])
        self.final_norm = nn.LayerNorm(spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
        return_residual_streams: bool = False,
        return_mlp_states: bool = False,
        head_mask: dict[int, torch.Tensor] | None = None,
        mlp_mask: dict[int, float | torch.Tensor] | None = None,
        neuron_mask: dict[int, torch.Tensor] | None = None,
        residual_patch: dict[str, torch.Tensor] | None = None,
    ) -> DecoderOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected rank-2 input_ids, got shape {tuple(input_ids.shape)}")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.spec.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.spec.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        remaining_patch_names = set(residual_patch) if residual_patch is not None else set()
        if residual_patch is not None and "embedding" in residual_patch:
            embedding_patch = residual_patch["embedding"].to(hidden.device)
            if embedding_patch.shape != hidden.shape:
                raise ValueError(f"Expected embedding patch shape {tuple(hidden.shape)}, got {tuple(embedding_patch.shape)}")
            hidden = embedding_patch
            remaining_patch_names.remove("embedding")
        residual_streams: dict[str, torch.Tensor] | None = None
        if return_residual_streams:
            residual_streams = {"embedding": hidden}
        mlp_states: dict[str, torch.Tensor] | None = None
        if return_mlp_states:
            mlp_states = {}

        attentions: list[torch.Tensor] | None = [] if return_attentions else None
        for layer_index, block in enumerate(self.blocks):
            layer_head_mask = None
            if head_mask is not None and layer_index in head_mask:
                layer_head_mask = head_mask[layer_index].to(hidden.device)
            layer_mlp_mask = None
            if mlp_mask is not None and layer_index in mlp_mask:
                layer_mlp_mask = mlp_mask[layer_index]
            layer_neuron_mask = None
            if neuron_mask is not None and layer_index in neuron_mask:
                layer_neuron_mask = neuron_mask[layer_index].to(hidden.device)
            post_attn_patch = None
            post_attn_name = f"layer_{layer_index}_post_attn"
            if residual_patch is not None and post_attn_name in residual_patch:
                post_attn_patch = residual_patch[post_attn_name].to(hidden.device)
                remaining_patch_names.remove(post_attn_name)
            post_mlp_patch = None
            post_mlp_name = f"layer_{layer_index}_post_mlp"
            if residual_patch is not None and post_mlp_name in residual_patch:
                post_mlp_patch = residual_patch[post_mlp_name].to(hidden.device)
                remaining_patch_names.remove(post_mlp_name)
            hidden, attn_probs, block_states, layer_mlp_states = block(
                hidden,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                mlp_mask=layer_mlp_mask,
                neuron_mask=layer_neuron_mask,
                post_attn_patch=post_attn_patch,
                post_mlp_patch=post_mlp_patch,
                return_attention=return_attentions,
                return_block_states=return_residual_streams,
                return_mlp_states=return_mlp_states,
            )
            if attentions is not None and attn_probs is not None:
                attentions.append(attn_probs)
            if residual_streams is not None:
                if block_states is None:
                    raise RuntimeError("return_residual_streams requested, but block states were not returned.")
                residual_streams[f"layer_{layer_index}_post_attn"] = block_states["post_attn"]
                residual_streams[f"layer_{layer_index}_post_mlp"] = block_states["post_mlp"]
            if mlp_states is not None:
                if layer_mlp_states is None:
                    raise RuntimeError("return_mlp_states requested, but MLP states were not returned.")
                mlp_states[f"layer_{layer_index}_hidden_pre"] = layer_mlp_states["hidden_pre"]
                mlp_states[f"layer_{layer_index}_hidden"] = layer_mlp_states["hidden"]

        hidden = self.final_norm(hidden)
        if residual_patch is not None and "final_norm" in residual_patch:
            final_norm_patch = residual_patch["final_norm"].to(hidden.device)
            if final_norm_patch.shape != hidden.shape:
                raise ValueError(f"Expected final_norm patch shape {tuple(hidden.shape)}, got {tuple(final_norm_patch.shape)}")
            hidden = final_norm_patch
            remaining_patch_names.remove("final_norm")
        if residual_streams is not None:
            residual_streams["final_norm"] = hidden
        if remaining_patch_names:
            raise ValueError(f"Unknown residual patch stages requested: {sorted(remaining_patch_names)}")
        logits = self.lm_head(hidden)
        return DecoderOutput(
            logits=logits,
            hidden_states=hidden,
            attentions=attentions,
            residual_streams=residual_streams,
            mlp_states=mlp_states,
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
