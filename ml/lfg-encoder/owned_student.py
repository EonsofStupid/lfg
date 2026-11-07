"""Experimental from-scratch byte student and decoupled Top-K objective.

This is a research candidate, not a released or performance-qualified model.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def validate_architecture(config: dict) -> None:
    fields = {'width', 'layers', 'heads', 'conv_kernel', 'context', 'ff_multiplier', 'attention_every'}
    if (not isinstance(config, dict) or set(config) != fields
            or any(type(config[key]) is not int or config[key] < 1 for key in fields)
            or config['width'] % config['heads'] != 0):
        raise ValueError('invalid student architecture')


class HybridBlock(nn.Module):
    def __init__(self, config: dict, attention: bool):
        super().__init__()
        width = config['width']
        self.kernel = config['conv_kernel']
        self.heads = config['heads']
        self.norm_conv = nn.LayerNorm(width)
        self.input_gate = nn.Linear(width, 2 * width)
        self.conv = nn.Conv1d(width, width, self.kernel, groups=width)
        self.conv_out = nn.Linear(width, width)
        self.attention = attention
        if attention:
            self.norm_attn = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width, bias=False)
            self.attn_out = nn.Linear(width, width, bias=False)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * config['ff_multiplier']), nn.GELU(),
                                nn.Linear(width * config['ff_multiplier'], width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.input_gate(self.norm_conv(x)).chunk(2, dim=-1)
        convolved = self.conv(F.pad(content.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        x = x + self.conv_out(convolved * torch.sigmoid(gate))
        if self.attention:
            batch, length, width = x.shape
            q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
            def split(tensor):
                return tensor.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
            attended = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
            x = x + self.attn_out(attended.transpose(1, 2).contiguous().reshape(batch, length, width))
        return x + self.ff(self.norm_ff(x))


class OwnedByteStudent(nn.Module):
    """A configurable research backbone initialized entirely from random weights."""
    def __init__(self, config: dict):
        super().__init__()
        validate_architecture(config)
        self.context = config['context']
        self.embed = nn.Embedding(256, config['width'])
        self.position = nn.Embedding(self.context, config['width'])
        self.blocks = nn.ModuleList(HybridBlock(config, (index + 1) % config['attention_every'] == 0)
                                    for index in range(config['layers']))
        self.norm = nn.LayerNorm(config['width'])
        self.output = nn.Linear(config['width'], 256, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.context:
            raise ValueError('tokens must be a batch within the configured context')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def decoupled_topk_loss(student_logits: torch.Tensor, indices: torch.Tensor,
                        teacher_top_probabilities: torch.Tensor, teacher_tail_probability: torch.Tensor,
                        temperature: float) -> torch.Tensor:
    """Forward membership KL plus teacher-mass-weighted tempered conditional KL.

    Teacher probabilities must be normalized over the full vocabulary before
    truncation. The explicit tail probability is required to recover top-K mass.
    """
    if (type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature < 1
            or student_logits.ndim < 2 or indices.shape != teacher_top_probabilities.shape
            or indices.shape[:-1] != student_logits.shape[:-1]
            or teacher_tail_probability.shape != student_logits.shape[:-1]
            or indices.dtype != torch.long or indices.shape[-1] < 1
            or indices.shape[-1] >= student_logits.shape[-1]):
        raise ValueError('invalid Top-K dimensions or temperature')
    if (not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(teacher_top_probabilities).all())
            or not bool(torch.isfinite(teacher_tail_probability).all())
            or bool((indices < 0).any()) or bool((indices >= student_logits.shape[-1]).any())
            or bool((teacher_top_probabilities <= 0).any()) or bool((teacher_tail_probability <= 0).any())):
        raise ValueError('invalid Top-K probabilities or indices')
    if indices.shape[-1] > 1 and bool((indices.sort(dim=-1).values.diff(dim=-1) == 0).any()):
        raise ValueError('duplicate Top-K indices')
    top_mass = teacher_top_probabilities.sum(dim=-1)
    if (bool((top_mass >= 1).any()) or bool((teacher_tail_probability >= 1).any())
            or not bool(torch.allclose(top_mass + teacher_tail_probability, torch.ones_like(top_mass), rtol=1e-5, atol=1e-7))):
        raise ValueError('teacher top and tail probabilities must sum to one')
    selected = student_logits.gather(-1, indices)
    mask = torch.zeros_like(student_logits, dtype=torch.bool).scatter(-1, indices, True)
    log_total = torch.logsumexp(student_logits, dim=-1)
    log_student_top = torch.logsumexp(selected, dim=-1) - log_total
    log_student_tail = torch.logsumexp(student_logits.masked_fill(mask, -torch.inf), dim=-1) - log_total
    binary = (top_mass * (top_mass.log() - log_student_top)
              + teacher_tail_probability * (teacher_tail_probability.log() - log_student_tail))
    teacher_conditional_log = F.log_softmax(teacher_top_probabilities.log() / temperature, dim=-1)
    student_conditional_log = F.log_softmax(selected / temperature, dim=-1)
    conditional = (teacher_conditional_log.exp() * (teacher_conditional_log - student_conditional_log)).sum(dim=-1)
    return (binary + top_mass * temperature**2 * conditional).mean()
