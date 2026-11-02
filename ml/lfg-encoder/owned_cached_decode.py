"""Bounded incremental reference inference for the experimental owned student.

Each session belongs to one model and one request batch. This Python reference
is not the native runtime and has not been performance-qualified.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from owned_student import OwnedByteStudent


class DecodeSession:
    """Cache short-convolution inputs and attention keys/values for one batch."""

    def __init__(self, model: OwnedByteStudent, batch_size: int):
        if not isinstance(model, OwnedByteStudent) or model.training:
            raise ValueError('an owned student in evaluation mode is required')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch size must be positive')
        self.model = model
        self.batch_size = batch_size
        self.position = 0
        self._states = [{'conv': None, 'keys': None, 'values': None} for _ in model.blocks]

    @torch.inference_mode()
    def step(self, token: torch.Tensor) -> torch.Tensor:
        """Consume one byte per batch item and return its next-byte logits.

        A session rejects context overflow instead of silently evicting history:
        absolute positional embeddings make sliding-window reuse inequivalent.
        """
        if self.model.training:
            raise ValueError('model must remain in evaluation mode')
        if self.position >= self.model.context:
            raise ValueError('configured context exhausted; create a new session')
        if (not isinstance(token, torch.Tensor) or token.dtype != torch.long
                or token.shape != (self.batch_size,) or token.device != self.model.embed.weight.device
                or bool(((token < 0) | (token > 255)).any())):
            raise ValueError('expected one byte token per batch item on the model device')
        x = self.model.embed(token).unsqueeze(1) + self.model.position.weight[self.position][None, None, :]
        next_states = []
        for block, state in zip(self.model.blocks, self._states):
            content, gate = block.input_gate(block.norm_conv(x)).chunk(2, dim=-1)
            content = content.transpose(1, 2)
            history = state['conv']
            if history is None:
                history = content.new_zeros((self.batch_size, content.shape[1], block.kernel - 1))
            window = torch.cat((history, content), dim=2)
            convolved = F.conv1d(window, block.conv.weight, block.conv.bias, groups=content.shape[1]).transpose(1, 2)
            x = x + block.conv_out(convolved * torch.sigmoid(gate))
            keys, values = state['keys'], state['values']
            if block.attention:
                _, _, width = x.shape
                q, k, v = block.qkv(block.norm_attn(x)).chunk(3, dim=-1)
                def split(tensor):
                    return tensor.reshape(self.batch_size, 1, block.heads, width // block.heads).transpose(1, 2)
                q, k, v = split(q), split(k), split(v)
                keys = k if keys is None else torch.cat((keys, k), dim=2)
                values = v if values is None else torch.cat((values, v), dim=2)
                attended = F.scaled_dot_product_attention(q, keys, values)
                x = x + block.attn_out(attended.transpose(1, 2).contiguous().reshape(self.batch_size, 1, width))
            x = x + block.ff(block.norm_ff(x))
            next_states.append({'conv': window[:, :, 1:] if block.kernel > 1 else window[:, :, :0],
                                'keys': keys, 'values': values})
        result = self.model.output(self.model.norm(x)).squeeze(1)
        self._states = next_states
        self.position += 1
        return result
