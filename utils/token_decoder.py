import json
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Pretrained LMs pack documents as text<eos>text<eos>, so a line starts after an EOS.
START_TOKEN_ID = 0
END_TOKEN_ID = 0
PAD_TOKEN_ID = 1
NEW_PARAMETER_MARKERS = ('cross_attn', 'memory_proj')
LM_CONFIG_KEYS = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                  'num_key_value_heads', 'rms_norm_eps', 'rope_theta')


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * hidden).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SelfAttention(nn.Module):
    """OLMo 3 attention: RMSNorm over the full query and key projections before RoPE, grouped KV heads."""

    def __init__(self, config: dict):
        super().__init__()
        hidden, heads, kv_heads = config['hidden_size'], config['num_attention_heads'], config['num_key_value_heads']
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(heads * self.head_dim, config['rms_norm_eps'])
        self.k_norm = RMSNorm(kv_heads * self.head_dim, config['rms_norm_eps'])

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.q_norm(self.q_proj(hidden)).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden)).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query = (query * cos + rotate_half(query) * sin).to(value.dtype)
        key = (key * cos + rotate_half(key) * sin).to(value.dtype)
        if cache is not None:
            if cache:
                key = torch.cat([cache[0], key], dim=2)
                value = torch.cat([cache[1], value], dim=2)
            cache[:] = [key, value]
        # a cached step has one query that may see every earlier key
        causal = cache is None or length > 1
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=causal, enable_gqa=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class CrossAttention(nn.Module):
    """Reads the encoder columns. The output projection starts at zero, so the pretrained LM is unchanged at
    step 0 yet that projection gets a gradient from the first step, which a closed scalar gate would block."""

    def __init__(self, config: dict):
        super().__init__()
        hidden, heads = config['hidden_size'], config['num_attention_heads']
        self.heads, self.head_dim = heads, hidden // heads
        self.norm = RMSNorm(hidden, config['rms_norm_eps'])
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def memory_keys_values(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, columns, _ = memory.shape
        key = self.k_proj(memory).view(batch, columns, self.heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(memory).view(batch, columns, self.heads, self.head_dim).transpose(1, 2)
        return key, value

    def forward(self, hidden: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                memory_attend_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.q_proj(self.norm(hidden)).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=memory_attend_mask)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden, intermediate = config['hidden_size'], config['intermediate_size']
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    """OLMo 3 normalizes each sublayer's output rather than its input; cross-attention sits between the two."""

    def __init__(self, config: dict):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.cross_attn = CrossAttention(config)
        self.mlp = MLP(config)
        self.post_feedforward_layernorm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])

    def forward(self, hidden, cos, sin, memory_kv, memory_attend_mask, cache=None):
        hidden = hidden + self.post_attention_layernorm(self.self_attn(hidden, cos, sin, cache))
        hidden = hidden + self.cross_attn(hidden, *memory_kv, memory_attend_mask)
        return hidden + self.post_feedforward_layernorm(self.mlp(hidden))


class TokenDecoder(nn.Module):
    """A pretrained subword LM that reads the OCR encoder through cross-attention. Its text is NFC, the
    form the LM was trained on; callers convert to and from DeepOCR's canonical form."""

    def __init__(self, config: dict, tokenizer_json: str, memory_dim: int):
        super().__init__()
        self.config = {key: config[key] for key in LM_CONFIG_KEYS}
        self.tokenizer_json = tokenizer_json
        self.tokenizer = Tokenizer.from_str(tokenizer_json)
        hidden = config['hidden_size']
        self.embed_tokens = nn.Embedding(config['vocab_size'], hidden)
        self.layers = nn.ModuleList(DecoderLayer(self.config) for _ in range(config['num_hidden_layers']))
        self.norm = RMSNorm(hidden, config['rms_norm_eps'])
        self.memory_proj = nn.Linear(memory_dim, hidden)
        head_dim = hidden // config['num_attention_heads']
        inverse_frequency = 1.0 / (config['rope_theta'] ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inverse_frequency', inverse_frequency, persistent=False)
        self.register_buffer('token_bytes', self._token_byte_lengths(config['vocab_size']), persistent=False)

    def _token_byte_lengths(self, vocab_size: int) -> torch.Tensor:
        """A byte-level BPE token spells one byte per character of its vocabulary string."""
        lengths = torch.zeros(vocab_size, dtype=torch.long)
        special = {token.content for token in self.tokenizer.get_added_tokens_decoder().values()}
        for token, index in self.tokenizer.get_vocab(with_added_tokens=False).items():
            if index < vocab_size and token not in special:
                lengths[index] = len(token)
        return lengths

    @classmethod
    def from_pretrained_lm(cls, lm_dir: Path, memory_dim: int) -> 'TokenDecoder':
        """Reads an OLMo 3 checkpoint directory (config.json, model.safetensors, tokenizer.json)."""
        lm_dir = Path(lm_dir)
        hf_config = json.loads((lm_dir / 'config.json').read_text(encoding='utf-8'))
        hf_config['rope_theta'] = hf_config.get('rope_parameters', {}).get('rope_theta', hf_config.get('rope_theta'))
        decoder = cls(hf_config, (lm_dir / 'tokenizer.json').read_text(encoding='utf-8'), memory_dim)
        state = {key.removeprefix('_orig_mod.').removeprefix('model.'): value
                 for key, value in load_file(lm_dir / 'model.safetensors').items()}
        state.pop('lm_head.weight', None)
        missing, unexpected = decoder.load_state_dict(state, strict=False)
        missing = [key for key in missing if not cls.is_new_parameter(key)]
        if missing or unexpected:
            raise RuntimeError(f'LM weights do not match the decoder: missing {missing[:5]}, unexpected {unexpected[:5]}')
        return decoder

    @staticmethod
    def is_new_parameter(name: str) -> bool:
        return any(marker in name for marker in NEW_PARAMETER_MARKERS)

    @property
    def vocab_size(self) -> int:
        return self.config['vocab_size']

    def rotary(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = positions.float()[:, None] * self.inverse_frequency[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def encode_texts(self, texts: Sequence[str]) -> List[List[int]]:
        normalized = [unicodedata.normalize('NFC', text) for text in texts]
        return [encoding.ids for encoding in self.tokenizer.encode_batch(normalized, add_special_tokens=False)]

    def build_io(self, texts: Sequence[str], device: torch.device
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Start-prefixed inputs and end-suffixed targets, both [B, L] and padded, plus the padding mask."""
        sequences = self.encode_texts(texts)
        length = max(len(ids) for ids in sequences) + 1 if sequences else 1
        inputs = torch.full((len(sequences), length), PAD_TOKEN_ID, dtype=torch.long)
        targets = torch.full((len(sequences), length), PAD_TOKEN_ID, dtype=torch.long)
        for row, ids in enumerate(sequences):
            inputs[row, :len(ids) + 1] = torch.tensor([START_TOKEN_ID] + ids)
            targets[row, :len(ids) + 1] = torch.tensor(ids + [END_TOKEN_ID])
        return inputs.to(device), targets.to(device), (targets == PAD_TOKEN_ID).to(device)

    def corrupt_inputs(self, inputs: torch.Tensor, padding: torch.Tensor, rate: float) -> torch.Tensor:
        """Swaps a share of the teacher-forced text tokens for other text tokens, so the next token
        cannot be predicted from the text alone and has to be read from the image."""
        text_tokens = self.vocab_size - (PAD_TOKEN_ID + 1)
        chosen = (torch.rand(inputs.shape, device=inputs.device) < rate) & ~padding
        chosen[:, 0] = False
        offset = torch.randint(1, text_tokens, inputs.shape, device=inputs.device)
        replaced = (inputs - (PAD_TOKEN_ID + 1) + offset) % text_tokens + PAD_TOKEN_ID + 1
        return torch.where(chosen, replaced, inputs)

    def _memory(self, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor]):
        """Encoder output is [S, B, D] with True marking padded columns; returns per-layer keys and values."""
        projected = self.memory_proj(memory.transpose(0, 1))
        attend_mask = None
        if memory_key_padding_mask is not None:
            attend_mask = ~memory_key_padding_mask[:, None, None, :]
        return [layer.cross_attn.memory_keys_values(projected) for layer in self.layers], attend_mask

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.norm(hidden), self.embed_tokens.weight)

    def forward(self, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor],
                input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits [B, L, V]. Padding sits at the end, so the causal mask alone keeps it out."""
        memory_kv, attend_mask = self._memory(memory, memory_key_padding_mask)
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary(torch.arange(input_ids.size(1), device=input_ids.device), hidden.dtype)
        for layer, layer_kv in zip(self.layers, memory_kv):
            hidden = layer(hidden, cos, sin, layer_kv, attend_mask)
        return self._logits(hidden)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor],
                 max_len: int, max_bytes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy decoding with a key-value cache. Returns [B, T] ids including the start token. A row
        ends instead of taking a token that would carry it past its max_bytes of text."""
        batch = memory.size(1)
        memory_kv, attend_mask = self._memory(memory, memory_key_padding_mask)
        caches = [[] for _ in self.layers]
        generated = torch.full((batch, 1), START_TOKEN_ID, dtype=torch.long, device=memory.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        written = torch.zeros(batch, dtype=torch.long, device=memory.device)
        for step in range(max_len):
            hidden = self.embed_tokens(generated[:, -1:])
            cos, sin = self.rotary(torch.tensor([step], device=memory.device), hidden.dtype)
            for layer, layer_kv, cache in zip(self.layers, memory_kv, caches):
                hidden = layer(hidden, cos, sin, layer_kv, attend_mask, cache)
            next_token = self._logits(hidden[:, -1]).argmax(dim=-1)
            if max_bytes is not None:
                written = written + self.token_bytes[next_token]
                next_token = torch.where(written > max_bytes, torch.full_like(next_token, END_TOKEN_ID), next_token)
            next_token = torch.where(finished, torch.full_like(next_token, PAD_TOKEN_ID), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token == END_TOKEN_ID
            if bool(finished.all()):
                break
        return generated

    @torch.no_grad()
    def beam_search(self, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor], max_len: int,
                    beam: int, max_bytes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search with a key-value cache. Returns [B, beam, T] ids including the start token and each
        beam's summed log probability [B, beam], best first. A beam never takes a token that would carry
        it past its line's max_bytes of text."""
        batch, device = memory.size(1), memory.device
        memory_kv, attend_mask = self._memory(memory, memory_key_padding_mask)
        memory_kv = [(key.repeat_interleave(beam, 0), value.repeat_interleave(beam, 0)) for key, value in memory_kv]
        if attend_mask is not None:
            attend_mask = attend_mask.repeat_interleave(beam, 0)
        budget = max_bytes.to(device).repeat_interleave(beam) if max_bytes is not None else None
        caches = [[] for _ in self.layers]
        sequences = torch.full((batch * beam, 1), START_TOKEN_ID, dtype=torch.long, device=device)
        scores = torch.full((batch, beam), float('-inf'), device=device)
        scores[:, 0] = 0.0
        finished = torch.zeros(batch * beam, dtype=torch.bool, device=device)
        written = torch.zeros(batch * beam, dtype=torch.long, device=device)
        line_offsets = torch.arange(batch, device=device)[:, None] * beam
        for step in range(max_len):
            hidden = self.embed_tokens(sequences[:, -1:])
            cos, sin = self.rotary(torch.tensor([step], device=device), hidden.dtype)
            for layer, layer_kv, cache in zip(self.layers, memory_kv, caches):
                hidden = layer(hidden, cos, sin, layer_kv, attend_mask, cache)
            log_probs = F.log_softmax(self._logits(hidden[:, -1]).float(), dim=-1)
            log_probs[:, PAD_TOKEN_ID] = float('-inf')
            if budget is not None:
                over = written[:, None] + self.token_bytes[None, :] > budget[:, None]
                over[:, END_TOKEN_ID] = False
                log_probs = log_probs.masked_fill(over, float('-inf'))
            # a finished beam continues with padding at no cost, so it keeps its place in the ranking
            log_probs[finished] = float('-inf')
            log_probs[finished, PAD_TOKEN_ID] = 0.0
            vocab = log_probs.size(-1)
            scores, flat = (scores.reshape(-1, 1) + log_probs).reshape(batch, beam * vocab).topk(beam, dim=-1)
            origin = (flat // vocab + line_offsets).reshape(-1)
            token = (flat % vocab).reshape(-1)
            for cache in caches:
                cache[:] = [tensor.index_select(0, origin) for tensor in cache]
            sequences = torch.cat([sequences[origin], token[:, None]], dim=1)
            written = written[origin] + self.token_bytes[token]
            finished = finished[origin] | (token == END_TOKEN_ID)
            if bool(finished.all()):
                break
        return sequences.reshape(batch, beam, -1), scores

    def ids_to_text(self, generated: torch.Tensor) -> List[str]:
        texts = []
        for row in generated.tolist():
            row = row[1:]
            if END_TOKEN_ID in row:
                row = row[:row.index(END_TOKEN_ID)]
            texts.append(self.tokenizer.decode([token for token in row if token != PAD_TOKEN_ID]))
        return texts
