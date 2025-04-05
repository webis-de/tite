from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.ops.activations import SwiGLUFunction
except ImportError:
    flash_attn_varlen_func = None
    SwiGLUFunction = None
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .pool import PackedAvgPool1d, PackedMetaData, compute_output_shape
from .rope import EagerRotaryPositionalEmbeddings, TritonRotaryPositionalEmbeddings


class RMSNorm(torch.nn.Module):
    """Llama2 RMSNorm implementation"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _init_weights(self, reset_params: bool = False):
        torch.nn.init.ones_(self.weight)


class SwiGLU(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        if SwiGLUFunction is None:
            return torch.nn.functional.silu(x) * y
        return SwiGLUFunction.apply(x, y)


ACT2FN["swiglu"] = SwiGLU
NORM_MAP = {"rms": RMSNorm, "layer": torch.nn.LayerNorm}


def compute_output_shapes(
    input_shape: int, kernel_sizes: Tuple[int | None, ...], strides: Tuple[int | None, ...]
) -> List[int]:
    output_shapes = [input_shape]
    for k, s in zip(kernel_sizes, strides):
        output_shapes.append(compute_output_shape(output_shapes[-1], k, s))
    return output_shapes


@dataclass
class TiteModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor
    hidden_states: List[torch.Tensor] | None = None
    attentions: List[torch.Tensor] | None = None


class TiteConfig(PretrainedConfig):

    model_type = "tite"

    def __init__(
        self,
        vocab_size: int = 30522,
        num_hidden_layers: int = 12,
        hidden_sizes: Tuple[int, ...] | int = 768,
        num_attention_heads: Tuple[int, ...] | int = 12,
        intermediate_sizes: Tuple[int, ...] | int = 3072,
        kernel_sizes: Tuple[int | None, ...] | int | None = None,
        strides: Tuple[int | None, ...] | int | None = None,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        hidden_act: str = "gelu_pytorch_tanh",
        positional_embedding_type: Literal["absolute", "rotary", "alibi"] = "rotary",
        rotary_interleaved: bool = False,
        norm_location: Literal["pre", "post"] = "pre",
        norm_type: Literal["rms", "layer"] = "rms",
        pooling_implementation: Literal["eager", "triton"] = "triton",
        rope_implementation: Literal["eager", "triton"] = "triton",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,) * num_hidden_layers
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * num_hidden_layers
        if isinstance(intermediate_sizes, int):
            intermediate_sizes = (intermediate_sizes,) * num_hidden_layers
        if isinstance(kernel_sizes, int) or kernel_sizes is None:
            kernel_sizes = (kernel_sizes,) * num_hidden_layers
        if isinstance(strides, int) or strides is None:
            strides = (strides,) * num_hidden_layers
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.intermediate_sizes = intermediate_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.positional_embedding_type = positional_embedding_type
        self.rotary_interleaved = rotary_interleaved
        self.norm_location = norm_location
        self.norm_type = norm_type
        self.pooling_implementation = pooling_implementation
        self.rope_implementation = rope_implementation

        iterator = zip(
            [
                "hidden_sizes",
                "num_attention_heads",
                "intermediate_sizes",
                "kernel_sizes",
                "strides",
            ],
            [hidden_sizes, num_attention_heads, intermediate_sizes, kernel_sizes, strides],
        )
        for name, setting in iterator:
            if len(setting) != num_hidden_layers:
                raise ValueError(
                    f"Length of {name} does not match num_hidden_layers. "
                    f"Expected {num_hidden_layers}, got {len(setting)}."
                )

        if self.output_shapes[0] != self.output_shapes[-1] and self.output_shapes[-1] != 1:
            raise ValueError(
                "kernel_sizes and strides are set, but do not reduce the maximum sequence length to 1. "
                "Please adjust kernel_size and stride."
            )

    @property
    def _attn_implementation(self):
        # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "flash_attention_2"
            else:
                return self._attn_implementation_internal
        else:
            return "flash_attention_2"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value

    @property
    def output_shapes(self) -> List[int]:
        return compute_output_shapes(self.max_position_embeddings, self.kernel_sizes, self.strides)

    @property
    def hidden_size(self) -> int:
        return self.hidden_sizes[-1]


class TiteEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = NORM_MAP[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor, position_idcs: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_idcs)
            embeddings = embeddings + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TiteAttention(torch.nn.Module):

    ATTENTION_FUNCTIONS = {
        "flash_attention_2": "flash_attn_forward",
        "sdpa": "sdpa_forward",
        "eager": "eager_forward",
    }

    ROPE_CLASSES = {
        "eager": EagerRotaryPositionalEmbeddings,
        "triton": TritonRotaryPositionalEmbeddings,
    }

    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        if config.hidden_sizes[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_sizes}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        self.config = config

        to_hidden_size = config.hidden_sizes[layer_idx]
        from_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        self.hidden_size_diff = to_hidden_size - from_hidden_size
        if self.hidden_size_diff < 0:
            raise ValueError("Only upscaling the hidden size is supported")

        if config.norm_location == "pre" and layer_idx == 0:
            self.norm = torch.nn.Identity()
        else:
            self.norm = NORM_MAP[config.norm_type](
                from_hidden_size if config.norm_location == "pre" else to_hidden_size, eps=config.layer_norm_eps
            )

        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(to_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(from_hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(from_hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(from_hidden_size, self.all_head_size)

        self.dense = torch.nn.Linear(to_hidden_size, to_hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        kernel_size = config.kernel_sizes[layer_idx]
        stride = config.strides[layer_idx]
        self.pooling = None
        if kernel_size is not None and stride is not None:
            self.pooling = PackedAvgPool1d(kernel_size, stride, config.pooling_implementation)

        if config.positional_embedding_type == "rotary":
            self.rope = self.ROPE_CLASSES[config.rope_implementation](
                self.attention_head_size, interleaved=config.rotary_interleaved
            )
            self.alibi = None
        elif config.positional_embedding_type == "alibi":
            self.rope = None
            x = (2**8) ** (1 / num_attention_heads)
            self.register_buffer(
                "alibi", torch.tensor([1 / x ** (i + 1) for i in range(num_attention_heads)]), persistent=False
            )
        else:
            self.rope = None
            self.alibi = None

        if config._attn_implementation == "flash_attention_2" and flash_attn_varlen_func is None:
            raise ValueError(
                "Flash Attention 2 is not installed. Please install or use a different attention implementation"
            )
        self.attention_function = getattr(self, self.ATTENTION_FUNCTIONS[config._attn_implementation])

    @torch.compiler.disable
    def flash_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_packed_meta_data: PackedMetaData,
        packed_meta_data: PackedMetaData,
        output_attention: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        if flash_attn_varlen_func is None:
            raise ValueError(
                "Flash Attention 2 is not installed. Please install or use a different attention implementation"
            )
        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            query_packed_meta_data.cu_seq_lens,
            packed_meta_data.cu_seq_lens,
            query_packed_meta_data.max_seq_len,
            packed_meta_data.max_seq_len,
            alibi_slopes=self.alibi,
        )
        return attn_output, None

    def _unpad(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        pad_x = torch.zeros(
            len(packed_meta_data.seq_lens), packed_meta_data.max_seq_len, *x.shape[1:], dtype=x.dtype, device=x.device
        )
        pad_x[packed_meta_data.idcs] = x
        return pad_x

    def _repad(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        packed_x = x[packed_meta_data.idcs]
        return packed_x

    def sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_packed_meta_data: PackedMetaData,
        packed_meta_data: PackedMetaData,
        output_attention: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        if self.alibi is not None:
            raise ValueError("ALiBi is not supported with SDPA")
        pad_query = self._unpad(query, query_packed_meta_data).transpose(1, 2)
        pad_key = self._unpad(key, packed_meta_data).transpose(1, 2)
        pad_value = self._unpad(value, packed_meta_data).transpose(1, 2)
        query_mask = self._unpad(
            torch.ones(query.shape[0], device=pad_query.device, dtype=torch.bool), query_packed_meta_data
        )
        key_mask = self._unpad(torch.ones(key.shape[0], device=pad_key.device, dtype=torch.bool), packed_meta_data)
        mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(-2)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            pad_query, pad_key, pad_value, attn_mask=mask[:, None]
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = self._repad(attn_output, query_packed_meta_data)
        return attn_output, None

    def eager_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_packed_meta_data: PackedMetaData,
        packed_meta_data: PackedMetaData,
        output_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.alibi is not None:
            raise ValueError("ALiBi is not supported with eager attention")
        pad_query = self._unpad(query, query_packed_meta_data).transpose(1, 2)
        pad_key = self._unpad(key, packed_meta_data).transpose(1, 2)
        pad_value = self._unpad(value, packed_meta_data).transpose(1, 2)
        query_mask = self._unpad(
            torch.ones(query.shape[0], device=pad_query.device, dtype=torch.bool), query_packed_meta_data
        )
        key_mask = self._unpad(torch.ones(key.shape[0], device=pad_key.device, dtype=torch.bool), packed_meta_data)
        mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(-2)

        attn_values = pad_query @ pad_key.transpose(-2, -1)
        attn_values = attn_values / (self.attention_head_size**0.5)
        attn_values = attn_values.masked_fill(~mask[:, None], torch.finfo(attn_values.dtype).min)
        attn_probs = torch.nn.functional.softmax(attn_values, dim=-1)
        attn_output = attn_probs @ pad_value
        attn_output = attn_output.transpose(1, 2)
        attn_output = self._repad(attn_output, query_packed_meta_data)
        if output_attention:
            return attn_output, attn_probs
        return attn_output, None

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData, output_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None, PackedMetaData]:
        if self.alibi is not None:
            self.alibi = self.alibi.to(torch.float32)

        if self.pooling is None:
            input_hidden_states = hidden_states
            output_packed_meta_data = packed_meta_data
            if self.config.norm_location == "pre":
                hidden_states = self.norm(hidden_states)
        else:
            query_hidden_states, output_packed_meta_data = self.pooling(hidden_states, packed_meta_data)
            input_hidden_states = query_hidden_states
            if self.config.norm_location == "pre":
                hidden_states = self.norm(hidden_states)
                query_hidden_states = self.norm(query_hidden_states)

        value = self.value(hidden_states)
        if packed_meta_data.max_seq_len == 1:
            hidden_states, attn_probs = value, None
        else:
            query = self.query(query_hidden_states)
            key = self.key(hidden_states)

            query = rearrange(query, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
            key = rearrange(key, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
            value = rearrange(value, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)

            if self.rope is not None:
                query = self.rope(query, output_packed_meta_data)
                key = self.rope(key, packed_meta_data)

            hidden_states, attn_probs = self.attention_function(
                query, key, value, output_packed_meta_data, packed_meta_data, output_attention
            )

            hidden_states = rearrange(hidden_states, "t h d -> t (h d)")

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.hidden_size_diff:
            input_hidden_states = torch.nn.functional.pad(input_hidden_states, (0, self.hidden_size_diff))
        hidden_states = hidden_states + input_hidden_states

        if self.config.norm_location == "post":
            hidden_states = self.norm(hidden_states)

        return hidden_states, attn_probs, output_packed_meta_data


class TiteMLP(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        intermediate_size = config.intermediate_sizes[layer_idx]
        self.config = config
        self.norm = NORM_MAP[config.norm_type](hidden_size, eps=config.layer_norm_eps)
        self.intermediate_dense = torch.nn.Linear(
            hidden_size, intermediate_size * 2 if config.hidden_act == "swiglu" else intermediate_size
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.out_dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_hidden_states = hidden_states
        if self.config.norm_location == "pre":
            hidden_states = self.norm(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.out_dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_hidden_states
        if self.config.norm_location == "post":
            hidden_states = self.norm(hidden_states)
        return hidden_states


class TiteLayer(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TiteAttention(config, layer_idx)
        self.mlp = TiteMLP(config, layer_idx)

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData, output_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None, PackedMetaData]:
        hidden_states, attention, packed_meta_data = self.attention(hidden_states, packed_meta_data, output_attention)
        layer_output = self.mlp(hidden_states)
        return layer_output, attention, packed_meta_data


class TiteEncoder(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(
            [TiteLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.norm_location == "pre":
            self.norm = NORM_MAP[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = torch.nn.Identity()
        if config.hidden_sizes[0] != config.hidden_sizes[-1]:
            self.downscale = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[0]),
                NORM_MAP[config.norm_type](config.hidden_sizes[0], eps=config.layer_norm_eps),
            )
        else:
            self.downscale = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        packed_meta_data: PackedMetaData,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | None, List[torch.Tensor] | None]:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions: List[torch.Tensor] | None = [] if output_attentions else None
        hidden_states = self.downscale(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attention, packed_meta_data = layer(hidden_states, packed_meta_data, output_attentions)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            if all_attentions is not None:
                all_attentions.append(attention)
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_hidden_states, all_attentions


class PreTrainingHead(torch.nn.Module, ABC):

    @abstractmethod
    def get_labels(self, *args, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def forward(
        self,
        output: TiteModelOutput,
        original_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...


class LMPredictionHead(torch.nn.Module):
    def __init__(self, config: TiteConfig, decoder: torch.nn.Linear | None = None):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.norm = NORM_MAP[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        if decoder is None:
            self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.decoder = decoder
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class BOWLMHead(PreTrainingHead):

    def __init__(self, config: TiteConfig, lm_decoder: torch.nn.Linear | None = None):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.lm_head = LMPredictionHead(config, lm_decoder)

    def forward(self, output: TiteModelOutput, *args, **kwargs) -> torch.Tensor:
        return self.lm_head(output.last_hidden_state)

    def get_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        targets = torch.zeros(input_ids.shape[0], self.vocab_size, device=input_ids.device)
        input_ids = input_ids.clone()
        input_ids[special_tokens_mask.bool()] = self.pad_token_id
        targets = targets.scatter(1, input_ids, 1)
        targets[:, self.pad_token_id] = -100
        return targets

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        labels = labels.view(-1)
        mask = labels == -100
        logits = logits[~mask]
        labels = labels[~mask]
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


class EnhancedMaskedAttention(TiteAttention):
    def __init__(self, config: TiteConfig, mask_strategy: float | Literal["causal"] = 0.0):
        super().__init__(config, layer_idx=-1)
        self.mask_strategy = mask_strategy

    def forward(  # type: ignore
        self, query_hidden_states: torch.Tensor, key_value_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, query_seq_len = query_hidden_states.shape[:2]
        key_value_seq_len = key_value_hidden_states.shape[1]
        if self.alibi is not None:
            self.alibi = self.alibi.to(torch.float32)

        input_hidden_states = query_hidden_states

        if self.config.norm_location == "pre":
            query_hidden_states = self.norm(query_hidden_states)
            key_value_hidden_states = self.norm(key_value_hidden_states)

        query = self.query(query_hidden_states)
        key = self.key(key_value_hidden_states)
        value = self.value(key_value_hidden_states)

        query = rearrange(query, "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size)
        key = rearrange(key, "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size)

        if self.mask_strategy == "causal":
            enhanced_decoding_mask = torch.tril(
                torch.ones((query_seq_len, key_value_seq_len), device=input_hidden_states.device, dtype=torch.bool)
            )[None].expand(input_hidden_states.shape[0], -1, -1)
        elif isinstance(self.mask_strategy, float):
            enhanced_decoding_mask = (
                torch.rand((batch_size, query_seq_len, key_value_seq_len), device=input_hidden_states.device)
                >= self.mask_strategy
            )
            # set diagonal to False
            enhanced_decoding_mask[:, range(query_seq_len), range(query_seq_len)] = False
            enhanced_decoding_mask = enhanced_decoding_mask.logical_and(attention_mask[..., None])
        else:
            raise ValueError("invalid mask strategy")

        if self.rope is not None:
            query_packed_meta_data = PackedMetaData.from_attention_mask(torch.ones_like(attention_mask))
            if query_hidden_states.shape[1] == key_value_hidden_states.shape[1]:
                key_packed_meta_data = query_packed_meta_data
            else:
                key_packed_meta_data = PackedMetaData.from_attention_mask(
                    torch.ones(batch_size, key_value_seq_len, device=input_hidden_states.device, dtype=torch.bool)
                )
            query = rearrange(
                self.rope(rearrange(query, ("b h s d -> (b s) h d")), query_packed_meta_data),
                "(b s) h d -> b h s d",
                b=batch_size,
            )
            key = rearrange(
                self.rope(rearrange(key, ("b h s d -> (b s) h d")), key_packed_meta_data),
                "(b s) h d -> b h s d",
                b=batch_size,
            )

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=enhanced_decoding_mask[:, None]
        )

        hidden_states = rearrange(
            hidden_states,
            "b h s d -> b s (h d)",
            b=batch_size,
            h=self.num_attention_heads,
            s=query_seq_len,
            d=self.attention_head_size,
        )

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + query_hidden_states
        if self.config.norm_location == "post":
            hidden_states = self.norm(hidden_states)
        return hidden_states


class EnhancedLMHead(PreTrainingHead):

    def __init__(
        self,
        config: TiteConfig,
        embeddings: TiteEmbeddings,
        lm_decoder: torch.nn.Linear | None,
        mask_strategy: float | Literal["causal"] = 0.5,
    ):
        super().__init__()
        config = deepcopy(config)
        config.num_hidden_layers = 1
        config.hidden_sizes = config.hidden_sizes[-1:]
        config.num_attention_heads = config.num_attention_heads[-1:]
        config.intermediate_sizes = config.intermediate_sizes[-1:]
        config.strides = (None,)
        config.kernel_sizes = (None,)

        self.vocab_size = config.vocab_size
        self.embeddings = embeddings
        self.lm_head = LMPredictionHead(config, lm_decoder)
        self.enhanced_attention = EnhancedMaskedAttention(config, mask_strategy)
        self.mask_strategy = mask_strategy

        if config.norm_location == "pre":
            self.norm = NORM_MAP[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = torch.nn.Identity()
        self.mlp = TiteMLP(config, -1)

    def forward(
        self,
        output: TiteModelOutput,
        original_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_idcs = torch.arange(original_input_ids.shape[1], device=original_input_ids.device)[None].expand_as(
            original_input_ids
        )
        token_embeddings = self.embeddings(original_input_ids, position_idcs)

        query_hidden_states = output.last_hidden_state.expand_as(token_embeddings)

        key_value_hidden_states = token_embeddings

        hidden_states = self.enhanced_attention(query_hidden_states, key_value_hidden_states, attention_mask)

        if self.mask_strategy == "causal":
            hidden_states = hidden_states[:, :-1]

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def get_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        targets = torch.where(special_tokens_mask.bool(), -100, input_ids)
        if self.mask_strategy == "causal":
            targets = targets[:, 1:]
        return targets

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        labels = labels.reshape(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class TitePreTrainedModel(PreTrainedModel):
    config_class = TiteConfig
    base_model_prefix = "tite"

    # Flash Attention 2 support
    _supports_flash_attn_2 = True

    # SDPA support
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # https://github.com/huggingface/transformers/blob/3d7c2f9dea45338b7ebcd459b452e2fad7abfa1f/src/transformers/models/bert/modeling_bert.py#L835
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (torch.nn.LayerNorm, RMSNorm)):
            if isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TiteModel(TitePreTrainedModel):

    _no_split_modules = ["TiteEmbeddings", "TiteLayer"]

    def __init__(self, config: TiteConfig):
        super().__init__(config)
        self.config: TiteConfig

        self.embeddings = TiteEmbeddings(config)
        self.encoder = TiteEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> TiteModelOutput:
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        batch_size, seq_len = input_ids.shape
        packed_meta_data = PackedMetaData.from_attention_mask(attention_mask)
        idcs = packed_meta_data.idcs
        assert idcs is not None
        if self.config._attn_implementation == "flash_attention_2" and self.config.pooling_implementation == "triton":
            packed_meta_data.idcs = None
        input_ids = input_ids[idcs]
        hidden_states = self.embeddings(input_ids, idcs[1])
        hidden_states, all_hidden_states, all_attentions = self.encoder(
            hidden_states, packed_meta_data, output_hidden_states, output_attentions
        )
        if hidden_states.shape[0] == packed_meta_data.seq_lens.shape[0]:
            hidden_states = hidden_states.unsqueeze(1)
        else:
            repad_hidden_states = torch.zeros(
                batch_size, seq_len, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
            )
            repad_hidden_states[idcs] = hidden_states
            hidden_states = repad_hidden_states.view(batch_size, seq_len, -1)
        return TiteModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    @staticmethod
    def _update_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in ("pooler.dense.weight", "pooler.dense.bias", "embeddings.token_type_embeddings.weight"):
                continue
            new_key = key
            new_key = new_key.replace("layer.", "layers.")
            new_key = new_key.replace("LayerNorm", "norm")
            new_key = new_key.replace("attention.self", "attention")
            new_key = new_key.replace("attention.output", "attention")
            new_key = new_key.replace("intermediate.dense", "mlp.intermediate_dense")
            new_key = new_key.replace("output.dense", "mlp.out_dense")
            new_key = new_key.replace("output", "mlp")
            if "Wqkv" in key:
                q, k, v = value.chunk(3, dim=0)
                new_state_dict[new_key.replace("Wqkv", "query")] = q
                new_state_dict[new_key.replace("Wqkv", "key")] = k
                new_state_dict[new_key.replace("Wqkv", "value")] = v
            else:
                new_state_dict[new_key] = value
        return new_state_dict

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
    ):
        new_state_dict = cls._update_state_dict(state_dict)
        loaded_keys = list(new_state_dict.keys())
        return super()._load_pretrained_model(
            model, new_state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )


@dataclass
class TitePreTrainingOutput(TiteModelOutput):
    losses: Dict[str, torch.Tensor] | None = None


class TiteForPreTraining(TitePreTrainedModel):

    _tied_weights_keys = [
        "tite.embeddings.word_embeddings.weight",
        "tite.embeddings.word_embeddings.bias",
        "lm_decoder.weight",
        "lm_decoder.bias",
        "heads.enhanced_masked_auto_encoding.lm_head.decoder.weight",
        "heads.enhanced_masked_auto_encoding.lm_head.decoder.bias",
        "heads.bow_auto_encoding.lm_head.decoder.weight",
        "heads.bow_auto_encoding.lm_head.decoder.bias",
    ]

    def __init__(
        self,
        config: TiteConfig,
        enhanced_masked_auto_encoding: bool = True,
        enhanced_causal_auto_encoding: bool = True,
        bow_auto_encoding: bool = True,
        mask_prob: float = 0.5,
    ):
        super().__init__(config)
        self.config: TiteConfig

        self.tite = TiteModel(config)
        self.heads = torch.nn.ModuleDict()
        self.lm_decoder = None

        lm_decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if enhanced_masked_auto_encoding:
            self.heads["enhanced_masked_auto_encoding"] = EnhancedLMHead(
                config, self.tite.embeddings, lm_decoder, mask_strategy=mask_prob
            )
            self.lm_decoder = lm_decoder
        if bow_auto_encoding:
            self.heads["bow_auto_encoding"] = BOWLMHead(config, lm_decoder)
            self.lm_decoder = lm_decoder
        if enhanced_causal_auto_encoding:
            self.heads["enhanced_causal_auto_encoding"] = EnhancedLMHead(
                config, self.tite.embeddings, lm_decoder, mask_strategy="causal"
            )

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        original_input_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> TitePreTrainingOutput:
        output = self.tite(input_ids, attention_mask, output_hidden_states=True, output_attentions=output_attentions)
        losses = None
        if labels is not None:
            losses = {}
            for task, head in self.heads.items():
                logits = head(
                    output=output,
                    original_input_ids=original_input_ids,
                    attention_mask=attention_mask,
                )
                losses[task] = head.compute_loss(logits, labels[task])
        return TitePreTrainingOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions,
            losses=losses,
        )
