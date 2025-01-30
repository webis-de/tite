from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from flash_attn.ops.activations import SwiGLUFunction
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .pool import PackedAvgPool1d, PackedMetaData, compute_output_shape
from .rope import RotaryPositionalEmbeddings


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
        return SwiGLUFunction.apply(x, y)


ACT2FN["swiglu"] = SwiGLU


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
    hidden_states: Tuple[torch.Tensor, ...] | None = None


class TiteConfig(PretrainedConfig):

    model_type = "tite"

    def __init__(
        self,
        vocab_size: int = 30522,
        num_hidden_layers: int = 12,
        hidden_sizes: Tuple[int, ...] = (768,) * 12,
        num_attention_heads: Tuple[int, ...] = (12,) * 12,
        intermediate_sizes: Tuple[int, ...] = (3072,) * 12,
        kernel_sizes: Tuple[int | None, ...] = (None,) * 12,
        strides: Tuple[int | None, ...] = (None,) * 12,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        hidden_act: str = "swiglu",
        positional_embedding_type: Literal["absolute", "rotary", "alibi"] = "rotary",
        upscale_hidden_sizes: bool = False,
        pooling_location: Literal["pre", "attention", "post"] = "attention",
        pooling_strategy: Literal["mean_conv", "select"] = "mean_conv",
        pooling_implementation: Literal["unfold", "sum_pool2d"] = "unfold",
        rotary_interleaved: bool = False,
        pre_norm: bool = True,
        norm_type: Literal["rms", "layer"] = "rms",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
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
        self.upscale_hidden_sizes = upscale_hidden_sizes
        self.pooling_location = pooling_location
        self.pooling_strategy = pooling_strategy
        self.pooling_implementation = pooling_implementation
        self.rotary_interleaved = rotary_interleaved
        self.pre_norm = pre_norm
        self.norm_type = norm_type

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
    def output_shapes(self) -> List[int]:
        return compute_output_shapes(self.max_position_embeddings, self.kernel_sizes, self.strides)

    @property
    def hidden_size(self) -> int:
        return self.hidden_sizes[-1]


class TiteModel(PreTrainedModel):
    config_class = TiteConfig

    def __init__(self, config: TiteConfig):
        super().__init__(config)

        self.embeddings = TiteEmbeddings(config)
        self.encoder = TiteEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def tie_decoder_weights(self, output_embeddings: torch.nn.Module):
        self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> TiteModelOutput:
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            idcs = attention_mask.nonzero(as_tuple=True)
            input_ids = input_ids[idcs]
            attention_mask = attention_mask.bool()
            seq_lens = attention_mask.sum(-1).int()
            max_seq_len = int(seq_lens.max().item())
            cu_seq_lens = torch.zeros(seq_lens.shape[0] + 1, dtype=seq_lens.dtype, device=seq_lens.device)
            cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0, dtype=seq_lens.dtype)
        hidden_states = self.embeddings(input_ids, idcs[1])
        packed_meta_data = PackedMetaData(seq_lens, cu_seq_lens, max_seq_len)
        hidden_states, all_hidden_states = self.encoder(hidden_states, packed_meta_data, output_hidden_states)
        if hidden_states.shape[0] == seq_lens.shape[0]:
            hidden_states = hidden_states.unsqueeze(1)
        else:
            repad_hidden_states = torch.zeros(
                batch_size, seq_len, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
            )
            repad_hidden_states[idcs] = hidden_states
            hidden_states = repad_hidden_states.view(batch_size, seq_len, -1)
        return TiteModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class TiteEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_size = config.hidden_sizes[0]
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, hidden_size)
        if config.norm_type == "layer":
            self.norm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        elif config.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {config.norm_type}")
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    # @torch.compile(dynamic=True)
    def forward(self, input_ids: torch.Tensor, position_idcs: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_idcs)
            embeddings = embeddings + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TiteAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        if config.hidden_sizes[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_sizes}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        self.config = config

        to_hidden_size = config.hidden_sizes[layer_idx]
        if config.upscale_hidden_sizes:
            from_hidden_size = to_hidden_size
        else:
            from_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]

        if config.pre_norm:
            if layer_idx == 0:
                self.norm = torch.nn.Identity()
            else:
                self.norm = torch.nn.LayerNorm(from_hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = torch.nn.LayerNorm(to_hidden_size, eps=config.layer_norm_eps)
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
            self.pooling = PackedAvgPool1d(kernel_size, stride)

        if config.positional_embedding_type == "rotary":
            self.rope = RotaryPositionalEmbeddings(
                self.attention_head_size, config.max_position_embeddings, config.rotary_interleaved
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

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData
    ) -> Tuple[torch.Tensor, PackedMetaData]:
        if self.alibi is not None:
            self.alibi = self.alibi.to(torch.float32)

        if self.pooling is not None and self.config.pooling_location == "pre":
            hidden_states, packed_meta_data = self.pooling(hidden_states, packed_meta_data)

        if self.config.pre_norm:
            hidden_states = self.norm(hidden_states)

        value = self.value(hidden_states)
        if packed_meta_data.max_seq_len == 1:
            return value, packed_meta_data

        query = self.query(hidden_states)
        key = self.key(hidden_states)

        query_packed_meta_data = packed_meta_data
        if self.pooling is not None and self.config.pooling_location == "attention":
            query, query_packed_meta_data = self.pooling(query, packed_meta_data)

        query = rearrange(query, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
        key = rearrange(key, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
        value = rearrange(value, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)

        if self.rope is not None:
            query = self.rope(query, query_packed_meta_data.cu_seq_lens, query_packed_meta_data.max_seq_len)
            key = self.rope(key, packed_meta_data.cu_seq_lens, packed_meta_data.max_seq_len)
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

        attn_output = rearrange(attn_output, "t h d -> t (h d)")

        attn_output = self.dense(attn_output)
        attn_output = self.dropout(attn_output)

        if self.pooling is not None and self.config.pooling_location == "attention":
            hidden_states, packed_meta_data = self.pooling(hidden_states, packed_meta_data)

        attn_output[..., : hidden_states.shape[-1]] = attn_output[..., : hidden_states.shape[-1]] + hidden_states

        if self.pooling is not None and self.config.pooling_location == "post":
            attn_output, packed_meta_data = self.pooling(attn_output, packed_meta_data)

        if not self.config.pre_norm:
            hidden_states = self.norm(attn_output)
        return attn_output, packed_meta_data


class TiteMLP(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        intermediate_size = config.intermediate_sizes[layer_idx]
        self.config = config
        if config.norm_type == "layer":
            self.norm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        elif config.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {config.norm_type}")
        self.intermediate_dense = torch.nn.Linear(
            hidden_size, intermediate_size * 2 if config.hidden_act == "swiglu" else intermediate_size
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.out_dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    # @torch.compile(dynamic=True)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mlp_output = hidden_states
        if self.config.pre_norm:
            mlp_output = self.norm(mlp_output)
        mlp_output = self.intermediate_dense(mlp_output)
        mlp_output = self.intermediate_act_fn(mlp_output)
        mlp_output = self.out_dense(mlp_output)
        mlp_output = self.dropout(mlp_output)
        mlp_output = mlp_output + hidden_states
        if not self.config.pre_norm:
            mlp_output = self.norm(mlp_output)
        return mlp_output


class TiteUpscale(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        old_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        self.upscale_layer = torch.nn.Linear(old_hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    # @torch.compile(dynamic=True)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upscale_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TiteLayer(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TiteAttention(config, layer_idx)
        self.mlp = TiteMLP(config, layer_idx)

        hidden_size = config.hidden_sizes[layer_idx]
        old_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        if config.upscale_hidden_sizes and old_hidden_size != hidden_size:
            self.upscale_layer = TiteUpscale(config, layer_idx)
        else:
            self.upscale_layer = torch.nn.Identity()

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData
    ) -> Tuple[torch.Tensor, PackedMetaData]:
        if self.upscale_layer is not None:
            hidden_states = self.upscale_layer(hidden_states)
        hidden_states, packed_meta_data = self.attention(hidden_states, packed_meta_data)
        layer_output = self.mlp(hidden_states)
        return layer_output, packed_meta_data


class TiteEncoder(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList(
            [TiteLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.pre_norm:
            if config.norm_type == "layer":
                self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            elif config.norm_type == "rms":
                self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            else:
                raise ValueError(f"Unknown norm type: {config.norm_type}")
        else:
            self.norm = torch.nn.Identity()

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...] | None]:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states, packed_meta_data = layer_module(hidden_states, packed_meta_data)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        return (hidden_states, tuple(all_hidden_states) if all_hidden_states is not None else None)
