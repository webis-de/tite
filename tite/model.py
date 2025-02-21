from dataclasses import dataclass
from typing import List, Literal, Tuple

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


class ComposedEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        small_embedding_dim: int,
        large_embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            large_embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )
        self.linear = None
        if small_embedding_dim != large_embedding_dim:
            self.linear = torch.nn.Linear(large_embedding_dim, small_embedding_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embeddings = torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        if self.linear is not None:
            embeddings = self.linear(embeddings)
        return embeddings


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
        hidden_act: str = "gelu_pytorch_tanh",
        positional_embedding_type: Literal["absolute", "rotary", "alibi"] = "rotary",
        upscale_hidden_sizes: bool = False,
        pooling_location: Literal["pre", "attention", "post"] = "attention",
        rotary_interleaved: bool = False,
        norm_location: Literal["pre", "post"] = "pre",
        norm_type: Literal["rms", "layer"] = "rms",
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        pooling_implementation: Literal["eager", "triton"] = "triton",
        rope_implementation: Literal["eager", "triton"] = "triton",
        compile: bool = True,
        **kwargs,
    ):
        if attn_implementation is None:
            attn_implementation = "flash_attention_2"
        super().__init__(pad_token_id=pad_token_id, attn_implementation=attn_implementation, **kwargs)
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
        self.rotary_interleaved = rotary_interleaved
        self.norm_location = norm_location
        self.norm_type = norm_type
        self.pooling_implementation = pooling_implementation
        self.rope_implementation = rope_implementation
        self.compile = compile

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

    # Flash Attention 2 support
    _supports_flash_attn_2 = True

    # SDPA support
    _supports_sdpa = True

    def __init__(self, config: TiteConfig):
        super().__init__(config)
        self.config: TiteConfig

        self.embeddings = TiteEmbeddings(config)
        if config.compile:
            self.embeddings.compile(dynamic=True)
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
        output_attentions: bool = False,
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
        packed_meta_data = PackedMetaData(
            seq_lens,
            cu_seq_lens,
            max_seq_len,
            (
                None
                if self.config._attn_implementation == "flash_attention_2"
                and self.config.pooling_implementation == "triton"
                else idcs
            ),
        )
        hidden_states, all_hidden_states, all_attentions = self.encoder(
            hidden_states, packed_meta_data, output_hidden_states, output_attentions
        )
        if hidden_states.shape[0] == seq_lens.shape[0]:
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
        missing_keys = set(model.state_dict().keys()) - set(new_state_dict.keys())
        unexpected_keys = set(new_state_dict.keys()) - set(model.state_dict().keys())
        missing_keys = missing_keys - {"encoder.norm.weight", "encoder.norm.bias"}
        unexpected_keys = unexpected_keys - {
            "encoder.layer.0.attention.norm.weight",
            "encoder.layer.0.attention.norm.bias",
        }
        assert not missing_keys, f"Missing keys: {missing_keys}"
        assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
        loaded_keys = list(new_state_dict.keys())
        model, *out = super()._load_pretrained_model(
            model, new_state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )
        return (model, *out)


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
        if config.upscale_hidden_sizes:
            from_hidden_size = to_hidden_size
        else:
            from_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        self.hidden_size_diff = to_hidden_size - from_hidden_size

        if config.norm_location == "pre" and layer_idx == 0:
            self.norm = torch.nn.Identity()
        else:
            if config.norm_type == "layer":
                self.norm = torch.nn.LayerNorm(
                    from_hidden_size if config.norm_location == "pre" else to_hidden_size, eps=config.layer_norm_eps
                )
            elif config.norm_type == "rms":
                self.norm = RMSNorm(
                    from_hidden_size if config.norm_location == "pre" else to_hidden_size, eps=config.layer_norm_eps
                )
            else:
                raise ValueError(f"Unknown norm type: {config.norm_type}")
        if config.compile:
            self.norm.compile(dynamic=True)

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
                self.attention_head_size, config.max_position_embeddings, interleaved=config.rotary_interleaved
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

        if self.pooling is not None and self.config.pooling_location == "pre":
            hidden_states, packed_meta_data = self.pooling(hidden_states, packed_meta_data)
        input_hidden_states = hidden_states

        if self.config.norm_location == "pre":
            hidden_states = self.norm(hidden_states)

        value = self.value(hidden_states)
        if packed_meta_data.max_seq_len == 1:
            hidden_states, attn_probs = value, None
        else:
            query = self.query(hidden_states)
            key = self.key(hidden_states)

            query_packed_meta_data = packed_meta_data
            if self.pooling is not None and self.config.pooling_location == "attention":
                query, query_packed_meta_data = self.pooling(query, packed_meta_data)

            query = rearrange(query, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
            key = rearrange(key, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)
            value = rearrange(value, "t (h d) -> t h d", h=self.num_attention_heads, d=self.attention_head_size)

            if self.rope is not None:
                query = self.rope(query, query_packed_meta_data)
                key = self.rope(key, packed_meta_data)

            hidden_states, attn_probs = self.attention_function(
                query, key, value, query_packed_meta_data, packed_meta_data, output_attention
            )

            hidden_states = rearrange(hidden_states, "t h d -> t (h d)")

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.pooling is not None and self.config.pooling_location == "attention":
            input_hidden_states, packed_meta_data = self.pooling(input_hidden_states, packed_meta_data)

        if self.hidden_size_diff > 0:
            input_hidden_states = torch.nn.functional.pad(input_hidden_states, (0, self.hidden_size_diff))
        elif self.hidden_size_diff < 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, -self.hidden_size_diff))
        hidden_states = hidden_states + input_hidden_states

        if self.pooling is not None and self.config.pooling_location == "post":
            hidden_states, packed_meta_data = self.pooling(hidden_states, packed_meta_data)

        if self.config.norm_location == "post":
            hidden_states = self.norm(hidden_states)
        return hidden_states, attn_probs, packed_meta_data


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


class TiteUpscale(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        old_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        self.upscale_layer = torch.nn.Linear(old_hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

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
        if config.compile:
            self.mlp.compile(dynamic=True)

        hidden_size = config.hidden_sizes[layer_idx]
        old_hidden_size = config.hidden_sizes[max(0, layer_idx - 1)]
        if config.upscale_hidden_sizes and old_hidden_size != hidden_size:
            self.upscale_layer = TiteUpscale(config, layer_idx)
        else:
            self.upscale_layer = torch.nn.Identity()
        if config.compile:
            self.upscale_layer.compile(dynamic=True)

    def forward(
        self, hidden_states: torch.Tensor, packed_meta_data: PackedMetaData, output_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None, PackedMetaData]:
        if self.upscale_layer is not None:
            hidden_states = self.upscale_layer(hidden_states)
        hidden_states, attention, packed_meta_data = self.attention(hidden_states, packed_meta_data, output_attention)
        layer_output = self.mlp(hidden_states)
        return layer_output, attention, packed_meta_data


class TiteEncoder(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList(
            [TiteLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm: torch.nn.Module
        if config.norm_location == "pre":
            if config.norm_type == "layer":
                self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            elif config.norm_type == "rms":
                self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            else:
                raise ValueError(f"Unknown norm type: {config.norm_type}")
        else:
            self.norm = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        packed_meta_data: PackedMetaData,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | None, List[torch.Tensor] | None]:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions: List[torch.Tensor] | None = [] if output_attentions else None
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states, attention, packed_meta_data = layer_module(
                hidden_states, packed_meta_data, output_attentions
            )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            if all_attentions is not None:
                all_attentions.append(attention)
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_hidden_states, all_attentions
