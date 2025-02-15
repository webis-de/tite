import math
from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
from einops import einsum, rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput


class RotaryPositionalEmbeddings(torch.nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init(dtype)

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self, dtype: torch.dtype | None = None) -> None:
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len, dtype)

    def build_rope_cache(self, max_seq_len: int = 4096, dtype: torch.dtype | None = None) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).to(dtype)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input tensor has shape [b, n_h, s, h_d]
        seq_len = x.shape[2]

        # extract the values based on whether input_pos is set or not
        rope_cache = self.cache[:seq_len]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        xshaped = x.view(*x.shape[:-1], self.dim // 2, 2)

        # reshape the cache for broadcasting
        rope_cache = rope_cache.view(seq_len, self.dim // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_output_shapes(
    input_shape: int, kernel_sizes: Tuple[int | None, ...], strides: Tuple[int | None, ...]
) -> List[int]:
    output_shapes = [input_shape]
    for k, s in zip(kernel_sizes, strides):
        if k is None or s is None:
            output_shapes.append(output_shapes[-1])
        else:
            output_shapes.append(ceil_div((max(0, output_shapes[-1] - k)), s) + 1)
    return output_shapes


@dataclass
class TiteModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...] | None = None


class TiteConfig(PretrainedConfig):

    model_type = "tite-legacy"

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
        positional_embedding_type: Literal["absolute", "ALiBi", "rotary"] = "ALiBi",
        upscale_hidden_sizes: bool = False,
        pooling_location: Literal["pre", "attention", "post"] = "attention",
        attention_based_pooling: bool = True,
        pooling_strategy: Literal["mean_conv", "select"] = "mean_conv",
        pooling_implementation: Literal["unfold", "sum_pool2d"] = "unfold",
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
        if not attention_based_pooling:
            self.pooling_location = "post"
        self.pooling_strategy = pooling_strategy
        self.pooling_implementation = pooling_implementation

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
                "Kernel_sizes and strides are set, but do not reduce the maximum sequence length to 1. "
                "Please adjust kernel_sizes and strides."
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
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
    ):
        if "embeddings.word_embeddings.linear.weight" in state_dict:
            linear = state_dict["embeddings.word_embeddings.linear.weight"]
            weight = state_dict["embeddings.word_embeddings.weight"]
            state_dict["embeddings.word_embeddings.weight"] = torch.matmul(weight, linear.transpose(0, 1))
        model, *out = super()._load_pretrained_model(
            model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, *args, **kwargs
        )
        return model, *out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> TiteModelOutput:
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embeddings(input_ids, attention_mask)
        hidden_states, all_hidden_states = self.encoder(hidden_states, attention_mask, output_hidden_states)
        return TiteModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class TiteEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_sizes = config.hidden_sizes[0]
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_sizes, padding_idx=config.pad_token_id)
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, hidden_sizes)
        self.LayerNorm = torch.nn.LayerNorm(hidden_sizes, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(
                torch.arange(input_ids.shape[1], device=input_ids.device)
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def sum_pool2d(
    input: torch.Tensor, kernel_sizes: tuple[int, int], strides: tuple[int, int], ceil_mode: bool = False
) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(
        input, kernel_sizes=kernel_sizes, strides=strides, ceil_mode=ceil_mode, divisor_override=1
    )


class MaskedAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_sizes: int, strides: int, implementation: Literal["unfold", "sum_pool2d"] = "unfold"):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.implementation = implementation

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.implementation == "unfold":
            return self.unfold_forward(x, mask)
        if self.implementation == "sum_pool2d":
            return self.sum_pool2d_forward(x, mask)
        raise ValueError(f"Unknown implementation {self.implementation}")

    def sum_pool2d_forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] == 1:
            return x, mask
        x = torch.where(mask.unsqueeze(-1).expand((-1, -1, x.shape[-1])), x, 0)
        kernel_sizes = min(self.kernel_sizes, mask.shape[-1])
        normalization = sum_pool2d(
            mask.float().unsqueeze(-1), kernel_sizes=(kernel_sizes, 1), strides=(self.strides, 1), ceil_mode=True
        )
        y_mask = (normalization != 0).squeeze(-1)
        normalization[normalization == 0] = 1
        sums = sum_pool2d(x, kernel_sizes=(kernel_sizes, 1), strides=(self.strides, 1), ceil_mode=True)
        y = sums / normalization
        return y, y_mask

    def unfold_forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE this implementation only works for kernel size == strides, other configurations are broken!
        if x.shape[-2] == 1:
            return x, mask
        padding = (x.shape[-2] - self.kernel_sizes) % self.strides
        if self.kernel_sizes > x.shape[-2]:
            padding = self.kernel_sizes - x.shape[-2]
        if padding != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding))
        x_blocks = x.unfold(-2, self.kernel_sizes, self.strides)
        mask_blocks = mask.unfold(-1, self.kernel_sizes, self.strides).unsqueeze(-2)
        x_masked = x_blocks * mask_blocks
        normalization = mask_blocks.sum(-1)
        normalization[normalization == 0] = 1
        y = x_masked.sum(-1) / normalization
        y_mask = mask_blocks.amax(-1).squeeze(-1)
        return y, y_mask


class MaskedSelect(torch.nn.Module):
    def __init__(self, kernel_sizes: int, strides: int):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        if kernel_sizes != strides:
            raise ValueError("Kernel size and strides must be equal for MaskedSelect")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] == 1:
            return x, mask
        x = x[..., :: self.strides, :]
        mask = mask[..., :: self.strides]
        return x, mask


class TiteSelfAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        if config.hidden_sizes[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_sizes}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        to_hidden_sizes = config.hidden_sizes[layer_idx]
        if config.upscale_hidden_sizes:
            from_hidden_sizes = to_hidden_sizes
        else:
            from_hidden_sizes = config.hidden_sizes[max(0, layer_idx - 1)]
        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(to_hidden_sizes / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Wqkv = torch.nn.Linear(from_hidden_sizes, 3 * self.all_head_size)

        kernel_sizes = config.kernel_sizes[layer_idx]
        strides = config.strides[layer_idx]
        self.pooling = None
        if kernel_sizes is not None and strides is not None and config.pooling_location == "attention":
            if config.pooling_strategy == "mean_conv":
                self.pooling = MaskedAvgPool1d(kernel_sizes, strides)
            elif config.pooling_strategy == "select":
                self.pooling = MaskedSelect(kernel_sizes, strides)
            else:
                raise ValueError(f"Unknown pooling strategy {config.pooling_strategy}")

        self.dropout_prob = config.dropout_prob
        if config.positional_embedding_type == "ALiBi":
            self.register_buffer(
                "alibi", self.get_alibi_embeddings(config.output_shapes[layer_idx], self.num_attention_heads)
            )
            self.rope = None
        elif config.positional_embedding_type == "rotary":
            self.alibi = None
            self.rope = RotaryPositionalEmbeddings(self.attention_head_size, config.max_position_embeddings)
        else:
            self.alibi = None
            self.rope = None

    def get_alibi_embeddings(self, max_position_embeddings: int, num_attention_heads: int) -> torch.Tensor:
        # https://github.com/mosaicml/examples/blob/daddaeff7a535273ff984b0134da4839c70a45b3/examples/benchmarks/bert/src/bert_layers.py#L432
        # ALiBi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        def _get_alibi_head_slopes(num_attention_heads: int) -> List[float]:

            def get_slopes_power_of_2(num_attention_heads: int) -> List[float]:
                start = 2 ** (-(2 ** -(math.log2(num_attention_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(num_attention_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(num_attention_heads).is_integer():
                return get_slopes_power_of_2(num_attention_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(num_attention_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: num_attention_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(max_position_embeddings)[:, None]
        memory_position = torch.arange(max_position_embeddings)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [num_attention_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(num_attention_heads, -1, -1)
        slopes = torch.tensor(_get_alibi_head_slopes(num_attention_heads), device=relative_position.device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, num_attention_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, num_attention_heads, max_position_embeddings, max_position_embeddings])

        return alibi

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = attention_mask.shape
        qkv = self.Wqkv(hidden_states)

        qkv = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=self.num_attention_heads, d=self.attention_head_size)

        query, key, value = qkv.unbind(dim=0)

        new_attention_mask = attention_mask
        if self.pooling is not None:
            query = rearrange(query, "b h s d -> b s (h d)")
            query, new_attention_mask = self.pooling(query, attention_mask)
            query = rearrange(query, "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size)
        new_seq_len = new_attention_mask.shape[1]

        if value.shape[-2] > 1:
            if self.rope is not None:
                query = self.rope(query)
                key = self.rope(key)
            if attention_mask.shape != new_attention_mask.shape:
                attn_weight = einsum(new_attention_mask, attention_mask, "b s1, b s2 -> b s1 s2")
            else:
                attn_weight = attention_mask[:, None].expand(batch_size, seq_len, seq_len)
            attn_weight = repeat(
                torch.where(attn_weight, 0, -10000.0), "b s1 s2 -> b h s1 s2", h=self.num_attention_heads
            )
            if self.alibi is not None:
                attn_weight = attn_weight + self.alibi[:, :, : attn_weight.shape[-2], : attn_weight.shape[-1]]

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query.to(value), key.to(value), value, attn_weight.to(value)
            )
        else:
            attn_output = value
        attn_output = rearrange(
            attn_output,
            "b h s d -> b s (h d)",
            b=batch_size,
            h=self.num_attention_heads,
            s=new_seq_len,
            d=self.attention_head_size,
        )
        return attn_output, new_attention_mask


class TiteSelfOutput(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_sizes = config.hidden_sizes[layer_idx]
        self.config = config
        self.dense = torch.nn.Linear(hidden_sizes, hidden_sizes)
        self.LayerNorm = torch.nn.LayerNorm(hidden_sizes, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        kernel_sizes = config.kernel_sizes[layer_idx]
        strides = config.strides[layer_idx]
        self.pooling = None
        if config.pooling_location in ("post", "attention") and kernel_sizes is not None and strides is not None:
            if config.pooling_strategy == "mean_conv":
                self.pooling = MaskedAvgPool1d(kernel_sizes, strides)
            elif config.pooling_strategy == "select":
                self.pooling = MaskedSelect(kernel_sizes, strides)
            else:
                raise ValueError(f"Unknown pooling strategy {config.pooling_strategy}")

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.pooling is not None and self.config.pooling_location == "attention":
            input_tensor, attention_mask = self.pooling(input_tensor, attention_mask)
        if hidden_states.shape[-1] == input_tensor.shape[-1]:
            hidden_states = hidden_states + input_tensor
        else:
            hidden_states[..., : input_tensor.shape[-1]] = hidden_states[..., : input_tensor.shape[-1]] + input_tensor
        if self.pooling is not None and self.config.pooling_location == "post":
            hidden_states, attention_mask = self.pooling(hidden_states, attention_mask)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states, attention_mask


class TiteAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.self = TiteSelfAttention(config, layer_idx)
        self.output = TiteSelfOutput(config, layer_idx)

        kernel_sizes = config.kernel_sizes[layer_idx]
        strides = config.strides[layer_idx]
        self.pooling = None
        if config.pooling_location == "pre" and kernel_sizes is not None and strides is not None:
            if config.pooling_strategy == "mean_conv":
                self.pooling = MaskedAvgPool1d(kernel_sizes, strides)
            elif config.pooling_strategy == "select":
                self.pooling = MaskedSelect(kernel_sizes, strides)
            else:
                raise ValueError(f"Unknown pooling strategy {config.pooling_strategy}")

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pooling is not None:
            hidden_states, attention_mask = self.pooling(hidden_states, attention_mask)
        self_outputs, new_attention_mask = self.self(hidden_states, attention_mask)
        attn_output, new_attention_mask = self.output(self_outputs, attention_mask, hidden_states)
        return attn_output, new_attention_mask


class TiteIntermediate(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_sizes = config.hidden_sizes[layer_idx]
        intermediate_sizes = config.intermediate_sizes[layer_idx]
        self.dense = torch.nn.Linear(hidden_sizes, intermediate_sizes)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TiteOutput(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_sizes = config.hidden_sizes[layer_idx]
        intermediate_sizes = config.intermediate_sizes[layer_idx]
        self.dense = torch.nn.Linear(intermediate_sizes, hidden_sizes)
        self.LayerNorm = torch.nn.LayerNorm(hidden_sizes, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TiteUpscale(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_sizes = config.hidden_sizes[layer_idx]
        old_hidden_sizes = config.hidden_sizes[max(0, layer_idx - 1)]
        self.upscale_layer = torch.nn.Linear(old_hidden_sizes, hidden_sizes)
        self.dropout = torch.nn.Dropout(config.dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(hidden_sizes, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upscale_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TiteLayer(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TiteAttention(config, layer_idx)
        self.intermediate = TiteIntermediate(config, layer_idx)
        self.output = TiteOutput(config, layer_idx)

        hidden_sizes = config.hidden_sizes[layer_idx]
        old_hidden_sizes = config.hidden_sizes[max(0, layer_idx - 1)]
        self.upscale_layer = None
        if config.upscale_hidden_sizes and old_hidden_sizes != hidden_sizes:
            self.upscale_layer = TiteUpscale(config, layer_idx)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.upscale_layer is not None:
            hidden_states = self.upscale_layer(hidden_states)
        attn_output, attention_mask = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output, attention_mask


class TiteEncoder(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList(
            [TiteLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.BoolTensor, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...] | None]:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer_module in self.layer:
            hidden_states, attention_mask = layer_module(hidden_states, attention_mask)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        return (hidden_states, tuple(all_hidden_states) if all_hidden_states is not None else None)


class MLMDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, hidden_act: str = "gelu_pytorch_tanh") -> None:
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class MAESelfAttention(torch.nn.Module):

    def __init__(self, config: TiteConfig, layer_idx: int, mask_prob: float = 0.0):
        super().__init__()

        if config.hidden_sizes[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_sizes}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        hidden_size = config.hidden_sizes[layer_idx]
        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Wq = torch.nn.Linear(hidden_size, self.all_head_size)
        self.Wkv = torch.nn.Linear(hidden_size, 2 * self.all_head_size)

        self.dropout_prob = config.dropout_prob
        self.mask_prob = mask_prob

        if config.positional_embedding_type != "absolute":
            raise ValueError(f"Unsupported positional embedding type: {config.positional_embedding_type}")

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        embx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape

        expanded_key_value_hidden_states = torch.cat([embx, key_value_hidden_states], dim=1)

        kv = self.Wkv(expanded_key_value_hidden_states)
        kv = rearrange(kv, "b s (t h d) -> t b h s d", t=2, h=self.num_attention_heads, d=self.attention_head_size)
        key, value = kv.unbind(dim=0)

        query = self.Wq(query_hidden_states)
        query = rearrange(
            query_hidden_states,
            "b s (h d) -> b h s d",
            h=self.num_attention_heads,
            d=self.attention_head_size,
        )
        decoding_mask = torch.rand((batch_size, seq_len, seq_len), device=embx.device) >= self.mask_prob
        diag_mask = ~torch.eye(seq_len, device=embx.device).bool()
        enhanced_decoding_mask = attention_mask[:, None].logical_and(decoding_mask).logical_and(diag_mask)
        enhanced_decoding_mask = torch.nn.functional.pad(enhanced_decoding_mask, (embx.shape[1], 0, 0, 0), value=True)

        attn_weight = repeat(
            torch.where(enhanced_decoding_mask, 0, -10000.0),
            "b s1 s2 -> b h s1 s2",
            h=self.num_attention_heads,
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_weight, self.dropout_prob if self.training else 0.0
        )
        attn_output = rearrange(
            attn_output,
            "b h s d -> b s (h d)",
            b=batch_size,
            h=self.num_attention_heads,
            s=seq_len,
            d=self.attention_head_size,
        )
        return attn_output


class MAESelfOutput(TiteSelfOutput):

    def __init__(self, config: TiteConfig, layer_idx: int, enhanced: bool = True):
        super().__init__(config, layer_idx)
        self.enhanced = enhanced

    def forward(self, hidden_states: torch.Tensor, query_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + query_hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MAEAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int, mask_prob: float = 0.0):
        super().__init__()
        self.self = MAESelfAttention(config, layer_idx, mask_prob)
        self.output = MAESelfOutput(config, layer_idx)

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        embx: torch.Tensor,
    ) -> torch.Tensor:
        self_outputs = self.self(query_hidden_states, key_value_hidden_states, attention_mask, embx)
        attn_output = self.output(self_outputs, query_hidden_states)
        return attn_output


class MAEEnhancedDecoder(PreTrainedModel):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        mask_id: int,
        mask_prob: float,
        query_strategy: Literal["embx", "mask"] = "embx",
        subvectors: bool = False,
    ):
        config = TiteConfig(
            num_hidden_layers=1,
            hidden_sizes=(hidden_size,),
            num_attention_heads=(num_attention_heads,),
            intermediate_sizes=(intermediate_size,),
            positional_embedding_type="absolute",
            strides=(None,),
            kernel_sizes=(None,),
        )
        super().__init__(config)
        self.mask_id = mask_id
        self.query_strategy = query_strategy
        self.subvectors = subvectors

        self.embeddings = TiteEmbeddings(config)
        self.attention = MAEAttention(config, 0, mask_prob)
        self.intermediate = TiteIntermediate(config, 0)
        self.output = TiteOutput(config, 0)

        self.mlm_decoder = MLMDecoder(config.vocab_size, hidden_size, config.hidden_act)
        self.decoder = self.mlm_decoder.decoder

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
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        embx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embx.shape[1] > 1:
            embx = embx[:, [0]]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask = attention_mask.bool() & ~special_tokens_mask.bool()

        key_value_hidden_states = self.embeddings(input_ids, attention_mask)
        if self.query_strategy == "embx":
            query_hidden_states = embx.expand_as(key_value_hidden_states)
            if self.embeddings.position_embeddings is not None:
                position_embeddings = self.embeddings.position_embeddings(
                    torch.arange(input_ids.shape[1], device=embx.device)
                )
                query_hidden_states = query_hidden_states + position_embeddings
        elif self.query_strategy == "mask":
            query_hidden_states = self.embeddings(torch.full_like(input_ids, self.mask_id), attention_mask)
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")

        attention_output = self.attention(query_hidden_states, key_value_hidden_states, attention_mask, embx)
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)
        logits = self.mlm_decoder(hidden_states)
        return logits
