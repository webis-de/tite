import math
import warnings
from typing import List, Literal, Tuple

import torch
from einops import einsum, rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from .unpad import repad, unpad


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_output_shapes(
    input_shape: int, kernel_size: Tuple[int | None, ...], stride: Tuple[int | None, ...]
) -> List[int]:
    output_shapes = [input_shape]
    for k, s in zip(kernel_size, stride):
        if k is None or s is None:
            output_shapes.append(output_shapes[-1])
        else:
            output_shapes.append(ceil_div((max(0, output_shapes[-1] - k)), s) + 1)
    return output_shapes


class TiteConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 30522,
        num_hidden_layers: int = 12,
        hidden_size: Tuple[int, ...] = (768,) * 12,
        num_attention_heads: Tuple[int, ...] = (12,) * 12,
        intermediate_size: Tuple[int, ...] = (3072,) * 12,
        kernel_size: Tuple[int | None, ...] = (None,) * 12,
        stride: Tuple[int | None, ...] = (None,) * 12,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        hidden_act: str = "gelu_pytorch_tanh",
        positional_embedding_type: Literal["absolute", "ALiBi"] = "ALiBi",
        unpadding: bool = False,
        upscale_hidden_size: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.positional_embedding_type = positional_embedding_type
        self.unpadding = unpadding
        self.upscale_hidden_size = upscale_hidden_size

        iterator = zip(
            [
                "hidden_size",
                "num_attention_heads",
                "intermediate_size",
                "kernel_size",
                "stride",
            ],
            [hidden_size, num_attention_heads, intermediate_size, kernel_size, stride],
        )
        for name, setting in iterator:
            if len(setting) != num_hidden_layers:
                raise ValueError(
                    f"Length of {name} does not match num_hidden_layers. "
                    f"Expected {num_hidden_layers}, got {len(setting)}."
                )

        if all(k is None and s is None for k, s in zip(kernel_size, stride)):
            warnings.warn("No pooling layers are used. The output shape will be the same as the input shape.")
        else:
            if self.output_shapes[-1] != 1:
                raise ValueError(
                    "Output shape with input of maximum sequence length is not 1. "
                    "Please adjust kernel_size and stride."
                )

    @property
    def output_shapes(self) -> List[int]:
        return compute_output_shapes(self.max_position_embeddings, self.kernel_size, self.stride)

    @property
    def last_hidden_size(self) -> int:
        return self.hidden_size[-1]


class TiteModel(PreTrainedModel):
    config_class = TiteConfig

    def __init__(self, config: TiteConfig):
        super().__init__(config)

        self.embeddings = TiteEmbeddings(config)
        self.encoder = TiteEncoder(config)

        self.unpadding = config.unpadding

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        batch_size, seq_len = input_ids.shape
        if self.unpadding:
            indices = torch.nonzero(attention_mask.flatten()).flatten()
            input_ids = unpad(input_ids, indices)
        hidden_states = self.embeddings(input_ids, attention_mask)
        hidden_states = self.encoder(hidden_states, attention_mask)
        if self.unpadding:
            if hidden_states.shape[0] != batch_size:
                hidden_states = repad(hidden_states, indices, batch_size, seq_len)
            else:
                hidden_states = hidden_states.unsqueeze(1)
        return hidden_states


class TiteEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_size = config.hidden_size[0]
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        self.unpadding = config.unpadding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            if self.unpadding:
                embeddings = embeddings + self.position_embeddings(attention_mask.nonzero()[:, 1].flatten())
            else:
                embeddings = embeddings + self.position_embeddings(
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def sum_pool2d(
    input: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int], ceil_mode: bool = False
) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(
        input, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode, divisor_override=1
    )


class MaskedAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        if x.shape[-2] == 1:
            return x, mask
        x = torch.where(mask.unsqueeze(-1).expand((-1, -1, x.shape[-1])), x, 0)
        kernel_size = min(self.kernel_size, mask.shape[-1])
        normalization = sum_pool2d(
            mask.float().unsqueeze(-1), kernel_size=(kernel_size, 1), stride=(self.stride, 1), ceil_mode=True
        )
        y_mask = (normalization != 0).squeeze(-1)
        normalization[normalization == 0] = 1
        sums = sum_pool2d(x, kernel_size=(kernel_size, 1), stride=(self.stride, 1), ceil_mode=True)
        y = sums / normalization
        return y, y_mask


class TiteSelfAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        if config.hidden_size[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        to_hidden_size = config.hidden_size[layer_idx]
        if config.upscale_hidden_size:
            from_hidden_size = to_hidden_size
        else:
            from_hidden_size = config.hidden_size[max(0, layer_idx - 1)]
        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(to_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Wqkv = torch.nn.Linear(from_hidden_size, 3 * self.all_head_size)

        kernel_size = config.kernel_size[layer_idx]
        stride = config.stride[layer_idx]
        self.pooling = None
        if kernel_size is not None and stride is not None:
            self.pooling = MaskedAvgPool1d(kernel_size, stride)

        self.dropout_prob = config.dropout_prob
        if config.positional_embedding_type == "ALiBi":
            self.register_buffer(
                "alibi", self.get_alibi_embeddings(config.output_shapes[layer_idx], self.num_attention_heads)
            )
        else:
            self.alibi = None

        self.unpadding = config.unpadding

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

        if self.unpadding:
            indices = torch.nonzero(attention_mask.flatten()).flatten()
            qkv = repad(qkv, indices, batch_size, seq_len)
        qkv = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=self.num_attention_heads, d=self.attention_head_size)

        query, key, value = qkv.unbind(dim=0)

        new_attention_mask = attention_mask
        if self.pooling is not None:
            query = rearrange(query, "b h s d -> b s (h d)")
            query, new_attention_mask = self.pooling(query, attention_mask)
            query = rearrange(query, "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size)
        new_seq_len = new_attention_mask.shape[1]

        attn_weight = repeat(
            torch.where(einsum(new_attention_mask, attention_mask, "b s1, b s2 -> b s1 s2"), 0, -10000.0),
            "b s1 s2 -> b h s1 s2",
            h=self.num_attention_heads,
        )
        if self.alibi is not None:
            attn_weight = attn_weight + self.alibi[:, :, : attn_weight.shape[-2], : attn_weight.shape[-1]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_weight, self.dropout_prob if self.training else 0.0
        )
        attn_output = rearrange(
            attn_output,
            "b h s d -> b s (h d)",
            b=batch_size,
            h=self.num_attention_heads,
            s=new_seq_len,
            d=self.attention_head_size,
        )
        if self.unpadding:
            if attention_mask.shape != new_attention_mask.shape:
                indices = torch.nonzero(new_attention_mask.flatten()).flatten()
            attn_output = unpad(attn_output, indices)
        return attn_output, new_attention_mask


class TiteSelfOutput(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size[layer_idx]
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        kernel_size = config.kernel_size[layer_idx]
        stride = config.stride[layer_idx]
        self.pooling = None
        if kernel_size is not None and stride is not None:
            self.pooling = MaskedAvgPool1d(kernel_size, stride)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.pooling is not None:
            input_tensor, _ = self.pooling(input_tensor, attention_mask)
        hidden_states[..., : input_tensor.shape[-1]] = hidden_states[..., : input_tensor.shape[-1]] + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TiteAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.self = TiteSelfAttention(config, layer_idx)
        self.output = TiteSelfOutput(config, layer_idx)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs, new_attention_mask = self.self(hidden_states, attention_mask)
        attn_output = self.output(self_outputs, attention_mask, hidden_states)
        return attn_output, new_attention_mask


class TiteIntermediate(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size[layer_idx]
        intermediate_size = config.intermediate_size[layer_idx]
        self.dense = torch.nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TiteOutput(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size[layer_idx]
        intermediate_size = config.intermediate_size[layer_idx]
        self.dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TiteUpscale(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size[layer_idx]
        old_hidden_size = config.hidden_size[max(0, layer_idx - 1)]
        self.upscale_layer = torch.nn.Linear(old_hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

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

        hidden_size = config.hidden_size[layer_idx]
        old_hidden_size = config.hidden_size[max(0, layer_idx - 1)]
        self.upscale_layer = None
        if config.upscale_hidden_size and old_hidden_size != hidden_size:
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states, attention_mask = layer_module(hidden_states, attention_mask)
        return hidden_states
