import math
import warnings
from typing import List, Literal, Tuple

import torch
from einops import rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_utils import apply_chunking_to_forward

# unpad from
# https://github.com/mosaicml/examples/blob/daddaeff7a535273ff984b0134da4839c70a45b3/examples/benchmarks/bert/src/bert_padding.py


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # type: ignore
        second_dim = other_shape.numel()  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, "z -> z d", d=second_dim),  # (indices,) -> (indices, second_dim)
        ).reshape(
            -1, *other_shape
        )  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Like unpad_input, but only return the unpadded first tensor.

    Save a small amount of overhead.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
    """
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, indices)  # type: ignore


def re_pad(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Add padding to sequences.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        batch: int batch_size
        seqlen: int max sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)  # type: ignore


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
        hidden_act: str = "gelu_new",
        positional_embedding_type: Literal["absolute", "ALiBi"] = "ALiBi",
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


class TiteModel(PreTrainedModel):
    def __init__(self, config: TiteConfig):
        super().__init__(config)

        self.embeddings = TiteEmbeddings(config)
        self.encoder = TiteEncoder(config)

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
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
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states


class TiteEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_size = config.hidden_size[0]
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_size)
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(
                torch.arange(input_ids.shape[1], device=input_ids.device)
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MaskedAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] == 1:
            return x, mask
        padding = 0
        if (x.shape[-2] - self.kernel_size) % self.stride != 0 or self.kernel_size:
            padding = (x.shape[-2] - self.kernel_size) % self.stride
        if self.kernel_size > x.shape[-2]:
            padding = self.kernel_size - x.shape[-2]
        if padding:
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding))
        x_blocks = x.unfold(-2, self.kernel_size, self.stride)
        mask_blocks = mask.unfold(1, self.kernel_size, self.stride).unsqueeze(-2)
        x_masked = x_blocks * mask_blocks
        normalization = mask_blocks.sum(-1)
        normalization[normalization == 0] = 1
        y = x_masked.sum(-1) / normalization
        y_mask = mask_blocks.amax(-1).squeeze(-1)
        return y, y_mask


class TiteSelfAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        if config.hidden_size[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

        hidden_size = config.hidden_size[layer_idx]
        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Wqkv = torch.nn.Linear(hidden_size, 3 * self.all_head_size)

        self.dropout_prob = config.dropout_prob
        if config.positional_embedding_type == "ALiBi":
            self.register_buffer(
                "alibi", self.get_alibi_embeddings(config.output_shapes[layer_idx], self.num_attention_heads)
            )
        else:
            self.alibi = None

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
        slopes = torch.Tensor(_get_alibi_head_slopes(num_attention_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, num_attention_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, num_attention_heads, max_position_embeddings, max_position_embeddings])

        return alibi

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        unpad_hidden_states = unpad(hidden_states, indices)
        batch_size, seq_len = hidden_states.shape[:2]
        qkv = re_pad(self.Wqkv(unpad_hidden_states), indices, batch_size, seq_len)
        qkv = rearrange(qkv, "b s (t h d) -> b s t h d", t=3, h=self.num_attention_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query = qkv[0]
        key = qkv[1]
        value = qkv[2]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.to(hidden_states)) * -10000.0
        if self.alibi is not None:
            attention_mask = attention_mask + self.alibi[:, :, :seq_len, :seq_len]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attention_mask, self.dropout_prob if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(hidden_states.shape)
        return attn_output


class TiteSelfOutput(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size[layer_idx]
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TiteAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.self = TiteSelfAttention(config, layer_idx)
        self.output = TiteSelfOutput(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask, indices)
        attn_output = self.output(self_outputs, hidden_states)
        return attn_output


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


class TiteLayer(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TiteAttention(config, layer_idx)
        self.intermediate = TiteIntermediate(config, layer_idx)
        self.output = TiteOutput(config, layer_idx)

        kernel_size = config.kernel_size[layer_idx]
        stride = config.stride[layer_idx]
        self.pooling = None
        if kernel_size is not None and stride is not None:
            self.pooling = MaskedAvgPool1d(kernel_size, stride)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        batch_size, seq_len = hidden_states.shape[:2]
        attn_output = self.attention(hidden_states, attention_mask, indices)
        attn_output = unpad(attn_output, indices)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attn_output,
        )
        layer_output = re_pad(layer_output, indices, batch_size, seq_len)
        if self.pooling is not None:
            layer_output, attention_mask = self.pooling(layer_output, attention_mask)
        return layer_output, attention_mask

    def feed_forward_chunk(self, attn_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


class TiteEncoder(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList(
            [TiteLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states, attention_mask = layer_module(hidden_states, attention_mask)
        return hidden_states
