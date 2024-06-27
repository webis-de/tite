import warnings
from typing import List, Sequence, Tuple

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_utils import apply_chunking_to_forward


def unpad(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    mask = mask.expand(x.shape[:-1])
    seq_lengths = mask.sum(1).detach().cpu().tolist()
    x = x[mask]
    return x, seq_lengths


def re_pad(x: torch.Tensor, seq_lengths: Sequence[int]) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(x.split(seq_lengths), batch_first=True)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_output_shape(
    input_shape: int,
    kernel_size: Tuple[int | None, ...],
    stride: Tuple[int | None, ...],
) -> int:
    output_shape = input_shape
    for k, s in zip(kernel_size, stride):
        if k is None or s is None:
            continue
        output_shape = ceil_div((max(0, output_shape - k)), s) + 1
    return output_shape


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
            warnings.warn(
                "No pooling layers are used. The output shape will be the same as"
                " the input shape."
            )
        else:
            output_shape = compute_output_shape(
                max_position_embeddings, kernel_size, stride
            )
            if output_shape != 1:
                raise ValueError(
                    "Output shape with input of maximum sequence length is not 1. "
                    "Please adjust kernel_size and stride."
                )


class TiteModel(PreTrainedModel):

    def __init__(self, config: TiteConfig):
        super().__init__(config)

        self.embeddings = TiteEmbeddings(config)
        self.encoder = TiteEncoder(config)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool
            )
        attention_mask = attention_mask.bool()
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states


class TiteEmbeddings(torch.nn.Module):

    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_size = config.hidden_size[0]
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_size)
        self.position_embeddings = torch.nn.Embedding(
            config.max_position_embeddings, hidden_size
        )
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout_prob = config.dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        unpad_hidden_states, seq_lengths = unpad(hidden_states, attention_mask)
        query = self.transpose_for_scores(
            re_pad(self.query(unpad_hidden_states), seq_lengths)
        )
        key = self.transpose_for_scores(
            re_pad(self.key(unpad_hidden_states), seq_lengths)
        )
        value = self.transpose_for_scores(
            re_pad(self.value(unpad_hidden_states), seq_lengths)
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask.unsqueeze(1).unsqueeze(-1),
            self.dropout_prob if self.training else 0.0,
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

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask)
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

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output, seq_lengths = unpad(attn_output, attention_mask)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attn_output,
        )
        layer_output = re_pad(layer_output, seq_lengths)
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
            [
                TiteLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states, attention_mask = layer_module(hidden_states, attention_mask)
        return hidden_states
