import torch
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch.nn import Module
from transformers import BertConfig as OriginalBertConfig
from transformers import BertForMaskedLM, PreTrainedModel
from transformers.activations import ACT2FN

from .bert import BertConfig, BertModel
from .model import (
    TiteConfig,
    TiteEmbeddings,
    TiteIntermediate,
    TiteLayer,
    TiteOutput,
    TiteSelfAttention,
    TiteSelfOutput,
)


class MLMDecoder(Module):
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
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class HFMLMDecoder(MLMDecoder):

    def __init__(self, model_name_or_path: str):
        config = OriginalBertConfig.from_pretrained(model_name_or_path)
        super().__init__(config.vocab_size, config.hidden_size, config.hidden_act)
        model = BertForMaskedLM.from_pretrained(model_name_or_path)
        state_dict = {}
        for key, value in model.cls.state_dict().items():
            state_dict[key.replace("predictions.", "").replace("transform.", "")] = value
        self.load_state_dict(state_dict)


class MAESelfAttention(torch.nn.Module):

    def __init__(self, config: TiteConfig, layer_idx: int, enhanced: bool = True, mask_prob: float = 0.0):
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

        self.Wq = torch.nn.Linear(hidden_size, self.all_head_size)
        self.Wkv = torch.nn.Linear(hidden_size, 2 * self.all_head_size)

        self.dropout_prob = config.dropout_prob

        self.enhanced = enhanced
        self.mask_prob = mask_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        embx: torch.Tensor,
        expanded_embx: torch.Tensor | None,
        mlm_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape

        expanded_hidden_states = torch.cat([embx, hidden_states], dim=1)

        kv = self.Wkv(expanded_hidden_states)
        kv = rearrange(kv, "b s (t h d) -> t b h s d", t=2, h=self.num_attention_heads, d=self.attention_head_size)
        key, value = kv.unbind(dim=0)

        if self.enhanced:
            # use embx as query + positional embeddings
            # embx at position x can attend to random tokens in the sequence but not to the original token at position x
            assert expanded_embx is not None
            query = self.Wq(expanded_embx)
            query = rearrange(
                expanded_embx,
                "b s (h d) -> b h s d",
                h=self.num_attention_heads,
                d=self.attention_head_size,
            )
            decoding_mask = torch.rand((batch_size, seq_len, seq_len), device=hidden_states.device) >= self.mask_prob
            diag_mask = ~torch.eye(seq_len, device=hidden_states.device).bool()
            enhanced_decoding_mask = (
                attention_mask.logical_and(~mlm_mask)[:, None].logical_and(decoding_mask).logical_and(diag_mask)
            )
            enhanced_decoding_mask = torch.nn.functional.pad(enhanced_decoding_mask, (1, 0, 0, 0), value=True)
            attn_weight = repeat(
                torch.where(enhanced_decoding_mask, 0, -10000.0),
                "b s1 s2 -> b h s1 s2",
                h=self.num_attention_heads,
            )
        else:
            # aggressively masked input can attend to embx and has to reconstruct original tokens
            query = rearrange(
                self.Wq(hidden_states), "b s (h d) -> b h s d", h=self.num_attention_heads, d=self.attention_head_size
            )
            extended_attention_mask = torch.nn.functional.pad(attention_mask, (1, 0), value=True)
            attn_weight = repeat(
                torch.where(
                    einsum(
                        attention_mask,
                        extended_attention_mask,
                        "b s1, b s2 -> b s1 s2",
                    ),
                    0,
                    -10000.0,
                ),
                "b s1 s2-> b h s1 s2",
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

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, expanded_embx: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.enhanced:
            assert expanded_embx is not None
            hidden_states = hidden_states + expanded_embx
        else:
            hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MAEAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int, enhanced: bool = True, mask_prob: float = 0.0):
        super().__init__()
        self.self = MAESelfAttention(config, layer_idx, enhanced, mask_prob)
        self.output = MAESelfOutput(config, layer_idx, enhanced)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        embx: torch.Tensor,
        expanded_embx: torch.Tensor | None,
        mlm_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask, embx, expanded_embx, mlm_mask)
        attn_output = self.output(self_outputs, hidden_states, expanded_embx)
        return attn_output


class MAEDecoder(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.mlm_decoder = MLMDecoder(config.vocab_size, config.hidden_size[0], config.hidden_act)
        self.decoder = self.mlm_decoder.decoder

        self.post_init()

    def forward(self, embx: Tensor, input_ids: Tensor, attention_mask: Tensor, *args, **kwargs) -> Tensor:
        attention_mask = attention_mask.bool()
        hidden_states = self.embeddings(input_ids, attention_mask)
        num_sub_vectors = embx.shape[-1] // hidden_states.shape[-1]
        embx = embx.view(embx.shape[0], num_sub_vectors, hidden_states.shape[-1])
        hidden_states = torch.cat([embx, hidden_states], dim=1)
        attention_mask = torch.nn.functional.pad(attention_mask, (1, 0), value=True)
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = hidden_states[:, num_sub_vectors:]
        logits = self.mlm_decoder(hidden_states)
        return logits


class BlockPredictor(Module):

    def __init__(
        self,
        num_hidden_layers: int = 2,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu_pytorch_tanh",
        position_embeddings: bool = False,
    ) -> None:
        super().__init__()
        config = TiteConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=(hidden_size,) * num_hidden_layers,
            num_attention_heads=(num_attention_heads,) * num_hidden_layers,
            intermediate_size=(intermediate_size,) * num_hidden_layers,
            hidden_act=hidden_act,
            kernel_size=(None,) * num_hidden_layers,
            stride=(None,) * num_hidden_layers,
        )
        self.position_embeddings = None
        if position_embeddings:
            self.position_embeddings = torch.nn.Embedding(512, hidden_size)
        self.layers = torch.nn.ModuleList()
        for idx in range(num_hidden_layers):
            self.layers.append(TiteLayer(config, idx))

    def format_block_embeddings(self, hidden_states: Tensor, student_batch_idcs: tuple[int]) -> Tensor:
        batch_idcs = torch.tensor(student_batch_idcs, device=hidden_states.device)
        num_blocks = batch_idcs.bincount()
        assert hidden_states.shape[1] == 1
        pooled_hidden_states = hidden_states[:, 0]
        split_hidden_states = pooled_hidden_states.split(num_blocks.tolist())
        padded_hidden_states = torch.nn.utils.rnn.pad_sequence(split_hidden_states, batch_first=True)
        if self.position_embeddings is not None:
            padded_hidden_states = padded_hidden_states + self.position_embeddings(
                torch.arange(padded_hidden_states.shape[1], device=hidden_states.device)
            )
        return padded_hidden_states


class BlockEmbeddingPredictor(BlockPredictor):

    def __init__(
        self,
        num_hidden_layers: int = 2,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu_pytorch_tanh",
    ) -> None:
        super().__init__(
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            position_embeddings=True,
        )
        self.cls_token = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden_states: Tensor, student_batch_idcs: tuple[int], *args, **kwargs) -> Tensor:
        batch_idcs = torch.tensor(student_batch_idcs, device=hidden_states.device)
        num_blocks = batch_idcs.bincount()
        hidden_states = self.format_block_embeddings(hidden_states, student_batch_idcs)
        hidden_states = hidden_states.repeat_interleave(num_blocks, dim=0)
        attention_mask = torch.cat(
            [
                torch.nn.functional.pad(
                    ~torch.eye(num_b, device=hidden_states.device, dtype=bool),
                    (0, num_blocks.max() - num_b),
                    value=False,
                )
                for num_b in num_blocks
            ],
            dim=0,
        )
        hidden_states = torch.cat(
            [self.cls_token[None, None, :].repeat(hidden_states.shape[0], 1, 1), hidden_states], dim=1
        )
        attention_mask = torch.nn.functional.pad(attention_mask, (1, 0), value=True)
        for layer in self.layers:
            hidden_states, attention_mask = layer(hidden_states, attention_mask)
        pred_hidden_states = hidden_states[:, [0]]
        return pred_hidden_states


class BlockOrderPredictor(BlockPredictor):

    def __init__(
        self,
        num_hidden_layers: int = 2,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu_pytorch_tanh",
    ) -> None:
        super().__init__(
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            position_embeddings=False,
        )
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Tensor, student_batch_idcs: tuple[int], *args, **kwargs) -> Tensor:
        hidden_states = self.format_block_embeddings(hidden_states, student_batch_idcs)
        batch_idcs = torch.tensor(student_batch_idcs, device=hidden_states.device)
        num_blocks = batch_idcs.bincount()
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(bs, device=hidden_states.device, dtype=bool) for bs in num_blocks], batch_first=True
        )
        for layer in self.layers:
            hidden_states, attention_mask = layer(hidden_states, attention_mask)
        pred = self.linear(hidden_states).squeeze(-1)
        return pred


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        return input
