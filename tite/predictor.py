import torch
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch.nn import Module
from transformers import BertConfig, BertForMaskedLM, PreTrainedModel
from transformers.activations import ACT2FN

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
        config = BertConfig.from_pretrained(model_name_or_path)
        super().__init__(config.vocab_size, config.hidden_size, config.hidden_act)
        model = BertForMaskedLM.from_pretrained(model_name_or_path)
        state_dict = {}
        for key, value in model.cls.state_dict().items():
            state_dict[key.replace("predictions.", "").replace("transform.", "")] = value
        self.load_state_dict(state_dict)


class MAESelfAttention(TiteSelfAttention):

    def __init__(self, config: TiteConfig, layer_idx: int, enhanced: bool = True, mask_prob: float = 0.0):
        super().__init__(config, layer_idx)
        self.enhanced = enhanced
        self.mask_prob = mask_prob
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size[0])

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, embx: torch.Tensor, mlm_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape

        extended_hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 1, 0))
        extended_hidden_states[:, 0] = embx[:, 0]

        qkv = self.Wqkv(extended_hidden_states)
        qkv = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=self.num_attention_heads, d=self.attention_head_size)
        query, key, value = qkv.unbind(dim=0)

        if self.enhanced:
            # use embx as query + positional embeddings
            # embx at position x can attend to random tokens in the sequence but not to the original token at position x
            embx_query = repeat(embx, "b 1 d -> b s d", s=seq_len) + (
                self.position_embeddings(torch.arange(seq_len, device=hidden_states.device))
            )
            query = rearrange(
                embx_query,
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
            query = query[:, :, 1:]
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

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # NOTE do not do skip connection because enhanced decoding applies it's own per token masking
        # hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MAEAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int, enhanced: bool = True, mask_prob: float = 0.0):
        super().__init__()
        self.self = MAESelfAttention(config, layer_idx, enhanced, mask_prob)
        self.output = MAESelfOutput(config, layer_idx)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, embx: torch.Tensor, mlm_mask: torch.Tensor
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask, embx, mlm_mask)
        attn_output = self.output(self_outputs, hidden_states)
        return attn_output


class MAEDecoder(PreTrainedModel):
    def __init__(self, config: TiteConfig, enhanced: bool = True, mask_prob: float = 0.0):
        super().__init__(config)
        self.embeddings = TiteEmbeddings(config)
        self.attention = MAEAttention(config, 0, enhanced, mask_prob)
        self.intermediate = TiteIntermediate(config, 0)
        self.output = TiteOutput(config, 0)
        self.mlm_decoder = MLMDecoder(config.vocab_size, config.hidden_size[0], config.hidden_act)
        self.decoder = self.mlm_decoder.decoder
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def tie_decoder_weights(self, output_embeddings: Module):
        self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_weights(self):
        self.mlm_decoder.decoder.bias = self.bias
        assert self.embeddings.position_embeddings is not None
        self.attention.self.position_embeddings.weight = self.embeddings.position_embeddings.weight

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
        self, embx: Tensor, input_ids: Tensor, attention_mask: Tensor, mlm_mask: Tensor, *args, **kwargs
    ) -> Tensor:
        attention_mask = attention_mask.bool()
        hidden_states = self.embeddings(input_ids, attention_mask)
        attn_output = self.attention(hidden_states, attention_mask, embx, mlm_mask)
        intermediate_output = self.intermediate(attn_output)
        hidden_states = self.output(intermediate_output, attn_output)
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
