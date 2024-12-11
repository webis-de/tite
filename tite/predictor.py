from typing import Literal

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import Module
from transformers import BertConfig as OriginalBertConfig
from transformers import BertForMaskedLM, PreTrainedModel
from transformers.activations import ACT2FN

from .bert import BertConfig, BertModel
from .model import TiteConfig, TiteEmbeddings, TiteIntermediate, TiteLayer, TiteOutput, TiteSelfOutput


class MLMDecoder(Module):
    def __init__(
        self, vocab_size: int, orig_hidden_size: int, hidden_size: int, hidden_act: str = "gelu_pytorch_tanh"
    ) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, orig_hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.LayerNorm = torch.nn.LayerNorm(orig_hidden_size, eps=1e-12)
        self.decoder = torch.nn.Linear(orig_hidden_size, vocab_size)

        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


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

    def __init__(self, config: TiteConfig, layer_idx: int, mask_prob: float = 0.0):
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
        orig_hidden_size: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        mask_id: int,
        mask_prob: float,
        query_strategy: Literal["embx", "mask"] = "embx",
    ):
        embeddings_config = BertConfig(
            num_hidden_layers=1, hidden_size=orig_hidden_size, positional_embedding_type="absolute"
        )
        attention_config = BertConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            positional_embedding_type="absolute",
        )
        super().__init__(attention_config)
        self.mask_id = mask_id
        self.query_strategy = query_strategy

        self.embeddings = TiteEmbeddings(embeddings_config)
        self.attention = MAEAttention(attention_config, 0, mask_prob)
        self.intermediate = TiteIntermediate(attention_config, 0)
        self.output = TiteOutput(attention_config, 0)

        self.upscale = None
        self.upscale_position_embeddings = None
        if orig_hidden_size != hidden_size:
            self.upscale = torch.nn.Sequential(
                torch.nn.Linear(orig_hidden_size, hidden_size), torch.nn.LayerNorm(hidden_size)
            )
            if self.embeddings.position_embeddings is not None:
                self.upscale_position_embeddings = torch.nn.Sequential(
                    torch.nn.Linear(orig_hidden_size, hidden_size), torch.nn.LayerNorm(hidden_size)
                )

        self.mlm_decoder = MLMDecoder(
            attention_config.vocab_size, orig_hidden_size, attention_config.hidden_size[0], attention_config.hidden_act
        )
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
        self, embx: Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if embx.shape[1] > 1:
            embx = embx[:, [0]]
        if attention_mask is None:
            attention_mask = torch.ones(1, input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        key_value_hidden_states = self.embeddings(input_ids, attention_mask)
        if self.upscale is not None:
            key_value_hidden_states = self.upscale(key_value_hidden_states)
        if self.query_strategy == "embx":
            query_hidden_states = embx.expand_as(key_value_hidden_states)
            if self.embeddings.position_embeddings is not None:
                position_embeddings = self.embeddings.position_embeddings(
                    torch.arange(input_ids.shape[1], device=embx.device)
                )
                if self.upscale_position_embeddings is not None:
                    position_embeddings = self.upscale_position_embeddings(position_embeddings)
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
        attention_mask = torch.nn.functional.pad(attention_mask, (num_sub_vectors, 0), value=True)
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = hidden_states[:, num_sub_vectors:]
        logits = self.mlm_decoder(hidden_states)
        return logits


class BOWDecoder(MLMDecoder):

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states[:, [0]]
        return super().forward(hidden_states, *args, **kwargs)


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
