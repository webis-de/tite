from typing import Literal

import torch
from einops import rearrange
from transformers import BertConfig as OriginalBertConfig
from transformers import BertForMaskedLM, PreTrainedModel

from ..model.bert import BertConfig, BertModel
from ..model.tite import ACT2FN, RMSNorm, TiteConfig, TiteMLP
from .decoder import Decoder


class MLMDecoder(Decoder):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        hidden_act: str = "gelu_pytorch_tanh",
        norm_type: Literal["layer", "rms"] = "layer",
    ) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        if norm_type == "layer":
            self.norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        elif norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=1e-12)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, encoder_output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        encoder_output = self.dense(encoder_output)
        encoder_output = self.transform_act_fn(encoder_output)
        encoder_output = self.norm(encoder_output)
        logits = self.decoder(encoder_output)
        return logits


class HFMLMDecoder(MLMDecoder):

    def __init__(self, model_name_or_path: str):
        config = OriginalBertConfig.from_pretrained(model_name_or_path)
        super().__init__(config.vocab_size, config.hidden_sizes, config.hidden_act)
        model = BertForMaskedLM.from_pretrained(model_name_or_path)
        state_dict = {}
        for key, value in model.cls.state_dict().items():
            state_dict[key.replace("predictions.", "").replace("transform.", "")] = value
        self.load_state_dict(state_dict)


class MAEEnhancedAttention(torch.nn.Module):
    def __init__(self, config: TiteConfig, layer_idx: int, mask_prob: float = 0.0):
        super().__init__()
        if config.positional_embedding_type != "absolute":
            raise ValueError(f"Unsupported positional embedding type: {config.positional_embedding_type}")
        self.config = config

        hidden_size = config.hidden_sizes[layer_idx]
        num_attention_heads = config.num_attention_heads[layer_idx]
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Wq = torch.nn.Linear(hidden_size, self.all_head_size)
        self.Wkv = torch.nn.Linear(hidden_size, 2 * self.all_head_size)

        self.dropout_prob = config.dropout_prob
        self.mask_prob = mask_prob

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        if config.norm_type == "layer":
            self.norm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        elif config.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {config.norm_type}")
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        if config.hidden_sizes[layer_idx] % config.num_attention_heads[layer_idx] != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_sizes}) is not a multiple of the "
                f"number of attention heads ({config.num_attention_heads})"
            )

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.norm_location == "pre":
            query_hidden_states = self.norm(query_hidden_states)
            key_value_hidden_states = self.norm(key_value_hidden_states)

        batch_size, seq_len = attention_mask.shape

        expanded_key_value_hidden_states = torch.cat([encoder_output, key_value_hidden_states], dim=1)

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
        decoding_mask = torch.rand((batch_size, seq_len, seq_len), device=encoder_output.device) >= self.mask_prob
        diag_mask = ~torch.eye(seq_len, device=encoder_output.device, dtype=torch.bool)
        enhanced_decoding_mask = attention_mask[:, None].logical_and(decoding_mask).logical_and(diag_mask)
        enhanced_decoding_mask = torch.nn.functional.pad(
            enhanced_decoding_mask, (encoder_output.shape[1], 0, 0, 0), value=True
        )

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, enhanced_decoding_mask[:, None]
        )
        hidden_states = rearrange(
            hidden_states,
            "b h s d -> b s (h d)",
            b=batch_size,
            h=self.num_attention_heads,
            s=seq_len,
            d=self.attention_head_size,
        )

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + query_hidden_states
        if self.config.norm_location == "post":
            hidden_states = self.norm(hidden_states)
        return hidden_states


class MAEEnhancedEmbeddings(torch.nn.Module):
    def __init__(self, config: TiteConfig):
        super().__init__()
        hidden_size = config.hidden_sizes[0]
        self.num_attention_heads = config.num_attention_heads
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = None
        if config.positional_embedding_type == "absolute":
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(
                torch.arange(input_ids.shape[1], device=input_ids.device)
            )
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MAEEnhancedDecoder(PreTrainedModel, Decoder):

    _supports_sdpa = True

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        mask_prob: float,
        norm_location: Literal["pre", "post"] = "post",
        norm_type: Literal["rms", "layer"] = "layer",
        subvectors: bool = False,
    ):
        config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            positional_embedding_type="absolute",
            attn_implementation="sdpa",
            norm_location=norm_location,
            norm_type=norm_type,
        )
        super().__init__(config)
        self.subvectors = subvectors

        self.embeddings = MAEEnhancedEmbeddings(config)
        self.attention = MAEEnhancedAttention(config, 0, mask_prob)
        self.mlp = TiteMLP(config, 0)

        if config.norm_location == "pre":
            if config.norm_type == "layer":
                self.norm = torch.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            elif config.norm_type == "rms":
                self.norm = RMSNorm(hidden_size, eps=config.layer_norm_eps)
            else:
                raise ValueError(f"Unknown norm type: {config.norm_type}")
        else:
            self.norm = torch.nn.Identity()

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
        elif isinstance(module, (torch.nn.LayerNorm, RMSNorm)):
            if isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        encoder_output: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_output.shape[1] > 1:
            encoder_output = encoder_output[:, [0]]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask = attention_mask.bool() & ~special_tokens_mask.bool()

        key_value_hidden_states = self.embeddings(input_ids)
        query_hidden_states = encoder_output.expand_as(key_value_hidden_states)
        if self.embeddings.position_embeddings is not None:
            position_embeddings = self.embeddings.position_embeddings(
                torch.arange(input_ids.shape[1], device=encoder_output.device)
            )
            query_hidden_states = query_hidden_states + position_embeddings

        hidden_states = self.attention(query_hidden_states, key_value_hidden_states, attention_mask, encoder_output)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.mlm_decoder(hidden_states)
        return logits


class MAEDecoder(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.mlm_decoder = MLMDecoder(config.vocab_size, config.hidden_sizes[0], config.hidden_act)
        self.decoder = self.mlm_decoder.decoder

        self.post_init()

    def forward(
        self, encoder_output: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        attention_mask = attention_mask.bool()
        hidden_states = self.embeddings(input_ids, attention_mask)
        num_sub_vectors = encoder_output.shape[-1] // hidden_states.shape[-1]
        encoder_output = encoder_output.view(encoder_output.shape[0], num_sub_vectors, hidden_states.shape[-1])
        hidden_states = torch.cat([encoder_output, hidden_states], dim=1)
        attention_mask = torch.nn.functional.pad(attention_mask, (num_sub_vectors, 0), value=True)
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = hidden_states[:, num_sub_vectors:]
        logits = self.mlm_decoder(hidden_states)
        return logits


class BOWDecoder(MLMDecoder):

    def forward(self, encoder_output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        encoder_output = encoder_output[:, [0]]
        return super().forward(encoder_output, *args, **kwargs)
