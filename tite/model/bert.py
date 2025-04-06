from typing import Dict, Literal

import torch

from .tite import (
    LMPredictionHead,
    PreTrainingHead,
    TiteConfig,
    TiteModel,
    TiteModelOutput,
    TitePreTrainedModel,
    TitePreTrainingOutput,
)


class BertConfig(TiteConfig):
    def __init__(
        self,
        vocab_size: int = 30522,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        hidden_act: str = "gelu_pytorch_tanh",
        positional_embedding_type: Literal["absolute", "rotary", "alibi"] = "rotary",
        rotary_interleaved: bool = False,
        norm_location: Literal["pre", "post"] = "pre",
        norm_type: Literal["rms", "layer"] = "rms",
        pooling: Literal["mean", "first"] | None = None,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        rope_implementation: Literal["eager", "triton"] = "triton",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_sizes=(hidden_size,) * num_hidden_layers,
            num_attention_heads=(num_attention_heads,) * num_hidden_layers,
            intermediate_sizes=(intermediate_size,) * num_hidden_layers,
            kernel_sizes=(None,) * num_hidden_layers,
            strides=(None,) * num_hidden_layers,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            hidden_act=hidden_act,
            positional_embedding_type=positional_embedding_type,
            rotary_interleaved=rotary_interleaved,
            norm_location=norm_location,
            norm_type=norm_type,
            attn_implementation=attn_implementation,
            rope_implementation=rope_implementation,
            **kwargs,
        )
        self.pooling = pooling


class BertPreTrainedModel(TitePreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"


class BertModel(TiteModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> TiteModelOutput:
        output = super().forward(
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = output.last_hidden_state
        if self.config.pooling == "first":
            return TiteModelOutput(hidden_states[:, [0]])
        elif self.config.pooling == "mean":
            return TiteModelOutput(hidden_states.mean(dim=1, keepdim=True))
        return output


class MaskLMHead(PreTrainingHead):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.lm_head = LMPredictionHead(config)

    def forward(self, output: TiteModelOutput, *args, **kwargs) -> torch.Tensor:
        return self.lm_head(output.last_hidden_state)

    def get_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        mlm_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        targets = torch.where(mlm_mask, input_ids, -100)
        return targets

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class BertForPreTraining(BertPreTrainedModel):

    _tied_weights_keys = [
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.bias",
        "heads.mlm.lm_head.decoder.weight",
        "heads.mlm.lm_head.decoder.bias",
    ]

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.heads = torch.nn.ModuleDict()
        self.heads["mlm"] = MaskLMHead(config)

    def get_output_embeddings(self):
        return self.lm_decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        original_input_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> TitePreTrainingOutput:
        output = self.bert(input_ids, attention_mask, output_hidden_states=True, output_attentions=output_attentions)

        losses = None
        if labels is not None:
            losses = {}
            for task, head in self.heads.items():
                logits = head(
                    output=output,
                    original_input_ids=original_input_ids,
                    attention_mask=attention_mask,
                )
                losses[task] = head.compute_loss(logits, labels[task])

        return TitePreTrainingOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions,
            losses=losses,
        )
