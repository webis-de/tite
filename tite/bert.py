from typing import Literal

from torch import Tensor
from torch.nn import Module
from transformers import BertConfig as HFBertConfig
from transformers import BertModel as HFBert

from .model import TiteConfig, TiteModel


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
        positional_embedding_type: Literal["absolute", "ALiBi"] = "ALiBi",
        pooling: Literal["mean", "first"] | None = None,
        unpadding: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size,
            num_hidden_layers,
            (hidden_size,) * num_hidden_layers,
            (num_attention_heads,) * num_hidden_layers,
            (intermediate_size,) * num_hidden_layers,
            (None,) * num_hidden_layers,
            (None,) * num_hidden_layers,
            dropout_prob,
            max_position_embeddings,
            initializer_range,
            layer_norm_eps,
            pad_token_id,
            hidden_act,
            positional_embedding_type,
            unpadding,
            **kwargs
        )
        self.pooling = pooling


class BertModel(TiteModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = super().forward(input_ids, attention_mask)
        if self.config.pooling == "first":
            return hidden_states[:, [0]]
        elif self.config.pooling == "mean":
            return hidden_states.mean(dim=1, keepdim=True)
        return hidden_states


class PreTrainedBertModel(BertModel):

    def __init__(self, model_name_or_path: str, pooling: Literal["mean", "first"] | None = None):
        config = HFBertConfig.from_pretrained(model_name_or_path)
        bert_config = BertConfig(**config.to_dict(), pooling=pooling)
        bert_config["positional_embeddings_type"] = config.position_embeddings_type
        super().__init__(bert_config)


class HFBertModel(Module):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: HFBertConfig | None = None,
        pooling: Literal["mean", "first"] | None = None,
    ) -> None:
        super().__init__()
        if model_name_or_path is None:
            if config is None:
                raise ValueError("Either model_name_or_path or config must be provided")
            self.model = HFBert(config)
        else:
            self.model = HFBert.from_pretrained(model_name_or_path, config=config)
        self.config = self.model.config
        self.config.last_hidden_size = self.config.hidden_size
        self.pooling = pooling

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, token_type_ids: Tensor | None = None
    ) -> Tensor:
        hidden_states = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        ).last_hidden_state
        if self.pooling == "first":
            return hidden_states[:, [0]]
        elif self.pooling == "mean":
            return hidden_states.mean(dim=1, keepdim=True)
        return hidden_states

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)
