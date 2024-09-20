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
        pooling: bool = False,
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
            **kwargs
        )
        self.pooling = pooling


class BertModel(TiteModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def tie_decoder_weights(self, output_embeddings: Module):
        self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = super().forward(input_ids, attention_mask)
        if self.config.pooling:
            return hidden_states[:, [0]]
        return hidden_states


class HFBertModel(Module):

    def __init__(
        self, model_name_or_path: str | None = None, config: HFBertConfig | None = None, pooling: bool = False
    ) -> None:
        super().__init__()
        if model_name_or_path is None:
            if config is None:
                raise ValueError("Either model_name_or_path or config must be provided")
            self._model = HFBert(config)
        else:
            self._model = HFBert.from_pretrained(model_name_or_path, config=config)
        self.config = self._model.config
        self.config.last_hidden_size = self.config.hidden_size
        self.pooling = pooling

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, token_type_ids: Tensor | None = None
    ) -> Tensor:
        hidden_states = self._model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        ).last_hidden_state
        if self.pooling:
            hidden_states = hidden_states[:, [0]]
        return hidden_states

    def save_pretrained(self, *args, **kwargs):
        self._model.save_pretrained(*args, **kwargs)
