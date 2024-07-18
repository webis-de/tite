from torch import Tensor
from torch.nn import Module

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
        hidden_act: str = "gelu_new",
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
        return super().forward(input_ids, attention_mask)
