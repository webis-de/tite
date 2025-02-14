from typing import Literal

from torch import Tensor
from torch.nn import Module
from transformers import BertConfig as HFBertConfig
from transformers import BertModel as HFBert

# from .legacy import TiteConfig, TiteModel, TiteModelOutput
from .model import TiteConfig, TiteModel, TiteModelOutput


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
        **kwargs
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
            **kwargs
        )
        self.pooling = pooling


class BertModel(TiteModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, output_hidden_states: bool = False
    ) -> TiteModelOutput:
        output = super().forward(input_ids, attention_mask, output_hidden_states=output_hidden_states)
        hidden_states = output.last_hidden_state
        if self.config.pooling == "first":
            return TiteModelOutput(hidden_states[:, [0]])
        elif self.config.pooling == "mean":
            return TiteModelOutput(hidden_states.mean(dim=1, keepdim=True))
        return output


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
