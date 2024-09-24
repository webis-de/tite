import torch
from torch import Tensor
from torch.nn import Module
from transformers import BertConfig, BertForMaskedLM
from transformers.activations import ACT2FN

from .model import TiteConfig, TiteLayer


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
