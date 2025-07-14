from pathlib import Path
from typing import Sequence, Tuple

import torch
from lightning_ir import (
    BiEncoderConfig,
    BiEncoderModel,
    BiEncoderModule,
    BiEncoderOutput,
    CrossEncoderModule,
    SearchConfig,
)
from lightning_ir.loss.loss import LossFunction, RegularizationLossFunction


class TiteBiEncoderModule(BiEncoderModule):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: BiEncoderConfig | None = None,
        model: BiEncoderModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        index_dir: Path | None = None,
        search_config: SearchConfig | None = None,
        compile: bool = True,
    ):
        super().__init__(
            model_name_or_path, config, model, loss_functions, evaluation_metrics, index_dir, search_config
        )
        if compile:
            self.model.compile(dynamic=True)


class FLOPSRegularization(RegularizationLossFunction):

    def __init__(
        self,
        query_num_non_zero: int = 0,
        doc_num_non_zero: int = 0,
        query_weight: float = 0.01,
        doc_weight: float = 0.01,
    ) -> None:
        super().__init__(query_weight, doc_weight)
        self.query_num_non_zero = query_num_non_zero
        self.doc_num_non_zero = doc_num_non_zero

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_flops = torch.sum(torch.mean(torch.abs(query_embeddings), dim=0) ** 2)
        query_loss = torch.abs(query_flops - self.query_num_non_zero)
        doc_flops = torch.sum(torch.mean(torch.abs(doc_embeddings), dim=0) ** 2)
        doc_loss = torch.abs(doc_flops - self.doc_num_non_zero)
        loss = self.query_weight * query_loss + self.doc_weight * doc_loss
        return loss


class L1Regularization(RegularizationLossFunction):

    def __init__(
        self,
        query_num_non_zero: int = 0,
        doc_num_non_zero: int = 0,
        query_weight: float = 0.01,
        doc_weight: float = 0.01,
    ) -> None:
        super().__init__(query_weight, doc_weight)
        self.query_num_non_zero = query_num_non_zero
        self.doc_num_non_zero = doc_num_non_zero

    def compute_loss(self, output: BiEncoderOutput) -> torch.Tensor:
        query_embeddings, doc_embeddings = self.process_embeddings(output)
        query_loss = torch.abs(query_embeddings.norm(p=1, dim=-1) - self.query_num_non_zero).mean()
        doc_loss = torch.abs(doc_embeddings.norm(p=1, dim=-1) - self.doc_num_non_zero).mean()
        loss = self.query_weight * query_loss + self.doc_weight * doc_loss
        return loss
