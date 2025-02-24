import math
from abc import ABC, abstractmethod

import torch
from torch.optim.optimizer import Optimizer as Optimizer


class LambdaWarmupScheduler(ABC):
    def __init__(
        self,
        num_warmup_steps: int,
        num_delay_steps: int = 0,
        *args,
        **kwargs,
    ) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_delay_steps = num_delay_steps
        super().__init__(*args, **kwargs)

    @abstractmethod
    def value_lambda(self, current_step: int) -> float: ...

    def check_delay(self, current_step: int) -> bool:
        return current_step < self.num_delay_steps

    def check_warmup(self, current_step: int) -> bool:
        return current_step < self.num_warmup_steps + self.num_delay_steps


class LinearSchedulerWithLinearWarmup(LambdaWarmupScheduler):

    def __init__(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        final_value: float = 0.0,
        num_delay_steps: int = 0,
        *args,
        **kwargs,
    ) -> None:
        self.num_training_steps = num_training_steps
        self.final_value = final_value
        super().__init__(num_warmup_steps, num_delay_steps, *args, **kwargs)

    def value_lambda(self, current_step: int) -> float:
        if self.check_delay(current_step):
            return 0.0
        if self.check_warmup(current_step):
            return (current_step - self.num_delay_steps) / self.num_warmup_steps
        current_step = current_step - self.num_delay_steps - self.num_warmup_steps
        remaining_steps = self.num_training_steps - self.num_delay_steps - self.num_warmup_steps
        step_size = (1 - self.final_value) / remaining_steps
        return max(self.final_value, 1 - step_size * current_step)


class SigmoidSchedulerWithLinearWarmup(LambdaWarmupScheduler):

    def __init__(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        final_value: float = 0.0,
        *args,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self.num_training_steps = num_training_steps
        self.final_value = final_value
        super().__init__(num_warmup_steps, *args, verbose=verbose, **kwargs)

    def value_lambda(self, current_step: int) -> float:
        if self.check_delay(current_step):
            return 0.0
        if self.check_warmup(current_step):
            return (current_step - self.num_delay_steps) / self.num_warmup_steps
        current_step = current_step - self.num_delay_steps - self.num_warmup_steps
        remaining_steps = self.num_training_steps - self.num_delay_steps - self.num_warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * current_step / remaining_steps))
        factor = q + self.final_value * (1 - q)
        return factor


class ConstantSchedulerWithLinearWarmup(LambdaWarmupScheduler):
    def value_lambda(self, current_step: int) -> float:
        if self.check_delay(current_step):
            return 0.0
        if self.check_warmup(current_step):
            return (current_step - self.num_delay_steps) / self.num_warmup_steps
        return 1.0


class WarmupLRScheduler(LambdaWarmupScheduler, torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        *args,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        last_epoch = -1
        self.interval = "step"
        super().__init__(
            *args,
            optimizer=optimizer,
            lr_lambda=self.value_lambda,
            num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch,
            verbose=verbose,
            **kwargs,
        )


class LARSScheduler(WarmupLRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        batch_size: int,
        base_factor: float | None = None,
        *args,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.base_factor = base_factor
        super().__init__(optimizer, num_warmup_steps, *args, verbose=verbose, **kwargs)

    def value_lambda(self, current_step: int) -> float:
        base_factor = self.batch_size / 256 if self.base_factor is None else self.base_factor
        max_steps = self.num_training_steps
        warmup_steps = self.num_warmup_steps
        if current_step < warmup_steps:
            factor = base_factor * current_step / warmup_steps
        else:
            current_step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * current_step / max_steps))
            end_factor = base_factor * 0.001
            factor = base_factor * q + end_factor * (1 - q)
        return factor


class LinearLRSchedulerWithLinearWarmup(WarmupLRScheduler, LinearSchedulerWithLinearWarmup):
    pass


class ConstantLRSchedulerWithLinearWarmup(WarmupLRScheduler, ConstantSchedulerWithLinearWarmup):
    pass


class SigmoidLRSchedulerWithLinearWarmup(WarmupLRScheduler, SigmoidSchedulerWithLinearWarmup):
    pass


LR_SCHEDULERS = [
    LinearLRSchedulerWithLinearWarmup,
    ConstantLRSchedulerWithLinearWarmup,
    LARSScheduler,
    SigmoidLRSchedulerWithLinearWarmup,
]
