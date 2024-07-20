from abc import ABC, abstractmethod

import torch


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


class LinearLRSchedulerWithLinearWarmup(WarmupLRScheduler, LinearSchedulerWithLinearWarmup):
    pass


class ConstantLRSchedulerWithLinearWarmup(WarmupLRScheduler, ConstantSchedulerWithLinearWarmup):
    pass


LR_SCHEDULERS = [LinearLRSchedulerWithLinearWarmup, ConstantLRSchedulerWithLinearWarmup]
