import torch


class LARS(torch.optim.Optimizer):
    # https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L224
    def __init__(
        self,
        params,
        lr_weights,
        lr_biases,
        weight_decay=1e-6,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    ):
        param_weights = []
        param_biases = []
        for param in params:
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        defaults = dict(
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(
            [{"params": param_weights, "lr": lr_weights}, {"params": param_biases, "lr": lr_biases}], defaults
        )

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0, torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one), one
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

        return loss
