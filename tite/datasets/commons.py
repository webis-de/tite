import os


def seed_or_none(seed: int | None = None) -> int | None:
    if seed is not None:
        return seed
    seed = os.environ.get("PL_GLOBAL_SEED", default=None)
    return int(seed) if seed is not None else None
