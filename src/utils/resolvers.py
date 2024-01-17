from omegaconf import OmegaConf


def sum_all(*x):
    return sum(x)


def register_new_resolver():
    OmegaConf.register_new_resolver("sum_all", sum_all)
