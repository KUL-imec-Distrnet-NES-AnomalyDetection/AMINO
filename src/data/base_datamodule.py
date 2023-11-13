import torch
from lightning import LightningDataModule


def basic_collect_fn(batch):
    tensordict = torch.stack([x[0] for x in batch], dim=0)
    meta_list = [x[1] for x in batch]
    return tensordict, meta_list


class DataModule(LightningDataModule):
    def __init__(self, datasets, dataloader):
        super().__init__()
        self.datasets = datasets
        self.dataloader = dataloader

    def train_dataloader(self):
        return self.x_dataloader("train")

    def val_dataloader(self):
        return self.x_dataloader("val")

    def test_dataloader(self):
        return self.x_dataloader("test")

    def x_dataloader(self, datasetname):
        if self.datasets[datasetname] is None:
            return None
        return self.dataloader(self.datasets[datasetname])


if __name__ == "__main__":
    import rootutils
    import hydra
    from lightning import LightningDataModule
    from omegeconf import OmegaConf

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    # ------------------------------------------------------------------------------------ #
    # the setup_root above is equivalent to:
    # - adding project root dir to PYTHONPATH
    #       (so you don't need to force user to install project as a package)
    #       (necessary before importing any local modules e.g. `from src import utils`)
    # - setting up PROJECT_ROOT environment variable
    #       (which is used as a base for paths in "configs/paths/default.yaml")
    #       (this way all filepaths are the same no matter where you run the code)
    # - loading environment variables from ".env" in root dir
    #
    # you can remove it if you:
    # 1. either install project as a package or move entry files to project root dir
    # 2. set `root_dir` to "." in "configs/paths/default.yaml"
    #
    # more info: https://github.com/ashleve/rootutils
    # ------------------------------------------------------------------------------------ #

    from src import utils

    log = utils.get_pylogger(__name__)

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    path = root / "configs" / "data" / "mimii_due.yaml"
    cfg = OmegaConf.load(path)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
