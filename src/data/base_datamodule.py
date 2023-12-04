import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2


def basic_collect_fn(batch):
    tensordict = torch.stack([x[0] for x in batch], dim=0)
    meta_list = [x[1] for x in batch]
    return tensordict, meta_list


class DatasetModule(LightningDataModule):
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


class DatapipeModule(LightningDataModule):
    def __init__(
        self, datapipes, reading_service, batch_size=5, using_dataloader2=False
    ):
        super().__init__()
        self.datapipes = datapipes
        self.reading_service = reading_service
        self.batch_size = batch_size
        self.dataloader = DataLoader2 if using_dataloader2 else DataLoader

    def train_dataloader(self):
        return self.x_dataloader("train")

    def val_dataloader(self):
        return self.x_dataloader("val")

    def test_dataloader(self):
        return self.x_dataloader("test")

    def x_dataloader(self, datasetname):
        if (datapipe := self.datapipes[datasetname]) is None:
            return None

        return self.dataloader(datapipe, reading_service=self.reading_service)


if __name__ == "__main__":
    import hydra
    import rootutils
    from lightning import LightningDataModule
    from omegaconf import OmegaConf

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
    path = root / "configs" / "data" / "mimii_due_classify.yaml"
    cfg = OmegaConf.load(path)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
    print(f"{datamodule=}")
    for name, loader in zip(
        ["train", "val"], [datamodule.train_dataloader(), datamodule.val_dataloader()]
    ):
        for i, batch in enumerate(loader):
            if i > 10:
                break
            print(f"{i}th batch in {name}: {batch}")
