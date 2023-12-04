import torch

def audio_extract(batch):
    audios = torch.stack([x[1]["audios"]for x in batch], dim=0)
    return audios

if __name__ == "__main__":
    import hydra
    import rootutils
    from lightning import LightningDataModule
    from omegaconf import OmegaConf

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    from src.data.base_datamodule import DatapipeModule

    path = root / "configs" / "data" / "mimii_due_classify.yaml"
    cfg = OmegaConf.load(path)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
    for name, loader in zip(
        ["train", "val"], [datamodule.train_dataloader(), datamodule.val_dataloader()]
    ):
        for i, batch in enumerate(loader):
            if i > 10:
                break
            # print(f"{i}th batch in {name}: {batch}")
            audios = audio_extract(batch)
            print(f"{i}th audios batch in {name}: {audios.shape}")
