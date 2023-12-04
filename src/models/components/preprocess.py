import torch

def tensor_stack_extract(batch, domain="features", key="audios"):
    audios = torch.stack([x[1][domain][key]for x in batch], dim=0)
    return audios

def domain_stack_extract(batch, domain="features"):
    keys = batch[1][domain].keys()
    out_dict = dict()
    for key in keys:
        out_dict[key] = tensor_stack_extract(batch, domain, key)
    return out_dict
    
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
