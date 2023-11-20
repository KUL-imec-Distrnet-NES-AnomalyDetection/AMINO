import os

import torch
import torchaudio
from tensordict import TensorDict
from torchdata.datapipes.iter import IterableWrapper

from src.data.components.base_datapipes import DPAudioRead


def mimii_due_file_list_generator(path, is_train="True"):
    out_list = list()
    if is_train:
        for file in os.listdir(path):
            wav_split = file.split("_")
            assert len(wav_split) > 6, f"the file name {file} in {path} is not correct"
            out_list.append(
                {
                    "path": os.path.join(path, file),
                    "section": int(wav_split[1]),
                    "domain": wav_split[2],
                    "label": wav_split[4],
                    "id": wav_split[5],
                    "strength": wav_split[7],
                }
            )
    else:
        for file in os.listdir(path):
            wav_split = file.split("_")
            assert len(wav_split) > 5, f"the file name {file} in {path} is not correct"
            out_list.append(
                {
                    "path": os.path.join(path, file),
                    "section": int(wav_split[1]),
                    "domain": wav_split[2],
                    "label": wav_split[4],
                    "id": wav_split[5],
                }
            )
    return out_list


class MimiidueDataset(torch.utils.data.Dataset):
    def __init__(self, path, is_train):
        super().__init__()
        self.file_list = mimii_due_file_list_generator(path, is_train)

    def __getitem__(self, index):
        item = self.file_list[index]
        audio, fs = torchaudio.load(item["path"])
        item["fs"] = fs
        tensordict = TensorDict({"audio": audio}, batch_size=1)
        return tensordict, item


def mimii_due_datapipe(path, is_train):
    data_list = mimii_due_file_list_generator(path, is_train)
    data_list = [[TensorDict(dict(), batch_size=1), x] for x in data_list]
    orginal_dp = IterableWrapper(data_list)
    return orginal_dp


if __name__ == "__main__":
    from functools import partial
    import rootutils

    from torchdata.datapipes.iter import Batcher, Shuffler

    from src.data.components.base_datapipes import build_datapipe

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    mimii_due_path = root / "data" / "mimii_due"
    select_path = mimii_due_path / "fan"
    tr_path = select_path / "train"
    cv_path = select_path / "source_test"

    tr_dataset = MimiidueDataset(tr_path, is_train=True)
    cv_dataset = MimiidueDataset(cv_path, is_train=False)
    for name, dataset in zip(["train", "val"], [tr_dataset, cv_dataset]):
        for i, item in enumerate(dataset):
            if i > 10:
                break
            print(f"{i}th item in {name} dataset: {item}")

    tr_datapipe = DPAudioRead(mimii_due_datapipe(tr_path, is_train=True))
    cv_datapipe = DPAudioRead(mimii_due_datapipe(cv_path, is_train=False))
    for name, datapipe in zip(["train", "val"], [tr_datapipe, cv_datapipe]):
        for i, item in enumerate(datapipe):
            if i > 10:
                break
            print(f"{i}th item in {name} datapipe: {item}")

    process_dps = [
        partial(Batcher, batch_size=5),
    ]
    cv_datapipe = build_datapipe([cv_datapipe], process_dps)
    process_dps.insert(0, partial(Shuffler, buffer_size=100))
    tr_datapipe = build_datapipe([tr_datapipe], process_dps)

    for name, datapipe in zip(["train", "val"], [tr_datapipe, cv_datapipe]):
        for i, item in enumerate(datapipe):
            if i > 10:
                break
            print(f"{i}th item in {name} datapipe: {item}")
