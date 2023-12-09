import torch

from torchdata.datapipes.iter import Mapper

class LabelAdder(Mapper):
    def __init__(self, dp, label):
        super().__init__(dp, self.item_func)
        self.label = torch.tensor(label).reshape(1, -1)
    
    def item_func(self, item):
        item[1].set(("labels", "classify"), self.label)
        return item


if __name__ == "__main__":
    import rootutils

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    from torchdata.datapipes.iter import Concater, Shuffler
    
    from src.data.components.mimii_due import mimii_due_datapipe
    
    mimii_due_path = root / "data" / "mimii_due"

    select_dict = {
        "fan": 0,
        "gearbox": 1,
        "pump": 2,
        "slider": 3,
        "valve": 4,
    }

    tr_datapipes = list()
    cv_datapipes = list()
    ts_datapipes = list()
    for name, label in select_dict.items():
        select_path = mimii_due_path / name
        tr_path = select_path / "train"
        cv_path = select_path / "source_test"
        ts_path = select_path / "target_test"

        tr_datapipe = mimii_due_datapipe(tr_path, is_train=True)
        cv_datapipe = mimii_due_datapipe(cv_path, is_train=False)
        ts_datapipe = mimii_due_datapipe(ts_path, is_train=False)

        tr_datapipe = LabelAdder(tr_datapipe, label)
        cv_datapipe = LabelAdder(cv_datapipe, label)
        ts_datapipe = LabelAdder(ts_datapipe, label)

        tr_datapipes.append(tr_datapipe)
        cv_datapipes.append(cv_datapipe)
        ts_datapipes.append(ts_datapipe)

    print("*" * 20)
    for name, datapipes in zip(
        ["train", "val", "test"],
        [tr_datapipes, cv_datapipes, ts_datapipes],
    ):
        for i, datapipe in enumerate(datapipes):
            temp_iter = iter(datapipe)
            for j, batch in enumerate(temp_iter):
                if j > 10:
                    break
                print(f"{name}, {i}, {j}: {batch}")
    
    print("*" * 20)
    tr_datapipe = Concater(*tr_datapipes)
    tr_datapipe = Shuffler(tr_datapipe, buffer_size=20000)
    temp_iter = iter(tr_datapipe)
    for i, batch in enumerate(temp_iter):
        print(f"{i}: {batch}")

    # print("*" * 20)
    # tr_datapipe = DPAudioRead(tr_datapipe)
    #     for i, batch in enumerate(tr_datapipe):
    #     print(f"{i}: {batch}")
