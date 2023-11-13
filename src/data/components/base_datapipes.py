import torchaudio
from torchdata.datapipes.iter import Mapper
from tensordict import TensorDict

class DPAudioRead(Mapper):
    def __init__(self, dp) -> None:
        super().__init__(dp, self.item_fn)

    def item_fn(self, item):
        audio, fs = torchaudio.load(item["path"])
        item["fs"] = fs
        tensordict = TensorDict({"audio": audio}, batch_size=1)
        return tensordict, item
