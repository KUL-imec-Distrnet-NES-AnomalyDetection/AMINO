import torchaudio
from tensordict import TensorDict
from torchdata.datapipes.iter import Concater, Mapper


class DPAudioRead(Mapper):
    def __init__(self, dp) -> None:
        super().__init__(dp, self.item_fn)

    def item_fn(self, item):
        audio, fs = torchaudio.load(item["path"])
        item["fs"] = fs
        tensordict = TensorDict({"audio": audio}, batch_size=1)
        return tensordict, item


def build_datapipe(source_datapipes, process_datapipes):
    dp = Concater(*source_datapipes)
    for process_dp in process_datapipes:
        dp = process_dp(dp)
    return dp
