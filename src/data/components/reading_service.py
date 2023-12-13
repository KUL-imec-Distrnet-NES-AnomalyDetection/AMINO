import torch.distributed as dist
from torchdata.dataloader2 import \
    DistributedReadingService as _DistributedReadingService
from torchdata.dataloader2.graph import DataPipe

from src import utils

log = utils.get_pylogger(__name__)


class DistributedReadingService(_DistributedReadingService):
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        if not (dist.is_available() and dist.is_initialized()):
            log.warning("Torch Distributed is required to be initialized")
            return datapipe
        return super().initialize(datapipe)

    def initialize_iteration(
        self,
        seed_generator,
        iter_reset_fn=None,
    ):
        if not (dist.is_available() and dist.is_initialized()):
            return None
        return super().initialize_iteration(seed_generator, iter_reset_fn)
