import gc
from typing import Optional, Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


def gc_cuda():
    """Garbage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ScheduledGarbageCollector(Callback):
    """Disable automatic garbage collection and collect garbage at interval.

    Args:
        gen_1_batch_interval(int, optional): Number of batches between calls to gc.collect(1)
    """

    def __init__(
        self,
        gen_1_batch_interval: Optional[int] = None,
        eval_keep_disabled: bool = True,
    ):
        self.gen_1_batch_interval = gen_1_batch_interval
        self.eval_keep_disabled = eval_keep_disabled

        self.gc_init_state = None


    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        # cache if automatic garbage collection is enabled; reset at on_train_end
        self.gc_init_state = gc.isenabled()

        # disable automatic garbage collection
        gc.disable()
        gc.collect(1)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        # reset automatic garbage collection at on_train_end
        if self.gc_init_state:
            gc.enable()
        else:
            gc.disable()

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ):
        global_step = trainer.global_step

        if global_step % self.gen_1_batch_interval == 0:
            gc.collect(1)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        gc_cuda()
        if not self.eval_keep_disabled:
            gc.enable()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.eval_keep_disabled:
            gc.disable()
        gc_cuda()
