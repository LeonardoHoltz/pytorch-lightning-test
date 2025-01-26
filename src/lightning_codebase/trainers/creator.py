from ..utils.registry import Registry
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary

TRAINERS = Registry("trainers")
CALLBACKS = Registry("callbacks")

TRAINERS.register(module=L.Trainer)

CALLBACKS.register(module=RichProgressBar)
CALLBACKS.register(module=RichModelSummary)

def create_callbacks(callbacks_cfg):
    def create_callback(cfg):
        return CALLBACKS.create(cfg)

    callbacks = []
    for callback in callbacks_cfg:
        callbacks.append(create_callback(callback))
    return callbacks

def create_trainer(cfg):
    cfg_callbacks_dict = cfg.get("callbacks", None)
    if cfg_callbacks_dict:
        cfg["callbacks"] = create_callbacks(cfg_callbacks_dict)
    return TRAINERS.create(cfg)
