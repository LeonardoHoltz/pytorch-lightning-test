from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_codebase.models import MnistSimpleModel
from lightning_codebase.datasets import MnistDataModule
import configs.mnist_simple_config as mnist_simple_config



def main():
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger("tb_logs, ")
    # Datamodule
    datamodule = MnistDataModule(
        data_dir=mnist_simple_config.data_dir,
        batch_size=mnist_simple_config.batch_size,
        num_workers=mnist_simple_config.num_workers,
    )

    # Initialize network
    model = MnistSimpleModel(
        input_size=mnist_simple_config.input_size, num_classes=mnist_simple_config.num_classes, learning_rate=mnist_simple_config.learning_rate
    ).to(device)

    # Trainer
    trainer = L.Trainer(
        accelerator=mnist_simple_config.accelerator,
        devices=mnist_simple_config.devices,
        min_epochs=1,
        max_epochs=mnist_simple_config.num_epochs,
        precision=mnist_simple_config.precision,
        callbacks=[RichProgressBar(leave=True), RichModelSummary()],
    )

    # Training and evaluation
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    # TODO: Qualitative prediction results


if __name__ == "__main__":
    main()
