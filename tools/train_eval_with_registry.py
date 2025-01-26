import argparse
from lightning_codebase.utils import load_config

import torch
from lightning_codebase.models import create_model
from lightning_codebase.datasets import create_datamodule
from lightning_codebase.trainers import create_trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to the YAML training configuration file')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    
    datamodule = create_datamodule(cfg.datamodule)
    model = create_model(cfg.model).to(device)
    trainer = create_trainer(cfg.trainer)

    # Training and evaluation
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    # TODO: Qualitative prediction results


if __name__ == "__main__":
    main()
