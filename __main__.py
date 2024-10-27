import sys
from datetime import datetime
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig, open_dict
from dataset import dataset_factory
from models import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import torch
import numpy as np
import random

def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg, dataloaders)

    # #Run on multiple GPUs if exists
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)  # Wrap your model with DataParallel

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    for _ in range(cfg.repeat_time):
        run = wandb.init()
        print(cfg)
        
        #For Replicability
        torch.manual_seed(1)
        np.random.seed(1111)
        random.seed(1111)
        torch.cuda.manual_seed_all(1111)

        #Training model
        model_training(cfg)

        run.finish()


if __name__ == '__main__':
    main()
