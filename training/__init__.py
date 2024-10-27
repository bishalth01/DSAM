from operator import mod
import sys
#from training import Train
from omegaconf import DictConfig
from typing import List
import torch
from components import LRScheduler
import logging
import torch.utils.data as utils
from training.FBNettraining import FBNetTrain
from training.maintraining import Train
from training.thctraining import THCTrain
from training.thctraining import BrainGNNTHCTrain
from training.braingnntraining import BrainGNNTrain
from training.braingnndsamtraining import BrainGNNDSAMTrain



def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger) -> Train:

    
    if config.model.name=="SpatioTemporalModel":
        train = "BrainGNNDSAMTrain"
    elif config.model.name=="BrainGNN":
        train= "BrainGNNTrain"
    else:
        train = config.model.get("train", None)
        if not train:
            train = config.training.name
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger)
    # return eval(train)(train_config=config,
    #                    model=model,
    #                    optimizers=optimizers,
    #                    lr_schedulers=lr_schedulers,
    #                    dataloaders=dataloaders,
    #                    logger=logger)



