from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer
from .DSAM.dsam_interpretable import SpatioTemporalModel
from .BrainGNN.braingnn import BrainGNN


def model_factory(config: DictConfig, dataloaders):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    elif config.model.name in ["SpatioTemporalModel"]:
        return eval(config.model.name)(config, dataloaders)#.cuda()   
    else:
        return eval(config.model.name)(config)#.cuda()
