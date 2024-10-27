from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data, load_abcd_data_dsam
from .hcp import load_hcp_data
from .hcp import load_hcp_data_dsam
from .abide import load_abide_data
from .dataloader import init_dataloader, init_dataloader_with_edges, init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide', 'hcp']

    
    
    if cfg.model.name=='SpatioTemporalModel' or cfg.model.name=='BrainGNN'  or cfg.model.name=='DISMSpatioTemporalModel':
        datasets = eval(
        f"load_{cfg.dataset.name}_data_dsam")(cfg)
        dataloaders = init_dataloader_with_edges(cfg, *datasets) \
    
    else:
        datasets = eval(
        f"load_{cfg.dataset.name}_data_dsam")(cfg)
        # f"load_{cfg.dataset.name}_data")(cfg)
        dataloaders = init_dataloader_with_edges(cfg, *datasets) \
            if cfg.dataset.stratified \
            else init_dataloader_with_edges(cfg, *datasets)

    return dataloaders
