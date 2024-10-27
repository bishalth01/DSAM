import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd

# from dataset.atlases.dataset_atlas_new import ABCDDatasetAtlas
from dataset.atlases.dataset_atlas_umah_zscored import ABCDDatasetAtlas
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abcd_data_dsam(cfg: DictConfig):
    #Load the dataset altogether
    dataset = ABCDDatasetAtlas(root=cfg.dataset.name,
                                target_var="gender",
                                num_nodes=100,
                                threshold=cfg.model.threshold,
                                connectivity_type="fmri",
                                normalisation="subject_norm",
                                analysis_type="",
                                encoding_strategy="",
                                time_length=360,
                                edge_weights=True)
    
    # cfg.dataset.edge_index = dataset.data.edge_index
    # cfg.dataset.edge_attr = dataset.data.edge_attr
    with open_dict(cfg):
        cfg.dataset.node_sz = 100
        cfg.dataset.node_feature_sz = 100
        cfg.dataset.timeseries_sz = dataset.data.x.shape[-1]

    return [],[],[], dataset

def load_abcd_data(cfg: DictConfig):

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    label_df = pd.read_csv(cfg.dataset.label)

    with open(cfg.dataset.node_id, 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines]

    with open(cfg.dataset.seires_id, 'r') as f:
        lines = f.readlines()
        ts_id = [line[:-1] for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2gender = dict(zip(label_df['id'], label_df['sex']))

    final_timeseires, final_label, final_pearson = [], [], []

    for ts, l in zip(ts_data, ts_id):
        if l in id2gender and l in id2pearson:
            if np.any(np.isnan(id2pearson[l])) == False:
                final_timeseires.append(ts)
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels
