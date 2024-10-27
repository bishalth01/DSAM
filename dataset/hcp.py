import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
import sys
# from dataset.atlases.dataset_atlas import HCPDatasetAtlas
# from dataset.atlases.dataset_atlas_new import HCPDatasetAtlas
# from dataset.atlases.dataset_atlas_umah_zscored_regression import HCPDatasetAtlas
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_hcp_data_dsam(cfg: DictConfig):
    #Load the dataset altogether
    dataset = HCPDatasetAtlas(root=cfg.dataset.name,
                                target_var="gender",
                                num_nodes=100,
                                threshold=100,#cfg.model.threshold,
                                connectivity_type="fmri",
                                normalisation="subject_norm",
                                analysis_type="",
                                encoding_strategy="",
                                time_length=1200,
                                edge_weights=True)

    # dataset = HCPDatasetAtlas(root=cfg.dataset.name,
    #                             target_var="CogTotalComp_AgeAdj",
    #                             num_nodes=100,
    #                             threshold=100,#cfg.model.threshold,
    #                             connectivity_type="fmri",
    #                             normalisation="subject_norm",
    #                             analysis_type="",
    #                             encoding_strategy="",
    #                             time_length=1200,
    #                             edge_weights=True)
    
    # cfg.dataset.edge_index = dataset.data.edge_index
    # cfg.dataset.edge_attr = dataset.data.edge_attr

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    label_df = np.load(cfg.dataset.label)


    pearson_id=np.load(cfg.dataset.node_id)
    ts_id = np.load(cfg.dataset.seires_id)

    id2pearson = dict(zip(pearson_id, pearson_data))

    # id2gender = dict(zip(label_df['id'], label_df['sex']))
    id2gender = dict(zip(pearson_id, label_df))


    final_timeseires, final_label, final_pearson = [], [], []


    for ts, l in zip(ts_data, ts_id):
        if l in id2gender and l in id2pearson:
            if np.any(np.isnan(id2pearson[l])) == False:
                final_timeseires.append(ts)
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df)

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

    return final_timeseires, final_pearson, labels, dataset


def load_hcp_data(cfg: DictConfig):

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    label_df = np.load(cfg.dataset.label)

    


    pearson_id=np.load(cfg.dataset.node_id)
    ts_id = np.load(cfg.dataset.seires_id)

    id2pearson = dict(zip(pearson_id, pearson_data))

    # id2gender = dict(zip(label_df['id'], label_df['sex']))
    id2gender = dict(zip(pearson_id, label_df))


    final_timeseires, final_label, final_pearson = [], [], []


    for ts, l in zip(ts_data, ts_id):
        if l in id2gender and l in id2pearson:
            if np.any(np.isnan(id2pearson[l])) == False:
                final_timeseires.append(ts)
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df)

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
