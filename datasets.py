from abc import ABC
import nibabel as nib
import networkx as nx
#import nolds
import numpy as np
import pandas as pd
import torch
from nilearn.connectome import ConnectivityMeasure
from numpy.random import default_rng
from scipy.stats import mstats
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch_geometric.data import InMemoryDataset, Data
from typing import List
import scipy.io as sio
from utils import Normalisation, ConnType, AnalysisType, EncodingStrategy, DatasetType
import nibabel as nib

HCP_DEMOGRAPHICS_PATH = '/data/qneuromark/Data/HCP/Data_info/HCP_demo.csv'

ABCD_DEMOGRAPHICS_PATH = '/data/users3/bthapaliya/SpatioTemporalBishalGNN/ABCDPhenotypeInfo.csv'



def threshold_adj_array(adj_array: np.ndarray, threshold: int, num_nodes: int) -> np.ndarray:
    num_to_filter: int = int((threshold / 100.0) * (num_nodes * (num_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj_array[np.tril_indices(num_nodes)] = 0

    # Following code is similar to bctpy
    indices = np.where(adj_array)
    sorted_indices = np.argsort(adj_array[indices])[::-1]
    adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

    # Just to get a symmetrical matrix
    adj_array = adj_array + adj_array.T

    # Diagonals need connection of 1 for graph operations
    adj_array[np.diag_indices(num_nodes)] = 1.0

    return adj_array



###################################################UTILS Functions #################################################################


def normalise_timeseries(timeseries: np.ndarray, normalisation: Normalisation) -> np.ndarray:
    """
    :param normalisation:
    :param timeseries: In  format TS x N
    :return:
    """
    flatten_timeseries = timeseries.flatten().reshape(-1, 1)
    scaler = RobustScaler().fit(flatten_timeseries)
    timeseries = scaler.transform(flatten_timeseries).reshape(timeseries.shape).T

    return timeseries


def create_thresholded_graph(adj_array: np.ndarray, threshold: int, num_nodes: int) -> nx.DiGraph:
    adj_array = threshold_adj_array(adj_array, threshold, num_nodes)

    return nx.from_numpy_array(adj_array, create_using=nx.DiGraph)


class BrainDataset(InMemoryDataset, ABC):
    def __init__(self, root, target_var: str, num_nodes: int, threshold: int, connectivity_type: ConnType,
                 normalisation: Normalisation, analysis_type: AnalysisType, edge_weights: bool, time_length: int,
                 encoding_strategy: EncodingStrategy,
                 transform=None, pre_transform=None):
        if threshold < 0 or threshold > 100:
            print("NOT A VALID threshold!")
            exit(-2)
        if normalisation not in [Normalisation.NONE, Normalisation.ROI, Normalisation.SUBJECT]:
            print("BrainDataset not prepared for that normalisation!")
            exit(-2)

        self.target_var: str = target_var
        self.num_nodes: int = num_nodes
        self.connectivity_type: ConnType = connectivity_type
        self.time_length: int = time_length
        self.threshold: int = threshold
        self.normalisation: Normalisation = normalisation
        self.analysis_type: AnalysisType = analysis_type
        self.encoding_strategy: EncodingStrategy = encoding_strategy
        self.include_edge_weights: bool = edge_weights

        super(BrainDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass


###########################################################END#########################################################################


class HCPDatasetAtlas(BrainDataset):
    def __init__(self, root, target_var: str, num_nodes: int, threshold: int, connectivity_type: ConnType,
                 normalisation: Normalisation, analysis_type: AnalysisType, edge_weights: bool, time_length: int = 1200,
                 encoding_strategy: EncodingStrategy = EncodingStrategy.NONE,
                 transform=None, pre_transform=None):

        if target_var not in ['gender']:
            print("HCPDataset not prepared for that target_var!")
            exit(-2)
        if connectivity_type not in [ConnType.STRUCT, ConnType.FMRI]:
            print("HCPDataset not prepared for that connectivity_type!")
            exit(-2)
            
        self._idx_to_filter = np.arange(0, 400)
        self.threshold = threshold
        self.ts_split_num: int = int(4800 / time_length)
        self.info_df = pd.read_csv(HCP_DEMOGRAPHICS_PATH, sep=",").set_index('Subject')
        # self.nodefeats_df = pd.read_csv('meta_data/node_features_powtransformer.csv', index_col=0)

        super(HCPDatasetAtlas, self).__init__(root, target_var=target_var, num_nodes=num_nodes, threshold=threshold,
                                         connectivity_type=connectivity_type, normalisation=normalisation,
                                         analysis_type=analysis_type, time_length=time_length,
                                         encoding_strategy=encoding_strategy, edge_weights=edge_weights,
                                         transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["stgcn_data_hcp_brain_custom_schaefer_100_thres_{0}_withpearsonr_check.dataset".format(self.threshold)]
        # return ["stgcn_data_hcp_brain_custom_schaefer_100_thres_{self.threshold}_withpearsonr_check.dataset"]

    def __create_data_object(self, person: int, ts: np.ndarray, ind: int, edge_attr: torch.Tensor,
                             edge_index: torch.Tensor, pearson_corr: np.ndarray):
        #assert ts.shape[0] > ts.shape[1]  # TS > N

        timeseries = normalise_timeseries(timeseries=ts, normalisation=self.normalisation)

        x = torch.tensor(timeseries, dtype=torch.float)
        if self.target_var == 'gender':
            gender_map = {'M': 0, 'F': 1}
            #y = torch.tensor([self.info_df.loc[person, 'Gender']], dtype=torch.float)
            y= gender_map[self.info_df.loc[int(person)]['Gender']]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pearson_corr= pearson_corr)
        data.hcp_id = torch.tensor([int(person)])
        data.index = torch.tensor([ind])

        return data
    
    def avergage_sliding_timepoints(self, time_series, num_nodes):
        sliding_window = 50
        stride=3
        dynamic_length = 1200
        num_sliding_windows = int((dynamic_length - sliding_window) / stride) + 1

        averaged_data = np.zeros((num_nodes, num_sliding_windows))

        for i in range(num_sliding_windows):
            start = i * stride
            end = start + sliding_window
            window_data = time_series[:, start:end]
            averaged_data[:, i] = np.mean(window_data, axis=1)
        
        return averaged_data

    def process(self):
        # Read data into huge `Data` list.
        data_list: List[Data] = []
        assert self.time_length == 1200 or self.time_length == 384
        filtered_people = pd.read_csv(HCP_DEMOGRAPHICS_PATH)["Subject"]
        #dataset = torch.load('/data/users2/bthapaliya/SpatialTemporalModel/abcd_hcp_schaefer.pt')
        dataset = torch.load('/data/users3/bthapaliya/SpatialTemporalModel/abcd_hcp_schaefer_100_final.pt')
        
        filtered_people = dataset['subjectids']

        for idx, person in enumerate(filtered_people):
            if self.connectivity_type == ConnType.STRUCT:
                if self.analysis_type != AnalysisType.ST_MULTIMODAL_AVG:
                    arr_struct = self._get_struct_arr(person)
                    G = create_thresholded_graph(arr_struct, threshold=self.threshold, num_nodes=self.num_nodes)
                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

            for ind, direction in enumerate(['1_LR']):
                index_of_person = dataset['subjectids'].index(person)
                ts = dataset['data'][index_of_person]
                # Because of normalisation part
                # Compute the mean and standard deviation of the data along the time axis
                mean = np.mean(ts, axis=0)
                std = np.std(ts, axis=0)

                # Z-score normalize the data across time points
                ts = (ts - mean) / std

                #ts = ts.T
                #ts = ts[:, self._idx_to_filter]
                assert ts.shape[0] == 1200
                assert ts.shape[1] == 100#400#68

                # # Crop timeseries
                # if self.time_length != 1200:#1185:
                #     ts = ts[:self.time_length, :]
                
                if self.time_length == 384:
                    ts = self.avergage_sliding_timepoints(ts.T, 400)
                    ts = ts.T

                if self.connectivity_type == ConnType.FMRI:
                    conn_measure = ConnectivityMeasure(
                        kind='correlation',
                        vectorize=False)
                    corr_arr = conn_measure.fit_transform([ts])
                    #assert corr_arr.shape == (1, 68, 68)
                    assert corr_arr.shape == (1, 100, 100)
                    corr_arr = corr_arr[0]
                    G = create_thresholded_graph(corr_arr, threshold=self.threshold, num_nodes=self.num_nodes)
                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()
                if self.include_edge_weights:
                    edge_attr = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                                             dtype=torch.float).unsqueeze(1)
                else:
                    edge_attr = None

                data = self.__create_data_object(person=person, ts=ts, ind=idx, edge_attr=edge_attr,
                                                 edge_index=edge_index, pearson_corr = corr_arr)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])





class ABCDDatasetAtlas(BrainDataset):
    def __init__(self, root, target_var: str, num_nodes: int, threshold: int, connectivity_type: ConnType,
                 normalisation: Normalisation, analysis_type: AnalysisType, edge_weights: bool, time_length: int = 1200,
                 encoding_strategy: EncodingStrategy = EncodingStrategy.NONE,
                 transform=None, pre_transform=None):

        if target_var not in ['gender']:
            print("ABCDDataset not prepared for that target_var!")
            exit(-2)
        if connectivity_type not in [ConnType.STRUCT, ConnType.FMRI]:
            print("ABCDDataset not prepared for that connectivity_type!")
            exit(-2)

        self._idx_to_filter = np.arange(0, 360)
        self.threshold = threshold
        self.ts_split_num: int = int(4800 / time_length)
        self.info_df = pd.read_csv(ABCD_DEMOGRAPHICS_PATH, sep="\t")

        super(ABCDDatasetAtlas, self).__init__(root, target_var=target_var, num_nodes=num_nodes, threshold=threshold,
                                         connectivity_type=connectivity_type, normalisation=normalisation,
                                         analysis_type=analysis_type, time_length=time_length,
                                         encoding_strategy=encoding_strategy, edge_weights=edge_weights,
                                         transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.threshold==100:
            return ['stmodel_data_abcd_brain_neuromark_latest_nothreshold_withpearsoncorr.dataset']
        else:
            return ["stmodel_data_abcd_brain_neuromark_latest_threshold_{0}_withpearsoncorr.dataset".format(self.threshold)]
        

    def __create_data_object(self, person: int, ts: np.ndarray, ind: int, edge_attr: torch.Tensor,
                             edge_index: torch.Tensor, pearson_corr: np.ndarray):
        assert ts.shape[0] > ts.shape[1]  # TS > N

        timeseries = normalise_timeseries(timeseries=ts, normalisation=self.normalisation)

        x = torch.tensor(timeseries, dtype=torch.float)

        labels = np.load("/data/users3/bthapaliya/BrainNetworkTransformer-main/abcd_inputs/genderlabels.npy") #Reader.get_sex_scores()

        y = labels[ind]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pearson_corr= pearson_corr)
        data.abcd_id = person#torch.tensor([int(person)])
        data.index = torch.tensor([ind])

        return data


    def process(self):
        # Read data into huge `Data` list.
        data_list: List[Data] = []
        assert self.time_length == 360 

        filtered_people = np.load("/data/users3/bthapaliya/BrainNetworkTransformer-main/abcd_inputs/subjectIds.npy") #Reader.get_ids()
        labels = np.load("/data/users3/bthapaliya/BrainNetworkTransformer-main/abcd_inputs/genderlabels.npy") #Reader.get_sex_scores()
        num_classes = 2
        num_subjects = len(filtered_people)
        y = np.zeros([num_subjects, 1])  # n x 1
        # Get class labels for all subjects
        for i in range(num_subjects):
            y[i] = int(labels[i])

        personIndex = 0
        path = '/data/users3/bthapaliya/BrainNetworkTransformer-main/abcd_inputs/timeseries_data_reduced.npy'
        abcd_matrix = np.load(path, allow_pickle=True)
        for idx, person in enumerate(filtered_people):
            for ind, direction in enumerate(['1_LR']):
                ts = abcd_matrix[personIndex]
                # Because of normalisation part
                ts = ts.T

                # Crop timeseries
                if self.time_length != 100*360:
                    ts = ts[:self.time_length, :]

                if self.connectivity_type == ConnType.FMRI:
                    conn_measure = ConnectivityMeasure(
                        kind='correlation',
                        vectorize=False)
                    corr_arr = conn_measure.fit_transform([ts])
                    #assert corr_arr.shape == (1, 68, 68)
                    assert corr_arr.shape == (1, 100, 100)
                    corr_arr = corr_arr[0]
                    G = create_thresholded_graph(corr_arr, threshold=self.threshold, num_nodes=self.num_nodes)
                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()
                if self.include_edge_weights:
                    edge_attr = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                                             dtype=torch.float).unsqueeze(1)
                else:
                    edge_attr = None

                data = self.__create_data_object(person=person, ts=ts, ind=idx, edge_attr=edge_attr,
                                                 edge_index=edge_index, pearson_corr= corr_arr)

                data_list.append(data)
            personIndex = personIndex + 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PersonNotFound(Exception):
    pass

