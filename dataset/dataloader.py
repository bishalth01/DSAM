import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from dsamcomponents.utils import merge_y_and_others, calculate_indegree_histogram
from collections import Counter, defaultdict
from copy import deepcopy
from enum import Enum, unique
from typing import NoReturn, Dict, Any
import random
from torch.utils.data import  random_split

def init_dataloader_with_edges(cfg: DictConfig,
                               final_timeseries: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               dataset):
    if cfg.model.name=='SpatioTemporalModel' or cfg.model.name=='BrainGNN':
        dataset.data.y = torch.tensor(dataset.data.y, dtype=torch.int64)
    else:
        dataset.data.y = torch.tensor(dataset.data.y, dtype=torch.int64)
        # dataset.data.y = torch.tensor(dataset.data.y, dtype=torch.float32)
        # dataset.data.y = F.one_hot(torch.tensor(dataset.data.y, dtype=torch.int64))
        
    N_OUT_SPLITS:int = 5
    N_INNER_SPLITS:int = 4
    SPLIT_TO_TEST:int = 1

    skf_outer_generator = create_fold_generator(dataset, N_OUT_SPLITS)

    # Getting train / test folds
    outer_split_num: int = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1
        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != SPLIT_TO_TEST:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        break

    # Train / test sets defined, running the rest
    print('Size is:', len(X_train_out), '/', len(X_test_out))
    print('Positive classes:', sum([data.y.item() for data in X_train_out]),
              '/', sum([data.y.item() for data in X_test_out]))

    skf_inner_generator = create_fold_generator(X_train_out, N_INNER_SPLITS)


    
    #################
    # Main inner-loop
    #################
    inner_loop_run: int = 0
    for inner_train_index, inner_val_index in skf_inner_generator:
        inner_loop_run += 1

        X_train_in = X_train_out[torch.tensor(inner_train_index)]
        X_val_in = X_train_out[torch.tensor(inner_val_index)]

        # X_train_in.data.y = F.one_hot(torch.tensor(X_train_in.data.y, dtype=torch.int64))
        # X_val_in.data.y = F.one_hot(torch.tensor(X_val_in.data.y, dtype=torch.int64))

        X_train_in.data.x = torch.tensor(X_train_in.data.x, dtype=torch.float32)
        X_val_in.data.x = torch.tensor(X_val_in.data.x, dtype=torch.float32)

        print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
        print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
                  "/", sum([data.y.item() for data in X_val_in]))

        indegree_histogram = calculate_indegree_histogram(X_train_in)
        print(f'--> Indegree distribution: {indegree_histogram}')

        # train_in_loader = DataLoader(X_train_in, batch_size=cfg.dataset.batch_size , shuffle=True, drop_last=True)#, **kwargs_dataloader)
        # val_loader = DataLoader(X_val_in, batch_size=cfg.dataset.batch_size , shuffle=False, drop_last=True)#, **kwargs_dataloader)
        # # X_test_out.data.y = F.one_hot(torch.tensor(X_test_out.data.y, dtype=torch.int64))
        # X_test_out.data.x = torch.tensor(X_test_out.data.x, dtype=torch.float32)
        # test_out_loader = DataLoader(X_test_out, batch_size=cfg.dataset.batch_size, shuffle=False)

        # Modify your data loading
        train_in_loader = DataLoader(X_train_in, batch_size=cfg.dataset.batch_size, 
                                    shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(X_val_in, batch_size=cfg.dataset.batch_size, 
                                shuffle=False, drop_last=True, pin_memory=True)

        # Assuming X_test_out.data.x is already tensorized as shown in original code
        test_out_loader = DataLoader(X_test_out, batch_size=cfg.dataset.batch_size, 
                                    shuffle=False, pin_memory=True)
        
        
        length = len(dataset.data.y)
        train_length = len(X_train_in)

        with open_dict(cfg):
            # total_steps, steps_per_epoch for lr schedular
            cfg.steps_per_epoch = (
                train_length - 1) // cfg.dataset.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs


        break
    
    return [train_in_loader, val_loader, test_out_loader]


# def init_dataloader_with_edges(cfg: DictConfig,
#                                final_timeseries: torch.tensor,
#                                final_pearson: torch.tensor,
#                                labels: torch.tensor,
#                                dataset):
#     # Check if model name requires integer or float labels
#     if cfg.model.name == 'SpatioTemporalModel' or cfg.model.name == 'BrainGNN':
#         dataset.data.y = torch.tensor(dataset.data.y, dtype=torch.int64)
#     else:
#         dataset.data.y = torch.tensor(dataset.data.y, dtype=torch.float32)

#     # Define the size of each split (Train, Validation, Test)
#     dataset_size = len(dataset)
#     train_size = int(0.7 * dataset_size)  # 70% for training
#     val_size = int(0.1 * dataset_size)    # 10% for validation
#     test_size = dataset_size - train_size - val_size  # Remaining for testing

#     # Randomly split the dataset into Train, Val, and Test sets
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
#     )

#     # Convert x values to float32
#     train_dataset.dataset.data.x = torch.tensor(train_dataset.dataset.data.x, dtype=torch.float32)
#     val_dataset.dataset.data.x = torch.tensor(val_dataset.dataset.data.x, dtype=torch.float32)
#     test_dataset.dataset.data.x = torch.tensor(test_dataset.dataset.data.x, dtype=torch.float32)

#     # Print size of splits
#     print(f'Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}, Test Size: {len(test_dataset)}')

#     # DataLoader objects for each split using torch_geometric's DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=True, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=True, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, pin_memory=True)

#     # Print some stats
#     # print(f"Positive classes in Train: {sum([data.y.item() for data in train_dataset])}")
#     # print(f"Positive classes in Validation: {sum([data.y.item() for data in val_dataset])}")
#     # print(f"Positive classes in Test: {sum([data.y.item() for data in test_dataset])}")

#     # Configure steps_per_epoch and total_steps for learning rate scheduling
#     with open_dict(cfg):
#         cfg.steps_per_epoch = (len(train_dataset) - 1) // cfg.dataset.batch_size + 1
#         cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

#     return [train_loader, val_loader, test_loader]


    # length = final_timeseries.shape[0]
    # train_length = int(length * cfg.dataset.train_set * cfg.datasz.percentage)
    # val_length = int(length * cfg.dataset.val_set)

    # with open_dict(cfg):
    #     # total_steps, steps_per_epoch for lr schedular
    #     cfg.steps_per_epoch = (
    #         train_length - 1) // cfg.dataset.batch_size + 1
    #     cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # if cfg.datasz.percentage == 1.0:
    #     test_length = length - train_length - val_length
    # else:
    #     test_length = int(length * (1 - cfg.dataset.val_set - cfg.dataset.train_set))

    # train_index = range(0, train_length)
    # val_index = range(train_length, train_length + val_length)
    # test_index = range(train_length + val_length, length)

    # train_dataset = Subset(dataset, train_index)
    # val_dataset = Subset(dataset, val_index)
    # test_dataset = Subset(dataset, test_index)

    # train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    # val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=True)

    # return [train_loader, val_loader, test_loader]


def get_empty_metrics_dict(run_cfg: Dict[str, Any]) -> Dict[str, list]:
    if run_cfg['target_var'] == 'gender':
        tmp_dict = {'loss': [], 'sensitivity': [], 'specificity': [], 'acc': [], 'f1': [], 'auc': [],
                    'ent_loss': [], 'link_loss': [], 'best_epoch': []}
    else:
        tmp_dict = {'loss': [], 'r2': [], 'r': [], 'ent_loss': [], 'link_loss': []}
    return tmp_dict


def create_fold_generator(dataset,  num_splits: int):
    skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
    # merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
    #                                        torch.cat([data.index for data in dataset], dim=0))
    merged_labels = np.array(dataset.data.y.cpu().detach().numpy()).squeeze()  #For ABCD, .squeeze()
    skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.index.item() for data in dataset])
    
    return skf_generator


# From https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
class StratifiedGroupKFold:

    def __init__(self, n_splits=5, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
