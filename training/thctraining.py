from utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from components import LRScheduler
import logging

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0


def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):

    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    y_0 = 1 - y_1

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def dominate_loss(A, soft_max=False):

    sz = A.shape[-1]

    m = torch.ones((sz, sz)).to(device=device)

    m.fill_diagonal_(-1)

    m = -m

    A = torch.matmul(A, m)

    if soft_max:
        max_ele = torch.logsumexp(A, dim=-1)
    else:
        max_ele, _ = torch.max(A, dim=-1, keepdim=False)

    max_ele = -max_ele
    max_ele = F.relu(max_ele)
    max_ele = torch.pow(max_ele, 2)
    return torch.sum(max_ele)


def topk_dominate_loss(A, k=3):

    all_ele = torch.sum(A, dim=-1)

    max_ele, _ = torch.topk(A, k, dim=-1, sorted=False)

    max_ele = torch.sum(max_ele, dim=-1)

    residual = F.relu(all_ele-2*max_ele)
    residual = torch.pow(residual, 2)

    return torch.sum(residual)


import torch
import numpy as np
import random
import scipy
# from torch_geometric.data import Data
# from torch_geometric.utils import dense_to_sparse


def mixup_data(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam


# def mixup_graph_data(data, alpha=1.0, device='cuda'):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     nodes = x = torch.reshape(data.x, (data.num_graphs, -1, data.x.shape[-1]))
#     batch_size = data.num_graphs
#     y = data.y
#     index = torch.randperm(batch_size).to(device)

#     mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     mixed_x_data = torch.reshape(mixed_x, (-1, data.x.shape[-1]))
#     mixed_nodes_data = torch.reshape(mixed_nodes, (-1, data.x.shape[-1]))
#     edge_index, edge_attr = dense_to_sparse(mixed_nodes)

#     new_data = Data(edge_index=edge_index, edge_attr=edge_attr,
#                     x=mixed_x_data, adj=mixed_nodes_data, y=y_a, num_graphs=data.num_graphs)
#     return new_data, y_a, y_b, lam


def mixup_renn_data(x, log_nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = []
    for i, j in enumerate(index):
        mixed_nodes.append(torch.matrix_exp(lam * log_nodes[i] + (1 - lam) * log_nodes[j]))

    mixed_nodes = torch.stack(mixed_nodes)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam



def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cal_step_connect(connectity, step):
    multi_step = connectity
    for _ in range(step):
        multi_step = np.dot(multi_step, connectity)
    multi_step[multi_step > 0] = 1
    return multi_step


def obtain_partition(dataloader, fc_threshold, step=2):
    pearsons = []
    for data_in, pearson, label in dataloader:
        pearsons.append(pearson)

    fc_data = torch.mean(torch.cat(pearsons), dim=0)

    fc_data[fc_data > fc_threshold] = 1
    fc_data[fc_data <= fc_threshold] = 0

    _, n = fc_data.shape

    final_partition = torch.zeros((n, (n-1)*n//2))

    connection = cal_step_connect(fc_data, step)
    temp = 0
    for i in range(connection.shape[0]):
        temp += i
        for j in range(i):
            if connection[i, j] > 0:
                final_partition[i, temp-i+j] = 1
                final_partition[j, temp-i+j] = 1
                # a = random.randint(0, n-1)
                # b = random.randint(0, n-1)
                # final_partition[a, temp-i+j] = 1
                # final_partition[b, temp-i+j] = 1

    connect_num = torch.sum(final_partition > 0)/n
    print(f'Final Partition {connect_num}')

    return final_partition.cuda().float(), connect_num

# class THCTrain:
#     def __init__(self, train_config: DictConfig,
#                  model: torch.nn.Module,
#                  optimizers: List[torch.optim.Optimizer],
#                  lr_schedulers: List[LRScheduler],
#                  dataloaders: List[utils.DataLoader],
#                  logger: logging.Logger) -> None:

#         device = "cuda"
#         self.model = model.to(device)
#         self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
#         self.epochs = train_config['epochs']
#         self.optimizers = optimizers
#         self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

#         self.group_loss = train_config['group_loss']
#         # self.group_loss_weight = train_config['group_loss_weight']

#         self.sparsity_loss = train_config['sparsity_loss']
#         self.sparsity_loss_weight = train_config['sparsity_loss_weight']

#         self.dominate_loss = train_config['dominate_loss'] if "dominate_loss" in train_config else None
  
#         self.dominate_loss_weight = train_config['dominate_loss_weight'] if "dominate_loss_weight" in train_config else None

#         # self.dominate_softmax = train_config['dominate_softmax']

#         self.topk = train_config['topk'] if "topk" in train_config else None

#         self.lr_schedulers = lr_schedulers

#         # self.save_path = Path(f"{train_config['log_folder']}/{}_{}")
#         # self.save_path = log_folder

#         self.save_learnable_graph = True

#         self.init_meters()

#     def init_meters(self):
#         self.train_loss, self.val_loss,\
#             self.test_loss, self.train_accuracy,\
#             self.val_accuracy, self.test_accuracy = [
#                 TotalMeter() for _ in range(6)]

#     def reset_meters(self):
#         for meter in [self.train_accuracy, self.val_accuracy,
#                       self.test_accuracy, self.train_loss,
#                       self.val_loss, self.test_loss]:
#             meter.reset()

#     def train_per_epoch(self, optimizer, lr_scheduler):
#         self.model.train()

#         # for time_series, node_feature, label in self.train_dataloader:
#         for data in self.train_dataloader:
#             time_series, node_feature, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
#             time_series= time_series.reshape(node_feature.shape[0], node_feature.shape[1], time_series.shape[-1])

#             # label = F.one_hot(label)

#             label = label.float()
#             self.current_step += 1

#             lr_scheduler.update(optimizer=optimizer, step=self.current_step)

#             time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            
#             # label = F.one_hot(label)

#             #if self.config.preprocess.continus:
#             # if self.preprocesscontinus:
#             #     time_series, node_feature, label = continus_mixup_data(
#             #         time_series, node_feature, y=label)

#             predict = self.model(time_series, node_feature)

#             loss = self.loss_fn(predict, label)

#             self.train_loss.update_with_weight(loss.item(), label.shape[0])
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # top1 = accuracy_score(torch.argmax(predict, dim=1).cpu().numpy(), torch.argmax(label, dim=1).cpu().numpy())
#             # self.train_accuracy.update_with_weight(top1, label.shape[0])
            
#             #Convert predictions and labels to numpy arrays
#             predictions = torch.argmax(predict, dim=1).cpu().numpy()
#             ground_truth = torch.argmax(label, dim=1).cpu().numpy()

#             # Calculate top-1 accuracy
#             top1_accuracy = accuracy_score(predictions, ground_truth)
#             # f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)

#             # # Calculate F1 score
#             # f1 = f1_score(ground_truth, predictions, average='weighted')  # Use 'weighted' for multiclass

#             # # Calculate precision and recall
#             # precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
#             # recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)

#             # self.train_accuracy.update_with_weight(f1, label.shape[0])
#             self.train_accuracy.update_with_weight(top1_accuracy, label.shape[0])
            
            
#             # wandb.log({"LR": lr_scheduler.lr,
#             #            "Iter loss": loss.item()})

#     def test_per_epoch(self, dataloader, loss_meter, acc_meter):
#         labels = []
#         result = []

#         self.model.eval()

#         # for time_series, node_feature, label in dataloader:
#         for data in dataloader:
#             time_series, node_feature, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
#             label = F.one_hot(label)
#             time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
#             time_series= time_series.reshape(node_feature.shape[0], node_feature.shape[1], time_series.shape[-1])
#             output = self.model(time_series, node_feature)

            
#             # label = F.one_hot(label)
#             label = label.float()

#             loss = self.loss_fn(output, label)

#             loss_meter.update_with_weight(
#                 loss.item(), label.shape[0])
            
#             # top1 = accuracy_score(torch.argmax(output, dim=1).cpu().numpy(), torch.argmax(label, dim=1).cpu().numpy())
#             # acc_meter.update_with_weight(top1, label.shape[0])

#             predictions = torch.argmax(output, dim=1).cpu().numpy()
#             ground_truth = torch.argmax(label, dim=1).cpu().numpy()

#             # Calculate top-1 accuracy
#             top1_accuracy = accuracy_score(predictions, ground_truth)

#             # Calculate F1 score
#             # f1 = f1_score(ground_truth, predictions, average='weighted')  # Use 'weighted' for multiclass

#             # f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0.0)  # Use 'weighted' for multiclass

#             # Calculate precision and recall
#             # precision = precision_score(ground_truth, predictions, average='weighted')
#             # recall = recall_score(ground_truth, predictions, average='weighted')

#             # acc_meter.update_with_weight(f1, label.shape[0])
#             acc_meter.update_with_weight(top1_accuracy, label.shape[0])


#             result += F.softmax(output, dim=1).tolist()
#             labels += torch.argmax(label, dim=1).tolist()
#             # labels += label.tolist()

#         result = np.argmax(result, axis=1)
#         auc = roc_auc_score(labels, result)
#         # result, labels = np.array(result), np.array(labels)
#         # result[result > 0.5] = 1
#         # result[result <= 0.5] = 0
#         result, labels = np.array(result), np.array(labels)
#         # result = np.argmax(result, axis=1)
#         # labels = np.argmax(labels, axis=1)

#         metric = precision_recall_fscore_support(
#             labels, result, average='micro', zero_division=0)

#         report = classification_report(
#             labels, result, output_dict=True, zero_division=0)

#         recall = [0, 0, 0]
#         # recall = [0, 0]
#         for k in report:
#             if isfloat(k):
#                 recall[int(float(k))] = report[k]['recall']
#         return [auc] + list(metric) + recall

#     def generate_save_learnable_matrix(self):

#         # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
#         learable_matrixs = []

#         labels = []

#         for time_series, node_feature, label in self.test_dataloader:
#             label = label.long()
#             time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
#             _, learable_matrix, _ = self.model(time_series, node_feature)

#             learable_matrixs.append(learable_matrix.cpu().detach().numpy())
#             labels += label.tolist()

#         self.save_path.mkdir(exist_ok=True, parents=True)
#         np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
#             learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

#     def save_result(self, results: torch.Tensor):
#         self.save_path.mkdir(exist_ok=True, parents=True)
#         np.save(self.save_path/"training_process.npy",
#                 results, allow_pickle=True)

#         torch.save(self.model.state_dict(), self.save_path/"model.pt")

#     def train(self):
#         training_process = []
#         self.current_step = 0
#         for epoch in range(self.epochs):
#             self.reset_meters()
#             self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
#             # self.train_per_epoch(self.optimizers, self.lr_schedulers)
#             val_result = self.test_per_epoch(self.val_dataloader,
#                                              self.val_loss, self.val_accuracy)

#             test_result = self.test_per_epoch(self.test_dataloader,
#                                               self.test_loss, self.test_accuracy)

#             self.logger.info(" | ".join([
#                 f'Epoch[{epoch}/{self.epochs}]',
#                 f'Train Loss:{self.train_loss.avg: .3f}',
#                 f'Train Acc:{self.train_accuracy.avg: .3f}%',
#                 f'Test Loss:{self.test_loss.avg: .3f}',
#                 f'Test Acc:{self.test_accuracy.avg: .3f}%',
#                 f'Val AUC:{val_result[0]:.4f}',
#                 f'Test AUC:{test_result[0]:.4f}',
#                 f'Test Sen:{test_result[-1]:.4f}',
#                 f'LR:{self.lr_schedulers[0].lr:.4f}'
#                 # f'LR:{self.lr_schedulers.lr:.4e}'
#             ]))

#             training_process.append({
#                 "Epoch": epoch,
#                 "Train Loss": self.train_loss.avg,
#                 "Train Acc": self.train_accuracy.avg,
#                 "Test Loss": self.test_loss.avg,
#                 "Test Acc": self.test_accuracy.avg,
#                 "Test AUC": test_result[0],
#                 'Test Sensitivity': test_result[-1],
#                 'Test Specificity': test_result[-2],
#                 'micro F1': test_result[-4],
#                 'micro recall': test_result[-5],
#                 'micro precision': test_result[-6],
#                 "Val AUC": val_result[0],
#                 "Val Loss": self.val_loss.avg,
#             })

#         if self.save_learnable_graph:
#             self.generate_save_learnable_matrix()
#         self.save_result(training_process)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class THCTrain:
    def __init__(self, train_config, model, optimizers, lr_schedulers, dataloaders, logger) -> None:
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']
        # self.group_loss_weight = train_config['group_loss_weight']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']

        self.dominate_loss = train_config['dominate_loss'] if "dominate_loss" in train_config else None
  
        self.dominate_loss_weight = train_config['dominate_loss_weight'] if "dominate_loss_weight" in train_config else None

        # self.dominate_softmax = train_config['dominate_softmax']

        self.topk = train_config['topk'] if "topk" in train_config else None

        self.lr_schedulers = lr_schedulers

        # self.save_path = Path(f"{train_config['log_folder']}/{}_{}")
        # self.save_path = log_folder
        self.logger = logger

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):
        self.model.train()

        # for data_in, pearson, label in self.train_dataloader:
        for data in self.train_dataloader:
            data_in, pearson, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
            label = label.long()

            data_in= data_in.reshape(pearson.shape[0], pearson.shape[1], data_in.shape[-1])

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)

            # output, learnable_matrix, edge_variance = self.model(inputs, nodes)
            output, learnable_matrix, edge_variance = self.model(inputs, nodes)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                    torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss
            if self.dominate_loss:
                dominate_graph_ls = self.dominate_loss_weight * \
                    topk_dominate_loss(learnable_matrix, k=self.topk)
                # print(dominate_graph_ls.item())
                loss += dominate_graph_ls

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            # self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        # for data_in, pearson, label in dataloader:
        for data in self.dataloader:
            data_in, pearson, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
            label = label.long()
            data_in= data_in.reshape(pearson.shape[0], pearson.shape[1], data_in.shape[-1])
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            # output, _, _ = self.model(data_in, pearson)
            output, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label in self.test_dataloader:
            label = label.long()
            data_in, nodes, label = data_in.to(
                device), nodes.to(device), label.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}'
            ]))

            wandb.log({
                "Epoch": epoch, 
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Val AUC": val_result[0],
                "Test AUC": test_result[0]
            })
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)


class BrainGNNTHCTrain(THCTrain):

    def __init__(self, train_config, model, optimizers, lr_schedulers, dataloaders, logger) -> None:
        super(BrainGNNTHCTrain, self).__init__(train_config, model, optimizers, lr_schedulers, dataloaders, logger)
        self.save_learnable_graph = False
        self.diff_loss = train_config.get('diff_loss', False)
        self.cluster_loss = train_config.get('cluster_loss', True)
        self.assignment_loss = train_config.get('assignment_loss', True)

    def train_per_epoch(self, optimizer):

        self.model.train()

        # for data_in, pearson, label in self.train_dataloader:
        #     label = label.long()

        for data in self.train_dataloader:
            data_in, pearson, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
            label = label.long()
            data_in= data_in.reshape(pearson.shape[0], pearson.shape[1], data_in.shape[-1])

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            _, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)


            output, assignments = self.model(nodes)
            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)
            if self.cluster_loss or self.assignment_loss:
                additional_loss = self.model.loss(assignments)
                if additional_loss is not None:
                    loss += additional_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        # for data_in, pearson, label in dataloader:
        #     label = label.long()
        for data in self.train_dataloader:
            data_in, pearson, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float).clone().detach(), data.y
            label = label.long()
            data_in= data_in.reshape(pearson.shape[0], pearson.shape[1], data_in.shape[-1])

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            output, assignments = self.model(pearson)
            # x = torch.reshape(x, (data.num_graphs, -1, x.shape[-1]))

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)
