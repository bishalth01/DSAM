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


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        # self.weights_imbalance=torch.tensor([15.66, 2.69, 1.76]).cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        # self.loss_fn = torch.nn.CrossEntropyLoss()

        # self.loss_fn = F.nll_loss()
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        
        self.preprocesscontinus=True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        # for time_series, node_feature, label in self.train_dataloader:
        for data in self.train_dataloader:
            time_series, node_feature, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float), data.y
            time_series= time_series.reshape(node_feature.shape[0], node_feature.shape[1], time_series.shape[-1])

            # label = F.one_hot(label)

            label = label.long().squeeze()
            self.current_step += 1

            # lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            predict = self.model(time_series, node_feature)

            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Convert predictions and labels to numpy arrays
            predictions = torch.argmax(predict, dim=1).cpu().numpy()

            # Calculate top-1 accuracy
            top1_accuracy = accuracy_score(predictions, label.cpu().numpy())

            # self.train_accuracy.update_with_weight(f1, label.shape[0])
            self.train_accuracy.update_with_weight(top1_accuracy, label.shape[0])
 

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        # for time_series, node_feature, label in dataloader:
        for data in dataloader:
            time_series, node_feature, label = data.x, torch.tensor(np.array(data.pearson_corr), dtype=torch.float), data.y
            # label = F.one_hot(label)
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            time_series= time_series.reshape(node_feature.shape[0], node_feature.shape[1], time_series.shape[-1])
            output = self.model(time_series, node_feature)

            
            # label = F.one_hot(label)
            label = label.long().squeeze()

            loss = self.loss_fn(output, label)

            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            
            # top1 = accuracy_score(torch.argmax(output, dim=1).cpu().numpy(), torch.argmax(label, dim=1).cpu().numpy())
            # acc_meter.update_with_weight(top1, label.shape[0])

            predictions = torch.argmax(output, dim=1).cpu().numpy()
            # ground_truth = torch.argmax(label, dim=1).cpu().numpy()

            # Calculate top-1 accuracy
            top1_accuracy = accuracy_score(predictions, label.cpu().numpy())

            # Calculate F1 score
            # f1 = f1_score(ground_truth, predictions, average='weighted')  # Use 'weighted' for multiclass

            # f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0.0)  # Use 'weighted' for multiclass

            # Calculate precision and recall
            # precision = precision_score(ground_truth, predictions, average='weighted')
            # recall = recall_score(ground_truth, predictions, average='weighted')

            # acc_meter.update_with_weight(f1, label.shape[0])
            acc_meter.update_with_weight(top1_accuracy, label.shape[0])


            result += F.softmax(output, dim=1).tolist()
            # labels += torch.argmax(label, dim=1).tolist()
            labels += label.tolist()

        result = np.argmax(result, axis=1)
        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # result, labels = np.array(result), np.array(labels)
        # result = np.argmax(result, axis=1)
        # labels = np.argmax(labels, axis=1)

        metric = precision_recall_fscore_support(
            labels, result, average='micro', zero_division=0)

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        # recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

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
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            # self.train_per_epoch(self.optimizers, self.lr_schedulers)
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Acc:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Acc:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
                # f'LR:{self.lr_schedulers.lr:.4e}'
            ]))

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Acc": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Acc": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)