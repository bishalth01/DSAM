from dsamcomponents.utils import EarlyStopping
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
from torch.cuda.amp import autocast, GradScaler
import copy
import os
import torch.nn as nn

class BrainGNNDSAMTrain:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:
        
        EPS = 1e-10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        self.identity_matrix = torch.eye(self.config.model.num_nodes, device=device)
        self.save_model=True

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
    
    def topk_loss(self, s, ratio):
        EPS = 1e-10
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res
    
    def braingnn_loss(self, opt, output, allpools, scores, loss_c):
        scores_list, s, pool_weights, loss_pools, loss_tpks = [], [], [], [], []

        for i in range(len(scores)):
            scores_list.append(torch.sigmoid(scores[i]).view(output.size(0), -1).view(-1).detach().cpu().numpy())
            s.append(torch.sigmoid(scores[i]).view(output.size(0), -1))

            module = allpools[i]
            module_params = [param for name, param in module.named_parameters() if param.requires_grad]
            pool_weights.extend(module_params)
            
            loss_pools.append((torch.norm(module_params[0], p=2) - 1) ** 2)  
            loss_tpks.append(self.topk_loss(s[i], opt.model.bgnnratio ))

        loss = opt.model.lamb0 * loss_c + opt.model.lamb1 * loss_pools[0] + opt.model.lamb2 * loss_pools[1] \
                + opt.model.lamb3 * loss_tpks[0] + opt.model.lamb4 * loss_tpks[1]

        return loss
    

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        # scaler = GradScaler()  # Mixed precision

        for data in self.train_dataloader:
            batch_size = len(data.y)
            pseudo = self.identity_matrix.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.identity_matrix.shape[-1])  # Get directly from GPU
            data.pos = pseudo

            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series = data.x.cuda(non_blocking=True)
            node_feature = data.pearson_corr  # if not used on GPU, keep it on CPU
            # node_feature = torch.tensor(data.pearson_corr, dtype=torch.float16).cuda(non_blocking=True)  # if not used on GPU, keep it on CPU
            label = data.y.cuda(non_blocking=True)
            edge_index = data.edge_index.cuda(non_blocking=True)
            edge_attr = data.edge_attr.cuda(non_blocking=True)
            pseudo_torch = data.pos
            batch = data.batch.cuda(non_blocking=True)

            optimizer.zero_grad()
            # with autocast():  # Enable mixed precision
            predict, allpools,scores, sfnc_matrix, attention_weights = self.model(data, time_series, edge_index, edge_attr, node_feature, pseudo_torch, batch)
            # predict = torch.nan_to_num(predict)
            label = label.long()
            loss_c = self.loss_fn(predict, label.squeeze())
            # reg_loss = F.mse_loss(attention_weights, sfnc_matrix)
            # loss = loss_c #+ 0.0 * reg_loss
            loss = self.braingnn_loss(self.config, predict, allpools, scores, loss_c)

            # scaler.scale(loss).backward()  # Use scaler for backprop
            # scaler.step(optimizer)  # Update weights
            # scaler.update()  # Update scaler for next iteration

            loss.backward()  # Use scaler for backprop
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            optimizer.step()  # Update weights

            

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(predict, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        self.model.eval()
        metrics_dict = {'loss': [], 'f1': [], 'auc': 0, 'specificity': [], 'sensitivity': [], 'ACC': []}

        for data in dataloader:
            batch_size = len(data.y)
            pseudo = self.identity_matrix.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.identity_matrix.shape[-1])
            data.pos = pseudo

            time_series = data.x.cuda(non_blocking=True)
            node_feature = data.pearson_corr  # if not used on GPU, keep it on CPU
            # node_feature = torch.tensor(data.pearson_corr, dtype=torch.float16).cuda(non_blocking=True) 
            label = data.y.cuda(non_blocking=True)
            edge_index = data.edge_index.cuda(non_blocking=True)
            edge_attr = data.edge_attr.cuda(non_blocking=True)
            pseudo_torch = data.pos
            batch = data.batch.cuda(non_blocking=True)

            with torch.no_grad():
                # with autocast():
                output, _, _, sfnc_matrix, attention_weights = self.model(data, time_series, edge_index, edge_attr, node_feature, pseudo_torch, batch)
                output = torch.nan_to_num(output)
                label = label.long()
                loss = self.loss_fn(output, label.squeeze())

            loss_meter.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(labels, result, average='micro')

        report = classification_report(labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        
        metrics_dict['auc'] = auc
        metrics_dict['loss'] = loss_meter.avg
        metrics_dict['f1'] = metric[2]
        metrics_dict['specificity'] = recall[0]
        metrics_dict['sensitivity'] = recall[1]
        metrics_dict['acc'] = acc_meter.avg


        return [auc] + list(metric) + recall, metrics_dict


    def train(self):
        training_process = []
        self.current_step = 0
        self.model = self.model.to(self.device)

        model_name = str(self.config.dataset.name) + str(self.config.model.dynamic) + str(self.config.model.threshold)
        early_stopping = EarlyStopping(patience=40, model_saving_name=model_name)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        self.logger.info(self.model)
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result, val_metrics = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)
            early_stopping(val_metrics, self.model)
            if early_stopping.early_stop:
                print("EARLY STOPPING IT")
                break
            
            
            if val_metrics["loss"] < best_loss and epoch > 5:
                print("saving best model")
                best_loss = val_metrics["loss"]
                best_model_wts = copy.deepcopy(self.model.state_dict())
                if self.save_model:
                    torch.save(best_model_wts, os.path.join('/data/users3/bthapaliya/BrainNetworkTransformer-main/saved_models',model_name))


            # test_result = self.test_per_epoch(self.test_dataloader,
            #                                   self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}%',
                f'Val Loss:{self.val_loss.avg: .3f}',
                f'Val AUC:{val_result[0]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            wandb.log({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Val Accuracy": self.val_accuracy.avg,
                "Val Loss": self.val_loss.avg,
                "Val AUC": val_result[0],
            })

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Val Accuracy": self.val_accuracy.avg,
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })



        #Loading the best model from validation set
        # self.model.load_state_dict(os.path.join('/data/users3/bthapaliya/BrainNetworkTransformer-main/saved_models',model_name)) 
        self.model.load_state_dict(best_model_wts)
        self.reset_meters()
        train_result, train_metrics = self.test_per_epoch(self.train_dataloader, self.train_loss, self.train_accuracy)
        val_result, val_metrics = self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
        test_result, test_metrics = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)

        #Saving best model now

        self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
        ]))


        wandb.log({
                "Train Loss": self.train_loss.avg,
                "kaiming_fnc": "Final",
                "Final Train Accuracy": self.train_accuracy.avg,
                "Final Test Loss": self.test_loss.avg,
                "Final Test Accuracy": self.test_accuracy.avg,
                "Final Val Acuracy": self.val_accuracy.avg,
                "Final Val AUC": val_result[0],
                "Final Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
        })

        training_process.append({
                "Epoch": epoch,
                "Final Train Loss": self.train_loss.avg,
                "Final Train Accuracy": self.train_accuracy.avg,
                "Final Test Loss": self.test_loss.avg,
                "Final Test Accuracy": self.test_accuracy.avg,
                "Final Test AUC": test_result[0],
                'Final Test Sensitivity': test_result[-1],
                'Final Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Final Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

