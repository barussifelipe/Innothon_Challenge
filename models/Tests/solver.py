import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.DataLoader import get_loader_segment


def my_kl_loss(p, q, device='cpu'):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1) # This returns a tensor of shape (batch_size,)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        # Determine device first, as it's needed for model and data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Data loaders
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        self.build_model()
        self.criterion = nn.MSELoss() # Default reduction='mean', so rec_loss will be a scalar.

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3, device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                sum_prior_dim = torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                
                # Each my_kl_loss returns a (batch_size,) tensor
                current_series_loss = (my_kl_loss(series[u], (prior[u] / sum_prior_dim).detach(), device=self.device) +
                                       my_kl_loss((prior[u] / sum_prior_dim).detach(), series[u], device=self.device))
                current_prior_loss = (my_kl_loss((prior[u] / sum_prior_dim), series[u].detach(), device=self.device) +
                                      my_kl_loss(series[u].detach(), (prior[u] / sum_prior_dim), device=self.device))
                
                if u == 0:
                    series_loss = current_series_loss
                    prior_loss = current_prior_loss
                else:
                    series_loss += current_series_loss
                    prior_loss += current_prior_loss
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            # --- MODIFIED: Take mean over batch dimension to make it a scalar ---
            series_loss_scalar = torch.mean(series_loss)
            prior_loss_scalar = torch.mean(prior_loss)

            rec_loss = self.criterion(output, input) # This is already a scalar

            loss_1.append((rec_loss - self.k * series_loss_scalar).item())
            loss_2.append((rec_loss + self.k * prior_loss_scalar).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    sum_prior_dim = torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                    
                    # Each my_kl_loss returns a (batch_size,) tensor
                    current_series_loss = (my_kl_loss(series[u], (prior[u] / sum_prior_dim).detach(), device=self.device) +
                                           my_kl_loss((prior[u] / sum_prior_dim).detach(), series[u], device=self.device))
                    current_prior_loss = (my_kl_loss((prior[u] / sum_prior_dim), series[u].detach(), device=self.device) +
                                          my_kl_loss(series[u].detach(), (prior[u] / sum_prior_dim), device=self.device))
                    
                    if u == 0:
                        series_loss = current_series_loss
                        prior_loss = current_prior_loss
                    else:
                        series_loss += current_series_loss
                        prior_loss += current_prior_loss

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                # --- MODIFIED: Take mean over batch dimension to make it a scalar ---
                series_loss_scalar = torch.mean(series_loss)
                prior_loss_scalar = torch.mean(prior_loss)

                rec_loss = self.criterion(output, input) # This is already a scalar

                loss1_list.append((rec_loss - self.k * series_loss_scalar).item()) # Use scalar version
                loss1 = rec_loss - self.k * series_loss_scalar # Use scalar version
                loss2 = rec_loss + self.k * prior_loss_scalar # Use scalar version

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=self.device))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1) # This 'loss' is (batch_size,)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                sum_prior_dim = torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                
                s_loss = my_kl_loss(series[u], (prior[u] / sum_prior_dim).detach(), device=self.device) * temperature
                p_loss = my_kl_loss((prior[u] / sum_prior_dim), series[u].detach(), device=self.device) * temperature
                
                if u == 0:
                    series_loss = s_loss
                    prior_loss = p_loss
                else:
                    series_loss += s_loss
                    prior_loss += p_loss

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = (metric * loss).detach().cpu().numpy() # This 'cri' is (batch_size,) - correct for numpy concat
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1) # This 'loss' is (batch_size,)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                sum_prior_dim = torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)

                s_loss = my_kl_loss(series[u], (prior[u] / sum_prior_dim).detach(), device=self.device) * temperature
                p_loss = my_kl_loss((prior[u] / sum_prior_dim), series[u].detach(), device=self.device) * temperature

                if u == 0:
                    series_loss = s_loss
                    prior_loss = p_loss
                else:
                    series_loss += s_loss
                    prior_loss += p_loss
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = (metric * loss).detach().cpu().numpy() # This 'cri' is (batch_size,) - correct for numpy concat
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        threshold_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) Evaluation on the TEST set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader): # Using self.test_loader now
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1) # This 'loss' is (batch_size,)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                sum_prior_dim = torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)

                s_loss = my_kl_loss(series[u], (prior[u] / sum_prior_dim).detach(), device=self.device) * temperature
                p_loss = my_kl_loss((prior[u] / sum_prior_dim), series[u].detach(), device=self.device) * temperature
                
                if u == 0:
                    series_loss = s_loss
                    prior_loss = p_loss
                else:
                    series_loss += s_loss
                    prior_loss += p_loss
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = (metric * loss).detach().cpu().numpy() # This 'cri' is (batch_size,) - correct for numpy concat
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        final_test_energy = np.array(attens_energy)
        final_test_labels = np.array(test_labels)

        pred = (final_test_energy > thresh).astype(int)
        gt = final_test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment
        pred_adjusted = np.copy(pred)
        gt_original = np.copy(gt)

        anomaly_state = False
        for i in range(len(gt_original)):
            if gt_original[i] == 1 and pred_adjusted[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt_original[j] == 0:
                        break
                    else:
                        if pred_adjusted[j] == 0:
                            pred_adjusted[j] = 1
                for j in range(i, len(gt_original)):
                    if gt_original[j] == 0:
                        break
                    else:
                        if pred_adjusted[j] == 0:
                            pred_adjusted[j] = 1
            elif gt_original[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred_adjusted[i] = 1

        pred = pred_adjusted
        gt = gt_original

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score