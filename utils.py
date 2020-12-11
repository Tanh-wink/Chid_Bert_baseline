import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os
from pyemd import emd
from numpy import float64
from torch.nn import CrossEntropyLoss


# def f1(y_hat, y_true, THRESHOLD=0.5):
#     '''
#     y_hat是未经过sigmoid函数激活的
#     输出的f1为Marco-F1
#     '''
#
#     epsilon = 1e-7
#     y_hat = y_hat > THRESHOLD
#     y_hat = np.int8(y_hat)
#     tp = np.sum(y_hat * y_true, axis=0)
#     fp = np.sum((1 - y_hat) * y_true, axis=0)
#     fn = np.sum(y_hat * (1 - y_true), axis=0)
#
#     p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
#     r = tp / (tp + fn + epsilon)
#
#     f1 = 2 * p * r / (p + r + epsilon)
#     f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
#
#     return np.mean(f1)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # loss_contrastive = torch.mean(label.float() * torch.pow(distance, 2)) / 2 + \
        #                    (1 - label.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean(label.float() * torch.pow(distance, 2) + (1.0 - label.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive / 2


def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    print(f'loaded {filePath}')
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):

    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)
    print(f'saved {filePath}')


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.best_threshold = 0.0
        self.delta = delta
        self.epoch = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_path)
        self.val_score = epoch_score


class ModelSaver:
    def __init__(self, mode="max"):
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_threshold = 0.0
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, threshold, model_path, step='', epoch=''):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.best_threshold = threshold
            self.save_checkpoint(epoch_score, model, model_path, step=step, epoch=epoch)

        elif score > self.best_score:
            self.best_score = score
            self.best_threshold = threshold
            self.save_checkpoint(epoch_score, model, model_path, step=step, epoch=epoch)

    def save_checkpoint(self, epoch_score, model, model_path, step='', epoch=''):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Epoch:{}, step:{} ,Validation score improved ({} --> {}). Saving model!'.format(epoch, step, self.val_score, epoch_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_path)
        self.val_score = epoch_score


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))