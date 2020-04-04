import torch
import sklearn.metrics
import logging
import numpy as np


class Metric:
    r'''all data from torch would be converted into numpy.array. So All you need  is just konwing numpy opration
    '''

    def __init__(self, threshold=0.5):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        self.threshold = threshold
        self.FN_list = []
        self.FP_list = []
        self.losses = []

    def add_data(self, predict, target, indexes, loss):
        predict = (predict > self.threshold).numpy()
        target = target.numpy()
        assert predict.shape == target.shape
        if predict.shape == indexes.shape:
            self.FP_list += list(indexes[(predict == 1) * (target == 0)])
            self.FN_list += list(indexes[(predict == 0) * (target == 1)])
        self.TP += np.sum((predict == 1) * (target == 1))
        self.FP += np.sum((predict == 1) * (target == 0))
        self.TN += np.sum((predict == 0) * (target == 0))
        self.FN += np.sum((predict == 0) * (target == 1))
        self.losses.append(loss)

    def get_loss(self):
        return np.sum(self.losses) / len(self.losses)

    def get_accuracy(self):
        return self.TP + self.TN / (self.TP + self.TN + self.FP + self.FN + 1e-6)

    def get_sensitivity(self):
        return self.TP / (self.TP + self.FN + 1e-6)

    def get_specificity(self):
        return self.TN / (self.TN + self.FP + 1e-6)

    def get_precision(self):
        return self.TP / (self.TP + self.FP + 1e-6)

    def get_precision2(self):
        return (self.TN) / (self.TN + self.FN + 1e-6)

    def get_F1(self):
        SE = self.get_sensitivity()
        PC = self.get_precision()
        return 2 * SE * PC / (SE + PC + 1e-6)

    def tumor_ratio(self):
        return (self.FN + self.TP) / (self.TP + self.TN + self.FP + self.FN)