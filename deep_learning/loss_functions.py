from __future__ import division
import numpy as np
from utils import accuracy_score

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def loss(self, y, y_pred):
        return 0.5 * ((y - y_pred) ** 2)
    
    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1-1e-15)
        return - y * np.log(p) - (1-y) * np.log(1-p)
    
    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
    
    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1-1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class MeanSquaredError(Loss):
    def loss(self, y, y_pred):
        return np.sum(np.power(y_pred - y, 2)) / y_pred.shape[0]
    
    def gradient(self, y, y_pred):
        return 2 * np.sum(y_pred - y) / y_pred.shape[0]
