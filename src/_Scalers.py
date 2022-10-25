""" Collection of scaler classes to be used for experiments """
from abc import ABCMeta, abstractclassmethod
import numpy as np
import sys
sys.path.append(sys.path[0]+'/..')


class Scaler():
    __metaclass__ = ABCMeta
    @abstractclassmethod
    def transform():
        pass
    @abstractclassmethod
    def name():
        pass


class Identity(Scaler):
    def transform(self, x):
        self.scaling_factors = np.array([1 for _ in range(x.shape[1])])
        return x
    def name(self):
        return "Identity"

class Normalizer(Scaler):
    def transform(self, x):
        stds = np.std(x, axis=0, keepdims=True)
        self.scaling_factors = stds
        return (x - np.mean(x, axis=0)) / stds
    def name(self):
        return "Normalizer"

class Inverter(Scaler):
    def transform(self, x):
        _vars = np.var(x, axis=0, keepdims=True)
        self.scaling_factors = _vars
        return (x - np.mean(x, axis=0)) / _vars
    def name(self):
        return "Inverter"
