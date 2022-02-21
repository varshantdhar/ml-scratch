import numpy as np

# collection of common activation functions

class Sigmoid():
    def __call__(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(x))
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Tanh():
    def __call__(self, x):
        x = np.clip(x, -25, 25)
        return np.exp(x) - np.exp(-x) / (np.exp(x) + np.exp(-x))
    
    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    def __call__(self, x):
        return np.where(x <= 0, 0, x)

    def gradient(self, x):
        return np.where(x <= 0, 0, 1)

class Softplus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))
    
    def gradient(self, x):
        return 1 / (1 + np.exp(-x))

class Softmax():
    def __call__(self, x):
        x = np.clip(x, -50, 50)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class LeakyReLU():
    def __call__(self, x, alpha = 0.01):
        return np.where(x < 0, alpha * x, x)
    
    def gradient(self, x, alpha=0.01):
        return np.where(x < 0, alpha, 1)