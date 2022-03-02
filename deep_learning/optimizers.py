import numpy as np

class StochasticGradientDescent():
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.wd = learning_rate / 4
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        self.w_updt = self.w_updt * self.momentum + (1 - self.momentum) * grad_wrt_w
        return w - self.learning_rate * self.w_updt - self.w_updt * self.wd
        
class StochasticGradientAscent():
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.wd = learning_rate / 4
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        self.w_updt = self.w_updt * self.momentum + (1 - self.momentum) * grad_wrt_w
        return w + self.learning_rate * self.w_updt + self.w_updt * self.wd

class Adam():
    def __init__(self, learning_rate=1e-5, b1 = 0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = 1e-8
        self.m = None
        self.v = None
    
    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1 * self.m + (1-self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1-self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        return w - self.learning_rate * m_hat/ (np.sqrt(v_hat) + self.eps)

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)

class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)