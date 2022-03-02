from __future__ import print_function, division
import math
import numpy as np
import copy
from deep_learning.activation import Sigmoid, ReLU, Softmax, Softplus, LeakyReLU, Tanh
from utils import merge, pad_along_axis

class Layer(object):

    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
       self.layer_input = None
       self.input_shape = input_shape
       self.n_units = n_units
       self.trainable = True
       self.initialized = False
       self.W = None
       self.b = None
    
    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[1])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[1], self.n_units))
        self.b = np.random.uniform(-limit, limit, (1, self.n_units))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)
        self.initialized = True
    
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)
    
    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.b

    def backward_pass(self, accum_grad):
        W = self.W

        if self.trainable:
            grad_w = np.tensordot(self.layer_input, accum_grad, axes=([0,1],[0,1]))
            grad_b = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softplus': Softplus,
    'leakyrelu': LeakyReLU,
    'softmax': Softmax
}


class Activation(Layer):
    """A layer that applies an activation operation to the input.
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True
        self.initialized = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, traing=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        activation_grad = self.activation_func.gradient(self.layer_input)
        return accum_grad * np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)

    def output_shape(self):
        return self.input_shape

class Reshape(Layer):
    def __init__(self, shape, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.shape = shape
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], ) + self.shape)

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape

class RNN(Layer):
    def __init__(self, n_units, activation = 'tanh', bptt_trunc=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.initialized = False
        self.activation = activation_functions[activation]()
        self.bptt_trunc = bptt_trunc
        self.trainable = True
        self.W = None # hidden to hidden
        self.U = None # input to hidden
        self.V = None # hidden to output
    
    def initialize(self, optimizer):
        _, input_dim = self.input_shape
        limit = 1 / math.sqrt(self.n_units)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        self.V = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))

        self.U_opt  = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)
        self.initialized = True

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, _ = X.shape

        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, self.n_units))

        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            self.state_input[:, t] = self.states[:, t-1].dot(self.W.T) +  X[:, t].dot(self.U.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)
        
        return self.outputs
    
    def backward_pass(self, accum_grad):

        _, timesteps, _ = accum_grad.shape

        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)

        accum_grad_next = np.zeros_like(self.layer_input)

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            activation_grad = self.activation.gradient(self.state_input[:, t])
            activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
            grad_wrt_state = accum_grad[:, t].dot(self.V) * activation_grad
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)

            # bptt
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:,t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                activation_grad = self.activation.gradient(self.state_input[:, t_ - 1])
                activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
                grad_wrt_state = grad_wrt_state.dot(self.W.T) * activation_grad
            
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self):
        return self.output_shape

class BiRNN(Layer):
    def __init__(self, n_units, activation = 'tanh', merge_mode = 'ave',bptt_trunc=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.bptt_trunc = bptt_trunc
        self.initialized = False
        self.trainable = True
        self.merge_mode = merge_mode
        self.W = None # hidden to hidden
        self.U = None # input to hidden
        self.V = None # hidden to output
        self.W_reverse = None # hidden to hidden
        self.U_reverse = None # input to hidden
        self.V_reverse = None # hidden to output

    def initialize(self, optimizer):
        _, input_dim = self.input_shape
        limit = 1 / math.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        self.U_reverse = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.V_reverse = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W_reverse = np.random.uniform(-limit, limit, (self.n_units, self.n_units))

        self.U_opt  = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)
        self.U_reverse_opt = copy.copy(optimizer)
        self.V_reverse_opt = copy.copy(optimizer)
        self.W_reverse_opt = copy.copy(optimizer)
        self.initialized = True

    def parameters(self):
        forward = np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)
        reverse = np.prod(self.W_reverse.shape) + np.prod(self.U_reverse.shape) + np.prod(self.V_reverse.shape)
        return forward + reverse
    
    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, _ = X.shape
        D = 2

        self.state_input = np.zeros((batch_size, timesteps, self.n_units, D))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units, D))
        self.outputs = np.zeros((batch_size, timesteps, self.n_units, D))

        self.states[:, -1] = np.zeros((batch_size, self.n_units, D))
        for t in range(timesteps):
            self.state_input[:, t, :, 0] = self.states[:, t-1, :, 0].dot(self.W.T) +  X[:, t].dot(self.U.T)
            self.states[:, t, :, 0] = self.activation(self.state_input[:, t, :, 0])
            self.outputs[:, t, :, 0] = self.states[:, t, :, 0].dot(self.V.T)
        
        for t in reversed(range(timesteps)):
            self.state_input[:, t, :, 1] = self.states[:, t-1, :, 1].dot(self.W_reverse.T) + X[:, t].dot(self.U_reverse.T)
            self.states[:, t, :, 1] = self.activation(self.state_input[:, t, :, 1])
            self.outputs[:, t, :, 1] = self.states[:, t, :, 1].dot(self.V_reverse.T)

        if self.merge_mode == 'None':
            return self.outputs
        else:
            return merge(self.outputs, self.merge_mode)
    
    def backward_pass(self, accum_grad):

        _, timesteps, _ = accum_grad.shape
        accum_grad = np.repeat(accum_grad[:, :, :, np.newaxis], 2, axis=3)

        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)

        grad_U_reverse = np.zeros_like(self.U_reverse)
        grad_V_reverse = np.zeros_like(self.V_reverse)
        grad_W_reverse = np.zeros_like(self.W_reverse)

        accum_grad_next = np.zeros_like(self.layer_input)
        accum_grad_next = np.repeat(accum_grad_next[:, :, :, np.newaxis], 2, axis=3)

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t, :, 0].T.dot(self.states[:, t, :, 0])
            activation_grad = self.activation.gradient(self.state_input[:, t, :, 0])
            activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
            grad_wrt_state = accum_grad[:, t, :, 0].dot(self.V) * activation_grad
            accum_grad_next[:, t, :, 0] = grad_wrt_state.dot(self.U)

            # bptt
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1, :, 0])
                activation_grad = self.activation.gradient(self.state_input[:, t_ - 1, :, 0])
                activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
                grad_wrt_state = grad_wrt_state.dot(self.W.T) * activation_grad
        
        for t in range(timesteps):
            grad_V_reverse += accum_grad[:, t, :, 1].T.dot(self.states[:, t, :, 1])
            activation_grad = self.activation.gradient(self.state_input[:, t, :, 1])
            activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
            grad_wrt_state = accum_grad[:, t, :, 1].dot(self.V_reverse) * activation_grad
            accum_grad_next[:, t, :, 1] = grad_wrt_state.dot(self.U_reverse)

            # bptt
            for t_ in np.arange(max(0, t - self.bptt_trunc), t+1):
                grad_U_reverse += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W_reverse += grad_wrt_state.T.dot(self.states[:, t_ - 1, :, 1])
                activation_grad = self.activation.gradient(self.state_input[:, t_ - 1, :, 1])
                activation_grad = np.clip(activation_grad / (np.linalg.norm(activation_grad) + 1e-5), -1.0, 1.0)
                grad_wrt_state = grad_wrt_state.dot(self.W_reverse.T) * activation_grad
            
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)
        self.U_reverse = self.U_reverse_opt.update(self.U_reverse, grad_U_reverse)
        self.V_reverse = self.V_reverse_opt.update(self.V_reverse, grad_V_reverse)
        self.W_reverse = self.W_reverse_opt.update(self.W_reverse, grad_W_reverse)

        if self.merge_mode == 'None':
            return accum_grad_next
        else:
            return merge(accum_grad_next, self.merge_mode)

    def output_shape(self):
        return self.output_shape



class LSTM(Layer):
    def __init__(self, n_units, activation_1 = 'sigmoid', activation_2 = 'tanh', 
    bptt_trunc=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.activation_1 = activation_functions[activation_1]()
        self.activation_2 = activation_functions[activation_2]()
        self.bptt_trunc = bptt_trunc
        self.initialized = False
        self.W_i = None
        self.U_i = None
        self.W_f = None
        self.U_f = None
        self.W_g = None
        self.U_g = None
        self.W_o = None
        self.U_o = None

    def initialize(self, optimizer):
        _, input_dim = self.input_shape
        limit = 1 / math.sqrt(input_dim)
        self.U_i  = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        self.U_f = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        self.U_g = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        self.U_o = np.random.uniform(-limit, limit, (self.n_units, input_dim))

        limit = 1 / math.sqrt(self.n_units)
        self.W_i = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W_f = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W_g = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.W_o = np.random.uniform(-limit, limit, (self.n_units, self.n_units))

        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))

        self.U_i_opt = copy.copy(optimizer)
        self.W_i_opt = copy.copy(optimizer)
        self.U_f_opt = copy.copy(optimizer)
        self.W_f_opt = copy.copy(optimizer)
        self.U_g_opt = copy.copy(optimizer)
        self.W_g_opt = copy.copy(optimizer)
        self.U_o_opt = copy.copy(optimizer)
        self.W_o_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.initialized = True

    def parameters(self):
        U_shape = np.prod(self.U_i.shape) + np.prod(self.U_f.shape) + np.prod(self.U_g.shape) + np.prod(self.U_o.shape)
        W_shape = np.prod(self.W_i.shape) + np.prod(self.W_f.shape) + np.prod(self.W_g.shape) + np.prod(self.W_o.shape)
        return U_shape + W_shape + np.prod(self.V.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        self.c = np.zeros((batch_size, timesteps, self.n_units))
        self.input_gate = np.zeros((batch_size, timesteps, self.n_units))
        self.forget_gate = np.zeros((batch_size, timesteps, self.n_units))
        self.cell_gate = np.zeros((batch_size, timesteps, self.n_units))
        self.output_gate = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.c = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        self.c[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.input_gate[:, t] = self.activation_1(self.layer_input[:, t].dot(self.U_i.T) + self.states[:, t-1].dot(self.W_i.T))
            self.forget_gate[:, t] = self.activation_1(self.layer_input[:, t].dot(self.U_f.T) + self.states[:, t-1].dot(self.W_f.T))
            self.cell_gate[:, t] = self.activation_2(self.layer_input[:, t].dot(self.U_g.T) + self.states[:, t-1].dot(self.W_g.T))
            self.output_gate[:, t] = self.activation_1(self.layer_input[:, t].dot(self.U_o.T) + self.states[:, t-1].dot(self.W_o.T))
            self.c[:, t] = self.forget_gate[:, t].multiply(self.c[:, t-1]) + self.input_gate[:, t].multiply(self.cell_gate[:, t])
            self.states[:, t] = self.output_gate[:, t].multiply(self.activation_2(self.c[:, t]))
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_pass(self, accum_grad):
        _, timesteps, _ = accum_grad.shape
        
        grad_W_i = np.zeros_like(self.W_i)
        grad_U_i = np.zeros_like(self.U_i)
        grad_W_f = np.zeros_like(self.W_f)
        grad_U_f = np.zeros_like(self.U_f)
        grad_W_g = np.zeros_like(self.W_g)
        grad_U_g = np.zeros_like(self.U_g)
        grad_W_o = np.zeros_like(self.W_o)
        grad_U_o = np.zeros_like(self.U_o)
        grad_V = np.zeros_like(self.V)

        accum_grad_next = np.zeros_like(self.layer_input)

        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])

            grad_wrt_state = accum_grad[:, t].dot(self.V)

            grad_wrt_output_gate = grad_wrt_state * self.activation_2(self.c[:, t])
            grad_wrt_cell_t = grad_wrt_state * self.output_gate[:, t] * self.activation_2.gradient(self.c[:, t])

            grad_wrt_input_gate = grad_wrt_cell_t * self.cell_gate[:, t]
            grad_wrt_cell_gate = grad_wrt_cell_t * self.input_gate[:, t]
            grad_wrt_forget_gate = grad_wrt_cell_t * self.cell_gate[:, t-1]

            z_o = self.layer_input[:, t].dot(self.U_o.T) + self.states[:, t-1].dot(self.W_o.T)
            grad_o = grad_wrt_output_gate * self.activation_1.gradient(z_o)

            z_f = self.layer_input[:, t].dot(self.U_f.T) + self.states[:, t-1].dot(self.W_f.T)
            grad_f = grad_wrt_forget_gate * self.activation_1.gradient(z_f)

            z_i = self.layer_input[:, t].dot(self.U_i.T) + self.states[:, t-1].dot(self.W_i.T)
            grad_i = grad_wrt_input_gate * self.activation_1.gradient(z_i)

            z_g = self.layer_input[:, t].dot(self.U_g.T) + self.states[:, t-1].dot(self.W_g.T)
            grad_g = grad_wrt_cell_gate * self.activation_2.gradient(z_g)

            accum_grad_next[:, t] = grad_g.dot(self.U_g) + grad_i.dot(self.U_i) + grad_f(self.U_f) + grad_o.dot(self.U_o)
            
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U_o += grad_o.T.dot(self.layer_input[:, t_])
                grad_W_o += grad_o.T.dot(self.states[:, t_-1])
                z_o = self.layer_input[:, t_-1].dot(self.U_o.T) + self.states[:, t_-2].dot(self.W_o.T)
                grad_o = grad_o.dot(self.W_o) * self.activation_1.gradient(z_o)

                grad_U_f += grad_f.T.dot(self.layer_input[:, t_])
                grad_W_f += grad_f.T.dot(self.states[:, t_-1])
                z_f = self.layer_input[:, t_-1].dot(self.U_f.T) + self.states[:, t_-2].dot(self.W_f.T)
                grad_f = grad_f.dot(self.W_f) * self.activation_1.gradient(z_f)

                grad_U_i += grad_i.T.dot(self.layer_input[:, t_])
                grad_W_i += grad_i.T.dot(self.states[:, t_-1])
                z_i = self.layer_input[:, t_-1].dot(self.U_i.T) + self.states[:, t_-2].dot(self.W_i.T)
                grad_i = grad_i.dot(self.W_i) * self.activation_1.gradient(z_i)

                grad_U_g += grad_g.T.dot(self.layer_input[:, t_])
                grad_W_g += grad_g.T.dot(self.states[:, t_-1])
                z_g = self.layer_input[:, t_-1].dot(self.U_g.T) + self.states[:, t_-2].dot(self.W_g.T)
                grad_g = grad_g.dot(self.W_g) * self.activation_2.gradient(z_g)
        
        self.U_o = self.U_o_opt.update(self.U_o, grad_U_o)
        self.W_o = self.W_o_opt.update(self.W_o, grad_W_o)
        self.U_f = self.U_f_opt.update(self.U_f, grad_U_f)
        self.W_f = self.W_f_opt.update(self.W_f, grad_W_f)
        self.U_i = self.U_i_opt.update(self.U_i, grad_U_i)
        self.W_i = self.W_i_opt.update(self.W_i, grad_W_i)
        self.U_g = self.U_g_opt.update(self.U_g, grad_U_g)
        self.W_g = self.W_g_opt.update(self.W_g, grad_W_g)
        self.V = self.V_opt.update(self.V, grad_V)

        return accum_grad_next

    def output_shape(self):
        return self.input_shape

class Dropout(Layer):
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
        self.trainable = True

    def forward_pass(self, X, training=True):
        if training:
            self.mask = np.random.binomial(n=1, p = 1 - self.p, size = X.shape)
        return X * self.mask
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.mask
    
    def output_shape(self):
        return self.input_shape
    










