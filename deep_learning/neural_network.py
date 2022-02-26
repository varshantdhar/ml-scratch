from __future__ import print_function, division
import numpy as np
from utils import batch_iterator
import wandb
import pickle

from tqdm import tqdm
from utils import random_generator
from deep_learning.optimizers import Adam, StochasticGradientAscent
from time import sleep

class DQN():
    
    def __init__(self, optimizer, time_network, action_network, amount_network, loss_function):
        self.optimizer = optimizer
        self.time_network = time_network
        self.action_network = action_network
        self.amount_network = amount_network
        self.loss_function = loss_function()
        self.lamb = 1
        self.eta = 10
    
    def set_trainable(self, trainable):
        self.time_network = trainable
        for action in self.action_network:
            action.trainable = trainable
        for amount in self.amount_network:
            amount.trainable = trainable
    
    def initialize(self):
        for time in self.time_network:
            if not time.initialized:
                time.initialize(optimizer=self.optimizer)
        for action in self.action_network:
            if not action.initalized:
                action.initialize(optimizer=self.optimizer)
        for amount in self.amount_network:
            if not amount.initalized:
                amount.initialize(optimizer=self.optimizer)
    
    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def _forward_pass(self, X, training=True):
        layer_output = X
        time_output = self.time_network.forward_pass(layer_output, training)

        for action in self.action_network:
            action_output = action.forward_pass(time_output, training)
        for amount in self.amount_network:
            amount_output = amount.forward_pass(time_output, training)
        layer_output = np.concatenate([action_output, amount_output], axis=1)

        return layer_output
    
    def _backward_pass(self, loss_grad):
        action_loss = loss_grad
        amount_loss = loss_grad
        for action in self.action_network:
            action_loss = action.backward_pass(action_loss)
        for amount in self.amount_network:
            amount_loss = amount.backward_pass(amount_loss)
        
        loss_grad = self.lamb * amount_loss + self.eta * action_loss
        loss_grad = self.time_network.backward_pass(loss_grad)
        
        return loss_grad
    
    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad = loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        wandb.init(project='DQN',
                  entity='varshantdhar')

        wandb.config = {
            "learning_rate": 1e-6,
            "epochs": n_epochs,
            "batch_size": batch_size
        }

        for epoch in tqdm(range(n_epochs)):

            batch_error = []
            for X_batch, y_batch in batch_iterator(X,y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                wandb.log({"loss": loss, "learning_rate": learning_rate / (np.power(10, (epoch // 10)))})
                batch_error.append(loss)
            
            self.errors["training"].append(np.mean(batch_error))
            
        return self.errors["training"]

    def predict(self, X):
        return self._forward_pass(X, training=False)



class TimeGAN():

    def __init__(self, optimizer, embedder, generator, recovery, supervisor,
                discriminator, reconstruction_loss, unsupervised_loss, supervised_loss):
        self.optimizer = optimizer
        self.embedder = embedder
        self.generator = generator
        self.supervisor = supervisor
        self.recovery = recovery
        self.discriminator = discriminator
        self.errors = {"embedder_training": [], "supervised_training":[], "discriminator_training":[],
                      "embedder_generator_training":[]}
        self.reconstruction_loss = reconstruction_loss()
        self.unsupervised_loss = unsupervised_loss()
        self.supervised_loss = supervised_loss()
        self.gamma = 1
    
    def set_trainable(self, trainable):
        self.embedder.trainable = trainable
        self.generator.trainable = trainable
        self.supervisor.trainable = trainable
        for recovery in self.recovery:
            recovery.trainable = trainable
        for discriminate in self.discriminator:
            discriminate.trainable = trainable

    def initialize(self):
        if not self.embedder.initialized:
            self.embedder.initialize(optimizer=self.optimizer)
        if not self.generator.initialized:
            self.generator.initialize(optimizer=self.optimizer)
        if not self.supervisor.initialized:
            self.supervisor.initialize(optimizer=self.optimizer)
        for recovery in self.recovery:
            if not recovery.initialized:
                recovery.initialize(optimizer=self.optimizer)
        for discriminate in self.discriminator:
            if not discriminate.initialized:
                discriminate.initialize(optimizer=self.optimizer)
    
    def embedder_training(self, X, training=True):
        # Embedder & Recovery
        H = self.embedder.forward_pass(X, training)
        output = H.copy()
        for recovery in self.recovery:
            output = recovery.forward_pass(output, training)
        X_tilde = output
         # Embedder Network Loss
        E_loss_T0 = self.reconstruction_loss.loss(X, X_tilde)
        E_loss_T0_grad = self.reconstruction_loss.gradient(X, X_tilde)
        E_loss0 = 10 * np.sqrt(E_loss_T0)
        E_loss0_grad = 5 / (np.sqrt(E_loss_T0_grad) + 1e-6)

        E_loss0_grad=np.clip(E_loss0_grad / (np.linalg.norm(E_loss0_grad) + 1e-5), 1e-4, 1.0)

        loss = E_loss0
        loss_grad = E_loss0_grad

        for recovery in reversed(self.recovery):
            loss_grad = recovery.backward_pass(loss_grad)
        self.embedder.backward_pass(loss_grad)

        wandb.log({"Embedder Network Loss": loss})

        return loss

    def supervised_training(self, X, training=True):
        H = self.embedder.forward_pass(X, training)
        H_hat_supervise = self.supervisor.forward_pass(H, training)

        # 2. Supervised loss
        G_loss_S = self.supervised_loss.loss(H, H_hat_supervise)
        G_loss_S_grad = self.supervised_loss.gradient(H, H_hat_supervise)

        G_loss_S_grad=np.clip(G_loss_S_grad / (np.linalg.norm(G_loss_S_grad) + 1e-5), 1e-4, 1.0)

        loss = G_loss_S
        loss_grad = G_loss_S_grad

        loss_grad = self.supervisor.backward_pass(loss_grad)
        self.generator.backward_pass(loss_grad)

        wandb.log({"Supervised Loss": loss})

        return loss

    def embedder_generator_training(self, X, Z, training=True):
        # Generator
        H = self.embedder.forward_pass(X, training)
        H_hat_supervise = self.supervisor.forward_pass(H, training)
        E_hat = self.generator.forward_pass(Z, training)
        H_hat = self.supervisor.forward_pass(E_hat, training)

        # Synthetic Data
        output = H_hat.copy()
        for recovery in self.recovery:
            output = recovery.forward_pass(output, training)
        X_hat = output

        # Discriminator
        output = H_hat.copy()
        for discriminate in self.discriminator:
            output = discriminate.forward_pass(output, training)
        Y_fake = output
        output = E_hat.copy()
        for discriminate in self.discriminator:
            output = discriminate.forward_pass(output, training)
        Y_fake_e = output

        # Generator loss
        # 1. Adversarial loss
        G_loss_U = self.unsupervised_loss.loss(np.ones_like(Y_fake), Y_fake)
        G_loss_U_grad = self.unsupervised_loss.gradient(np.ones_like(Y_fake), Y_fake)
        G_loss_U_e = self.unsupervised_loss.loss(np.ones_like(Y_fake_e), Y_fake_e)
        G_loss_U_e_grad = self.unsupervised_loss.gradient(np.ones_like(Y_fake_e), Y_fake_e)

        # 2. Supervised loss
        G_loss_S = self.supervised_loss.loss(H, H_hat_supervise)
        G_loss_S_grad = self.supervised_loss.gradient(H, H_hat_supervise)

        # 3. Two Momments
        G_loss_V1 = np.mean(np.abs(np.sqrt(np.var(X_hat, axis=0) + 1e-6) - np.sqrt(np.var(X, axis=0) + 1e-6)))
        G_loss_V2 = np.mean(np.abs(np.mean(X_hat, axis=0) - np.mean(X, axis=0)))
        G_loss_V = G_loss_V1 + G_loss_V2

        X_var = np.sqrt(np.var(X_hat, axis=0) + 1e-6) - np.sqrt(np.var(X, axis=0) + 1e-6)
        X_var_grad = (np.mean(X_hat - np.mean(X_hat, axis=0)))/np.sqrt(np.var(X_hat, axis=0) + 1e-6) - (np.mean(X - np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0) + 1e-6)
        G_loss_V1_grad = np.mean(X_var * X_var_grad / np.abs(X_var))

        X_mean = np.mean(X_hat, axis=0) - np.mean(X, axis=0)
        X_mean_grad = np.mean(np.ones_like(X_hat), axis=0) - np.mean(np.ones_like(X), axis=0)
        G_loss_V2_grad = np.mean(X_mean * X_mean_grad / np.abs(X_mean))
        
        G_loss_V_grad = G_loss_V1_grad + G_loss_V2_grad

        # 4. Summation
        G_loss = G_loss_U + self.gamma * G_loss_U_e + 100 * np.sqrt(G_loss_S) + 100 * G_loss_V
        G_loss_grad = G_loss_U_grad + self.gamma * G_loss_U_e_grad + 100 * np.sqrt(G_loss_S_grad) + 100 * G_loss_V_grad

        G_loss_grad=np.clip(G_loss_grad / (np.linalg.norm(G_loss_grad) + 1e-5), 1e-4, 1.0)

        loss_grad = G_loss_grad

        loss_grad = self.supervisor.backward_pass(loss_grad)
        self.generator.backward_pass(loss_grad)

        output = H.copy()
        for recovery in self.recovery:
            output = recovery.forward_pass(output, training)
        X_tilde = output
        # Embedder Network Loss
        E_loss_T0 = self.reconstruction_loss.loss(X, X_tilde)
        E_loss_T0_grad = self.reconstruction_loss.gradient(X, X_tilde)
        E_loss0 = 10 * np.sqrt(E_loss_T0)
        E_loss0_grad = 5 / (np.sqrt(E_loss_T0_grad) + 1e-6)
        E_loss = E_loss0  + 0.1 * G_loss_S
        E_loss_grad = E_loss0_grad + 0.1 * G_loss_S_grad

        E_loss_grad=np.clip(E_loss_grad / (np.linalg.norm(E_loss_grad) + 1e-5), 1e-4, 1.0)

        loss_grad_ = E_loss_grad
        
        for recovery in reversed(self.recovery):
            loss_grad_ = recovery.backward_pass(loss_grad_)
        self.embedder.backward_pass(loss_grad_)
        
        wandb.log({"Generator Loss": G_loss, "Embedder Loss": E_loss})

        return G_loss, E_loss

    def discriminator_training(self, X, Z, training=True):
        # Generator
        H = self.embedder.forward_pass(X, training)
        E_hat = self.generator.forward_pass(Z, training)
        H_hat = self.supervisor.forward_pass(E_hat, training)

        # Discriminator
        output = H_hat.copy()
        for discriminate in self.discriminator:
            output = discriminate.forward_pass(output, training)
        Y_fake = output
        output = E_hat.copy()
        for discriminate in self.discriminator:
            output = discriminate.forward_pass(output, training)
        Y_fake_e = output
        output = H.copy()
        for discriminate in self.discriminator:
            output = discriminate.forward_pass(output, training)
        Y_real = output

         # Discriminator loss
        D_loss_real = self.unsupervised_loss.loss(np.ones_like(Y_real), Y_real)
        D_loss_real_grad = self.unsupervised_loss.gradient(np.ones_like(Y_real), Y_real)
        D_loss_fake = self.unsupervised_loss.loss(np.ones_like(Y_fake), Y_fake)
        D_loss_fake_grad = self.unsupervised_loss.gradient(np.ones_like(Y_fake), Y_fake)
        D_loss_fake_e = self.unsupervised_loss.loss(np.ones_like(Y_fake_e), Y_fake_e)
        D_loss_fake_e_grad = self.unsupervised_loss.gradient(np.ones_like(Y_fake_e), Y_fake_e)
        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
        D_loss_grad = D_loss_real_grad + D_loss_fake_grad + self.gamma * D_loss_fake_e_grad

        loss = D_loss
        loss_grad = D_loss_grad

        for discriminate in self.discriminator:
            loss_grad = discriminate.backward_pass(loss_grad)
        
        wandb.log({"Discriminator Loss": loss})

        return loss


    def fit(self, X, z_dim, ori_time, max_seq_len, n_epochs, batch_size):
        wandb.init(project='timegan_generator',entity='varshantdhar')
        wandb.config = {"epochs": n_epochs,"batch_size": batch_size,"learning_rate":1e-3}
        
        for _ in tqdm(range(n_epochs)):
            embedder_error = []
            for X_batch in batch_iterator(X, batch_size=batch_size):
                loss = self.embedder_training(X_batch)
                embedder_error.append(loss)
            self.errors["embedder_training"].append(np.mean(embedder_error))
            sleep(1/n_epochs)
        
        with open("model/embedder.pkl","wb") as f:
            pickle.dump(self.embedder, f)
        with open("model/recovery.pkl","wb") as f:
            pickle.dump(self.recovery, f)
        
        for _ in tqdm(range(n_epochs)):
            supervised_error = []
            for X_batch in batch_iterator(X, batch_size=batch_size):
                loss = self.supervised_training(X_batch)
                supervised_error.append(loss)
            self.errors["supervised_training"].append(np.mean(supervised_error))
            sleep(1/n_epochs)
        
        with open("model/generator.pkl","wb") as f:
            pickle.dump(self.generator, f)
        with open("model/supervisor.pkl","wb") as f:
            pickle.dump(self.supervisor, f)
        
        for _ in tqdm(range(n_epochs)):
            embedder_generator_error = []
            discriminator_error = []
            for _ in range(2):
                for X_batch in batch_iterator(X, batch_size=batch_size):
                    Z_batch = np.array(random_generator(batch_size, z_dim, ori_time, max_seq_len))
                    loss = self.embedder_generator_training(X_batch, Z_batch)
                    embedder_generator_error.append(loss)    
                self.errors["embedder_generator_training"].append(np.mean(embedder_generator_error))
            for X_batch in batch_iterator(X, batch_size=batch_size):
                Z_batch = np.array(random_generator(batch_size, z_dim, ori_time, max_seq_len))
                loss = self.discriminator_training(X_batch, Z_batch)
                discriminator_error.append(loss)     
            self.errors["discriminator_training"].append(np.mean(discriminator_error))
            sleep(1/n_epochs)
        
        with open("model/embedder_joint.pkl","wb") as f:
            pickle.dump(self.embedder, f)
        with open("model/recovery_joint.pkl","wb") as f:
            pickle.dump(self.recovery, f)
        with open("model/generator_joint.pkl","wb") as f:
            pickle.dump(self.generator, f)
        with open("model/supervisor_joint.pkl","wb") as f:
            pickle.dump(self.supervisor, f)
        with open("model/discriminator.pkl","wb") as f:
            pickle.dump(self.discriminator, f)

    def generate(self, Z, training=True):
        E_hat = self.generator.forward_pass(Z, training)
        H_hat = self.supervisor.forward_pass(E_hat, training)

        # Synthetic Data
        output = H_hat.copy()
        for recovery in self.recovery:
            output = recovery.forward_pass(output, training)
        X_hat = output
        return X_hat


class NeuralNetwork():

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y":y}
    
    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        
        self.layers.append(layer)
    
    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def _forward_pass(self, X, training=True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        
        return layer_output
    
    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
        
        return loss_grad
    
    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad = loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        for _ in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X,y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            
            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
            
        return self.errors["training"], self.errors["validation"]

    def predict(self, X):
        return self._forward_pass(X, training=False)


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)



            
        