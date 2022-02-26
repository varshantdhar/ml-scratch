from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import random_generator, shuffle_data
from deep_learning.optimizers import Adam
from deep_learning.loss_functions import MeanSquaredError, CrossEntropy
from deep_learning.layers import RNN, Activation, Dense, BiRNN
from deep_learning.neural_network import TimeGAN
from tqdm import tqdm
from time import sleep


def main():
    optimizer = Adam()

    with open("data.pkl","rb") as f:
        data = pickle.load(f)
    print("Data Loaded")

    no, seq_len, dim = data.shape
    ori_time = [seq_len] * no
    max_seq_len = seq_len
    X = shuffle_data(data.copy(), seed=42)
    print("Data Shuffled")
    z_dim = 12

    # Build network
    # Embedding network
    rnn_units = 200
    embedder = RNN(n_units=rnn_units, activation='tanh', bptt_trunc=12, input_shape=(max_seq_len, dim))
    # Generator network
    generator = RNN(n_units=rnn_units, activation='tanh', bptt_trunc=12, input_shape=(max_seq_len, z_dim))
    # Supervisor network
    supervisor = RNN(n_units=rnn_units, activation='tanh', bptt_trunc=12, input_shape=(max_seq_len, rnn_units))
    # Recovery network
    recovery = []
    # recovery.append()
    recovery.append(Dense(n_units=dim, input_shape=(max_seq_len, rnn_units)))
    recovery.append(Activation('sigmoid'))

    # Discrimator Network
    discriminator = []
    birnn_units = 300
    discriminator.append(BiRNN(n_units=birnn_units, activation='tanh', bptt_trunc=12, input_shape=(max_seq_len, rnn_units)))
    discriminator.append(Dense(n_units=1, input_shape=(max_seq_len, birnn_units)))
    discriminator.append(Activation('sigmoid'))

    model = TimeGAN(
        optimizer = optimizer,
        embedder = embedder, 
        generator = generator, 
        supervisor = supervisor,
        recovery = recovery, 
        discriminator = discriminator,
        reconstruction_loss = MeanSquaredError, 
        unsupervised_loss = CrossEntropy, 
        supervised_loss = MeanSquaredError
    )

    model.initialize()

    model.fit(X, z_dim, ori_time, max_seq_len, n_epochs=500, batch_size=512)

    for i in tqdm(range(100)):
        Z = np.array(random_generator(no, z_dim, ori_time, max_seq_len))
        X_hat = model.generate(Z)
        with open("synthetic_data/synthetic_set_" + str(i) + ".pkl","wb") as f:
            pickle.dump(X_hat, f)
        sleep(1/100)


if __name__ == "__main__":
    main()
