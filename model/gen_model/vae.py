from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

class VAE():
    def __init__(
        self,
        n_input: int,
        n_labels: int = 10,
        n_batch: int = 64,
        n_latent: int = 100,
        n_hidden: int = 128,
        n_cont_cov: int = 0,
        dropout_rate: float = 1e-3
    ):
        self.n_input = n_input,
        self.n_labels = n_labels,
        self.n_batch = n_batch,
        self.n_latent = n_latent,
        self.n_hidden = n_hidden,
        self.n_cont_cov = n_cont_cov,
        self.dropout_rate = dropout_rate

    def xavier_init(self, shape):
        in_dim = shape[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=shape, stddev=xavier_stddev)

    def getQzX(self):
        X = tf.placeholder(tf.float32, shape=[None, self.n_input])
        z = tf.placeholder(tf.float32, shape=[None, self.n_latent])

        Q_W1 = tf.Variable(self.xavier_init([self.n_input, self.n_hidden]))
        Q_b1 = tf.Variable(tf.zeros(shape=[self.n_hidden]))

        Q_W2_mu = tf.Variable(self.xavier_init([self.n_hidden, self.n_latent]))
        Q_b2_mu = tf.Variable(tf.zeros(shape=[self.n_latent]))

        Q_W2_sigma = tf.Variable(self.xavier_init([self.n_hidden, self.n_latent]))
        Q_b2_sigma = tf.Variable(tf.zeros(shape=[self.n_latent]))

        h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
        z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
        z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma

        return z_mu, z_logvar
        
