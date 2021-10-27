import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# tf.debugging.set_log_device_placement(True)

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
        self.n_input = n_input
        self.n_labels = n_labels
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_cont_cov = n_cont_cov
        self.dropout_rate = dropout_rate

    def xavier_init(
        self, 
        shape
    ):
        in_dim = shape[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=shape, stddev=xavier_stddev)

    def encoder(
        self,
        X
    ):
        Q_W1 = tf.constant(self.xavier_init([self.n_input, self.n_hidden]))
        Q_b1 = tf.constant(tf.zeros(shape=[self.n_hidden]))

        Q_W2_mu = tf.constant(self.xavier_init([self.n_hidden, self.n_latent]))
        Q_b2_mu = tf.constant(tf.zeros(shape=[self.n_latent]))

        Q_W2_sigma = tf.constant(self.xavier_init([self.n_hidden, self.n_latent]))
        Q_b2_sigma = tf.constant(tf.zeros(shape=[self.n_latent]))

        h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
        z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
        z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma

        return z_mu, z_logvar

    def decoder(
        self,
        z
    ):
        P_W1 = tf.constant(self.xavier_init([self.n_latent, self.n_hidden]))
        P_b1 = tf.constant(tf.zeros(shape=[self.n_hidden]))

        P_W2 = tf.constant(self.xavier_init([self.n_hidden, self.n_input]))
        P_b2 = tf.constant(tf.zeros(shape=[self.n_input]))

        h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
        logits = tf.matmul(h, P_W2) + P_b2
        prob = tf.nn.sigmoid(logits)

        return prob, logits

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def train(
        self
    ):
        z = tf.keras.Input(shape=(None,self.n_latent), dtype=tf.dtypes.float32)
        X = tf.keras.Input(shape=(None,self.n_input), dtype=tf.dtypes.float32)

        z_mu, z_logvar = self.encoder(X)
        z_sample = self.sample_z(z_mu,z_logvar)
        _, logits = self.decoder(z_sample)
        X_samples, _ = self.decoder(z)

        # E[log P(X|z)]
        recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
        # D_KL(Q(z|X) || P(z))
        kl_loss= 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # VAE loss
        vae_loss = tf.reduce_mean(recon_loss + kl_loss)