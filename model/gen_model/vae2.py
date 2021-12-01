import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


tfkl.Dense #encoder ouput
tfkl.Conv2DTranspose # decoder output

# loss
tf.GradientTape
tf.nn.sigmoid_cross_entropy_with_logits

def vae_loss(x, model, analytic_kl = True, kl_weight = 1.0):
    z_sample, mu, sd = model.encode(x)
    x_recons_logits = model.decoder(z_sample)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels= x,
        logits= x_recons_logits
    )

    neg_log_likelihood = tf.math.reduce_sum(
        cross_ent, 
        axis=[1,2,3]
        )

    if analytic_kl:
        kl_divergence = -0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sdaxis=1)
        )
    else:
        # TODO: add Monte Carlo approximation
        pass

    elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)
    return -elbo

class Encoder(tfk.layers.Layer): 

    def __init__(self, dim_z, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z