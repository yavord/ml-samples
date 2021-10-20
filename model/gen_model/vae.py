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
        n_hidden: int = 128
    ):
        pass
