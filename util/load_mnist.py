from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import numpy as np
import collections


class MnistDataset():
    def __init__(
        self,
        mnist: tuple = tf.keras.datasets.mnist.load_data(),
        x_train: np.ndarray = None,
        y_train: np.ndarray = None,
        x_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ):
        self.mnist = mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist

    def one_hot(self):
        self.y_train, self.y_test = to_categorical(self.y_train), to_categorical(self.y_test)

    def scale_img(self):
        self.x_train, self.x_test = self.x_train[..., np.newaxis]/255.0, self.x_test[..., np.newaxis]/255.0

    def remove_contradicting(self, xs: np.ndarray, ys: np.ndarray):
        mapping = collections.defaultdict(set)
        orig_x = {}
        # Determine the set of labels for each unique image:
        for x,y in zip(xs,ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass

        print("Initial number of images: ", len(xs))
        print("Remaining non-contradicting unique images: ", len(new_x))

        return np.array(new_x), np.array(new_y)
