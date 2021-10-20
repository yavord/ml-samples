from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import numpy as np
import collections

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# one-hot encoding labels
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# min-max scaling pixels
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

def remove_contradicting(xs, ys):
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

class MnistDataset():
    def __init__(self) -> None:
        pass