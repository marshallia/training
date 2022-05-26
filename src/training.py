""" This file is used to train the model's top layer"""

import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback
import tensorflow as tf
import yaml

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py feature\n")
    sys.exit(1)

BASE_PATH = '/Users/marshallia/PycharmProjects/training'

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)
params = yaml.safe_load(open(os.path.join(BASE_PATH, "params.yaml")))["train"]

train_feature = os.path.join(sys.argv[1], "bottleneck_features_train.npy")
val_feature = os.path.join(sys.argv[1], "bottleneck_features_validation.npy")
train_label_output = os.path.join(sys.argv[1], "bottleneck_label_features_train.npy")
validation_label_output = os.path.join(sys.argv[1], "bottleneck_label_features_validation.npy")

# dimensions of our images.
batch_size = params["batch_size"]
epochs = params["epochs"]
dropout = params["dropout"]
optimizer = params["optimizer"]
model_name = os.path.join(BASE_PATH, params["model_name"])

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def train_top_model():
    """ This method is used to train the model's top layer"""

    train_data = np.load(open(train_feature, 'rb'))
    train_labels = np.load(open(train_label_output, 'rb'))
    validation_data = np.load(open(val_feature, 'rb'))
    validation_labels = np.load(open(validation_label_output, 'rb'))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=METRICS)

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=0,
              callbacks=[TqdmCallback(), CSVLogger("metrics.csv")])
    model.save(model_name)


train_top_model()
