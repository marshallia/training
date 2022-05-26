""" This file is used to extract features from dataset"""

import sys
import os
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
    sys.exit(1)

BASE_PATH = '/Users/marshallia/PycharmProjects/training'

params = yaml.safe_load(open(os.path.join(BASE_PATH, "params.yaml")))["feature"]
img_width = params["img_width"]
img_height = params["img_height"]
batch_size = params["batch_size"]

train_data_dir = os.path.join(sys.argv[1], "train")
validation_data_dir = os.path.join(sys.argv[1], "validation")
test_data_dir = os.path.join(sys.argv[1], "test")

train_output = os.path.join(sys.argv[2], "bottleneck_features_train.npy")
validation_output = os.path.join(sys.argv[2], "bottleneck_features_validation.npy")
test_output = os.path.join(sys.argv[2], "bottleneck_features_test.npy")

train_label_output = os.path.join(sys.argv[2], "bottleneck_label_features_train.npy")
validation_label_output = os.path.join(sys.argv[2], "bottleneck_label_features_validation.npy")
test_label_output = os.path.join(sys.argv[2], "bottleneck_label_features_test.npy")


def generator(data_path):
    """ This file is used to generate features from dataset"""

    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        data_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, )


def generate_label(path, val=False):
    """ This method is used to generate label"""

    if val:
        nb_samples = 2 * (len([name for name in os.listdir(path)
                               if os.path.isfile(
                os.path.join(path, name))]) - 1)
    else:
        nb_samples = 2 * len([name for name in os.listdir(path)
                              if os.path.isfile(
                os.path.join(path, name))])
    return np.array(
        [0] * (int(nb_samples / 2)) + [1] * (int(nb_samples / 2))), nb_samples


def save_features(path, feature):
    """ This file use to create file from generated features"""

    np.save(open(path, 'wb'),
            feature)


def save_bottlebeck_features():
    """ This file act as main method to call other methods"""

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    cats_train_path = os.path.join(train_data_dir, 'cats')
    train_labels, nb_train_samples = generate_label(cats_train_path)
    save_features(train_label_output, train_labels)
    train_generator = generator(train_data_dir)
    bottleneck_features_train = model.predict(
        train_generator, nb_train_samples // batch_size)
    save_features(train_output, bottleneck_features_train)

    cats_val_path = os.path.join(validation_data_dir, 'cats')
    val_generator = generator(validation_data_dir)
    validation_labels, nb_validation_samples = generate_label(cats_val_path, True)
    save_features(validation_label_output, validation_labels)
    bottleneck_features_validation = model.predict(
        val_generator, nb_validation_samples // batch_size)
    save_features(validation_output, bottleneck_features_validation)

    cats_test_path = os.path.join(test_data_dir, 'cats')
    test_label, nb_test_samples = generate_label(cats_test_path)
    save_features(test_label_output, test_label)
    test_generator = generator(test_data_dir)
    bottleneck_features_test = model.predict(test_generator, nb_test_samples // batch_size)
    save_features(test_output, bottleneck_features_test)


save_bottlebeck_features()
