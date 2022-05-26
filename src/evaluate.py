""" This file is used to evaluate the model performance"""

import sys
import os
import json
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm.keras import TqdmCallback
import numpy as np
import yaml

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features model output_plot\n")
    sys.exit(1)
BASE_PATH = '/home/kuro/PycharmProjects/training' # change this according ur need

params = yaml.safe_load(open(os.path.join(BASE_PATH, "params.yaml")))["evaluate"]
batch_size = params["batch_size"]

test_feature = os.path.join(sys.argv[2], "bottleneck_features_test.npy")
test_label_feature = os.path.join(sys.argv[2], "bottleneck_label_features_test.npy")
model_name = os.path.join(BASE_PATH, sys.argv[1])
plot_output = os.path.join(BASE_PATH, sys.argv[3])
prc_path = os.path.join(plot_output, 'prc.json')
roc_path = os.path.join(plot_output, 'roc.json')


def evaluate():
    """ This method is used to evaluate the model performance"""

    model = tf.keras.models.load_model(model_name)
    test_data = np.load(open(test_feature, 'rb'))
    test_label = np.load(open(test_label_feature, 'rb'))

    evaluate = model.evaluate(test_data, test_label, batch_size=batch_size, return_dict=True,
                              callbacks=[TqdmCallback()])
    out_file = "test_result.json"
    with open(out_file, "w") as file:
        json.dump(evaluate, file,
                  indent=4, )
    prediction = model.predict(test_data, callbacks=[TqdmCallback()])

    precision, recall, thresholds = precision_recall_curve(test_label, prediction)
    save_ploting(prc_path, 'prc', precision, recall, thresholds,
                 ['precision', 'recall', 'threshold'])

    fpr, tpr, thresholds = roc_curve(test_label, prediction)
    save_ploting(roc_path, 'roc', fpr, tpr, thresholds, ['fpr', 'tpr', 'threshold'])


def save_ploting(path, name, list_val1, list_val2, thresholds, keys):
    """ This method is used to save ploting points"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as file:
        json.dump(
            {
                name: [
                    {keys[0]: str(val1), keys[1]: str(val2), keys[2]: str(val3)}
                    for val1, val2, val3 in zip(list_val1, list_val2, thresholds)
                ]
            },
            file,
            indent=4,
        )


evaluate()
