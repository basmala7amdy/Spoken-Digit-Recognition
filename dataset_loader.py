import os
import numpy as np
from feature_extraction import extract_feature

DATASET_PATH = "dataset"


def load_dataset():

    X = []
    y = []

    for label_folder in os.listdir(DATASET_PATH):

        folder_path = os.path.join(DATASET_PATH, label_folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                feature = extract_feature(file_path)

                X.append(feature)
                y.append(int(label_folder))

    return np.array(X), np.array(y)