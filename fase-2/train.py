import os
import pickle

import pandas as pd
from common import preprocess_data, preprocess_output
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 10
data = pd.read_csv("./input/train.csv")

X_train = data.drop(columns=["income"])
y_train = data["income"]

X_train = preprocess_data(X_train)
y_train = preprocess_output(y_train)

sample_weights_train = X_train.pop("fnlwgt")

rf_classifier = RandomForestClassifier(
    random_state=RANDOM_STATE, n_estimators=500, max_depth=20
)
rf_classifier.fit(X_train, y_train, sample_weight=sample_weights_train)


output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open("./output/model.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)
