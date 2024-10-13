import os
import pickle

import pandas as pd
from common import preprocess_data
from sklearn.ensemble import RandomForestClassifier

with open("model.pkl", "rb") as file:
    model: RandomForestClassifier = pickle.load(file)

X_test = preprocess_data(
    pd.read_csv("./input/test.csv"), columns=model.feature_names_in_
)

y_pred = model.predict(X_test)

output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

y_pred_df = pd.DataFrame(y_pred, columns=["income"])
y_pred_df["income"] = y_pred_df["income"].map({0: "<=50K", 1: ">50K"})
y_pred_df.to_csv("./output/predictions.csv", index=False, header=False)

print("Predictions saved to", os.path.abspath(output_dir))
