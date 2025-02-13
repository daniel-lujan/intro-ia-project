from time import time

import requests


def main():
    pred_data = {
        "age": 60,
        "workclass": "Private",
        "education": "Bachelors",
        "marital_status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0,
        "native_country": "United-States",
    }

    print("Making Predict request")
    t = time()
    res = requests.post("http://api:8000/predict", json=pred_data)
    print(res.json())
    print(f"Predict request took {time() - t:.2f}s")
    print("===================================")

    print("Making Train request")
    t = time()
    res = requests.post("http://api:8000/train")
    print(res.json())
    print(f"Train request took {time() - t:.2f}s")


if __name__ == "__main__":
    main()
