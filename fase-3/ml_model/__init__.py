import pickle

from sklearn.ensemble import RandomForestClassifier

from ml_model.preprocessing import build_prediction_vector
from models import PredictableRecord


def load_model() -> RandomForestClassifier:
    print("Loading dataset...", end="")

    with open("data/X-preprocessed.p", "rb") as f:
        X = pickle.load(f)

    with open("data/y-preprocessed.p", "rb") as f:
        y = pickle.load(f)

    sample_weights = X.pop("fnlwgt")

    print(" done ✅\nTraining model...", end="")

    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=10)
    model.fit(X.values, y.values.flatten(), sample_weight=sample_weights)

    print(" done ✅")

    return model


class Model:
    instance: RandomForestClassifier = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.model = load_model()
        return cls.instance

    def predict(self, data: PredictableRecord) -> bool:
        vector = build_prediction_vector(data)
        return bool(self.model.predict([vector])[0])

    @classmethod
    def retrain_model(cls) -> None:
        """Retrain the model with the data in `data` directory.
        See `Model.__new__` method."""

        cls.instance = None
        cls()
