import numpy as np
from sklearn.preprocessing import OneHotEncoder

from models import PredictableRecord

WORKCLASS_MODE = "Private"
NATIVE_COUNTRY_MODE = "United-States"

EDUCATION_CAT_ORDER = [
    "Preschool",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "HS-grad",
    "Some-college",
    "Assoc-voc",
    "Assoc-acdm",
    "Bachelors",
    "Masters",
    "Prof-school",
    "Doctorate",
]


def build_prediction_vector(data: PredictableRecord) -> np.array:
    vector = np.array([])
    vector = np.append(vector, data.age)
    vector = np.append(vector, EDUCATION_CAT_ORDER.index(data.education) + 1)
    vector = np.append(vector, data.capital_gain)
    vector = np.append(vector, data.capital_loss)
    vector = np.append(vector, data.hours_per_week)

    workclass_encoder = OneHotEncoder(
        categories=[
            [
                "Federal-gov",
                "Local-gov",
                "Never-worked",
                "Private",
                "Self-emp-inc",
                "Self-emp-not-inc",
                "State-gov",
                "Without-pay",
            ]
        ]
    )
    vector = np.append(
        vector,
        workclass_encoder.fit_transform(
            [[data.workclass if data.workclass is not None else WORKCLASS_MODE]]
        )
        .toarray()
        .flatten(),
    )

    marital_status_encoder = OneHotEncoder(
        categories=[
            [
                "Divorced",
                "Married-AF-spouse",
                "Married-civ-spouse",
                "Married-spouse-absent",
                "Never-married",
                "Separated",
                "Widowed",
            ]
        ]
    )
    vector = np.append(
        vector,
        marital_status_encoder.fit_transform([[data.marital_status]])
        .toarray()
        .flatten(),
    )

    occupation_encoder = OneHotEncoder(
        categories=[
            [
                "?",
                "Adm-clerical",
                "Armed-Forces",
                "Craft-repair",
                "Exec-managerial",
                "Farming-fishing",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Other-service",
                "Priv-house-serv",
                "Prof-specialty",
                "Protective-serv",
                "Sales",
                "Tech-support",
                "Transport-moving",
            ]
        ]
    )
    vector = np.append(
        vector,
        occupation_encoder.fit_transform(
            [[data.occupation if data.occupation is not None else "?"]]
        )
        .toarray()
        .flatten(),
    )

    relationship_encoder = OneHotEncoder(
        categories=[
            [
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Own-child",
                "Unmarried",
                "Wife",
            ]
        ]
    )
    vector = np.append(
        vector,
        relationship_encoder.fit_transform([[data.relationship]]).toarray().flatten(),
    )

    race_encoder = OneHotEncoder(
        categories=[
            ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]
        ]
    )
    vector = np.append(
        vector, race_encoder.fit_transform([[data.race]]).toarray().flatten()
    )

    sex_encoder = OneHotEncoder(categories=[["Female", "Male"]])
    vector = np.append(
        vector, sex_encoder.fit_transform([[data.sex]]).toarray().flatten()
    )

    native_country_encoder = OneHotEncoder(
        categories=[
            [
                "Cambodia",
                "Canada",
                "China",
                "Columbia",
                "Cuba",
                "Dominican-Republic",
                "Ecuador",
                "El-Salvador",
                "England",
                "France",
                "Germany",
                "Greece",
                "Guatemala",
                "Haiti",
                "Holand-Netherlands",
                "Honduras",
                "Hong",
                "Hungary",
                "India",
                "Iran",
                "Ireland",
                "Italy",
                "Jamaica",
                "Japan",
                "Laos",
                "Mexico",
                "Nicaragua",
                "Outlying-US(Guam-USVI-etc)",
                "Peru",
                "Philippines",
                "Poland",
                "Portugal",
                "Puerto-Rico",
                "Scotland",
                "South",
                "Taiwan",
                "Thailand",
                "Trinadad&Tobago",
                "United-States",
                "Vietnam",
                "Yugoslavia",
            ]
        ]
    )
    vector = np.append(
        vector,
        native_country_encoder.fit_transform(
            [
                [
                    (
                        data.native_country
                        if data.native_country is not None
                        else NATIVE_COUNTRY_MODE
                    )
                ]
            ]
        )
        .toarray()
        .flatten(),
    )

    return vector
