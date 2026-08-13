import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class DebrisRiskClassifier:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False

    def _generate_synthetic_labels(self, df):
        """
        Generate baseline risk labels.

        0 = Low
        1 = Medium
        2 = High
        """

        risk_score = (
            (1000 - np.clip(
                df["perigee_alt"],
                300,
                1000
            )) / 10.0

            + (df["rcs_size_m2"] * 15)

            + (np.abs(df["bstar_drag"]) * 10000)
        )

        labels = pd.qcut(
            risk_score,
            q=3,
            labels=[0, 1, 2]
        )

        return labels.astype(int)

    def train(self, df):

        features = [
            "inclination",
            "eccentricity",
            "perigee_alt",
            "apogee_alt",
            "bstar_drag",
            "rcs_size_m2"
        ]

        X = df[features]

        y = self._generate_synthetic_labels(df)

        self.model.fit(X, y)

        self.is_trained = True

        print(
            "[INFO] ML Debris Risk Classifier trained."
        )

    def predict_risk(self, df):

        if not self.is_trained:
            self.train(df)

        features = [
            "inclination",
            "eccentricity",
            "perigee_alt",
            "apogee_alt",
            "bstar_drag",
            "rcs_size_m2"
        ]

        X = df[features]

        probabilities = self.model.predict_proba(X)

        risk_classes = self.model.predict(X)

        risk_scores = (
            probabilities[:, 1] * 50
            + probabilities[:, 2] * 100
        ).round(2)

        df["risk_category"] = risk_classes

        df["risk_score"] = risk_scores

        return df
