from modules.data_engine import fetch_orbital_data
from modules.ml_model import DebrisRiskClassifier


df = fetch_orbital_data()

classifier = DebrisRiskClassifier()

result = classifier.predict_risk(df)

print("\n===== ML RISK RESULTS =====")

print(
    result[
        [
            "norad_id",
            "name",
            "perigee_alt",
            "rcs_size_m2",
            "risk_category",
            "risk_score"
        ]
    ].head(10)
)
