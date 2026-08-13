from modules.data_engine import fetch_orbital_data
from modules.ml_model import DebrisRiskClassifier
from modules.conjunction_engine import analyze_conjunctions


# Step 1: Get orbital data
df = fetch_orbital_data()


# Step 2: Train ML model
classifier = DebrisRiskClassifier()


# Step 3: Calculate risk
risk_df = classifier.predict_risk(df)


# Step 4: Analyze conjunctions
conjunction_df = analyze_conjunctions(risk_df)


# Step 5: Display results
print("\n===== CONJUNCTION ANALYSIS =====")

print(
    conjunction_df[
        [
            "norad_id",
            "name",
            "miss_distance_km",
            "collision_prob",
            "delta_v_cost_ms",
            "priority_score",
            "risk_score"
        ]
    ].head(10)
)
