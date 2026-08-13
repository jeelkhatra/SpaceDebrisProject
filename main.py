from modules.data_engine import fetch_orbital_data
from modules.ml_model import DebrisRiskClassifier
from modules.conjunction_engine import analyze_conjunctions


def main():

    print("=" * 60)
    print("       SPACE DEBRIS COLLISION RISK ENGINE")
    print("=" * 60)

    # ---------------------------------------
    # STEP 1: Fetch orbital data
    # ---------------------------------------

    print("\n[1] Fetching orbital data...")

    df = fetch_orbital_data()

    print(
        f"[INFO] {len(df)} orbital objects loaded."
    )

    # ---------------------------------------
    # STEP 2: ML risk classification
    # ---------------------------------------

    print("\n[2] Running ML risk classifier...")

    classifier = DebrisRiskClassifier()

    risk_df = classifier.predict_risk(df)

    print("[INFO] Risk classification completed.")

    # ---------------------------------------
    # STEP 3: Conjunction analysis
    # ---------------------------------------

    print("\n[3] Running conjunction analysis...")

    conjunction_df = analyze_conjunctions(
        risk_df
    )

    print("[INFO] Conjunction analysis completed.")

    # ---------------------------------------
    # STEP 4: Display priority queue
    # ---------------------------------------

    print("\n" + "=" * 60)
    print("          TOP 10 PRIORITY OBJECTS")
    print("=" * 60)

    display_columns = [
        "norad_id",
        "name",
        "miss_distance_km",
        "collision_prob",
        "delta_v_cost_ms",
        "priority_score",
        "risk_score"
    ]

    print(
        conjunction_df[
            display_columns
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("             ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
