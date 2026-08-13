import numpy as np
import pandas as pd


# Target spacecraft
TARGET_ASSET = {
    "name": "ISS (International Space Station)",
    "altitude_km": 420.0,
    "inclination_deg": 51.64
}


def analyze_conjunctions(df):
    """
    Calculates:
    - Miss distance
    - Collision probability
    - Delta-V cost
    - Mission priority
    """

    conjunction_results = []

    for _, row in df.iterrows():

        # Difference in altitude
        alt_diff = abs(
            row["perigee_alt"]
            - TARGET_ASSET["altitude_km"]
        )

        # Difference in inclination
        inc_diff = abs(
            row["inclination"]
            - TARGET_ASSET["inclination_deg"]
        )

        # Approximate miss distance
        miss_distance_km = np.sqrt(
            alt_diff ** 2
            + (inc_diff * 111.0) ** 2
        )

        # Position uncertainty
        sigma = 1.5

        # Collision probability
        collision_prob = np.exp(
            -0.5 * (miss_distance_km / sigma) ** 2
        )

        # Estimated Delta-V cost
        delta_v_cost = (
            50
            + (alt_diff * 1.2)
            + (inc_diff * 15.0)
        )

        # Priority score
        priority_score = (
            (row["risk_score"] * 0.6)
            + (collision_prob * 100 * 0.4)
        )

        conjunction_results.append({
            "norad_id": row["norad_id"],
            "name": row["name"],
            "miss_distance_km": round(
                miss_distance_km, 2
            ),
            "collision_prob": float(
                f"{collision_prob:.6f}"
            ),
            "delta_v_cost_ms": round(
                delta_v_cost, 1
            ),
            "priority_score": round(
                priority_score, 2
            ),
            "risk_score": row["risk_score"]
        })

    result_df = pd.DataFrame(
        conjunction_results
    )

    return result_df.sort_values(
        by="priority_score",
        ascending=False
    ).reset_index(drop=True)
