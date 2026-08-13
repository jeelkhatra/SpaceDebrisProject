import os
import json
import requests
import numpy as np
import pandas as pd


# CelesTrak API URL
CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=active&FORMAT=json"
)

# Location of offline cache file
CACHE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/cached_debris_tle.json"
)


def fetch_orbital_data():
    """
    Fetch satellite orbital data from CelesTrak.

    If internet connection fails, the function loads
    previously saved data from the local cache.
    """

    try:
        print("[INFO] Connecting to CelesTrak...")

        response = requests.get(
            CELESTRAK_URL,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            # Make sure data folder exists
            os.makedirs(
                os.path.dirname(CACHE_PATH),
                exist_ok=True
            )

            # Save data for offline use
            with open(CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)

            print(
                "[INFO] Live data fetched and cached "
                "offline successfully."
            )

            return parse_tle_data(data)

        else:
            print(
                f"[WARN] CelesTrak returned "
                f"status code {response.status_code}"
            )

    except Exception as e:
        print(
            f"[WARN] Connection failed ({e}). "
            "Loading offline dataset..."
        )

    # Offline fallback
    if os.path.exists(CACHE_PATH):

        print("[INFO] Loading offline cached data...")

        with open(CACHE_PATH, "r") as f:
            data = json.load(f)

        return parse_tle_data(data)

    else:
        raise FileNotFoundError(
            "No offline cache found! "
            "Please connect to the internet and run again."
        )


def parse_tle_data(raw_json):
    """
    Parse orbital parameters and calculate
    perigee and apogee altitudes.
    """

    parsed = []

    # Use maximum 100 objects
    for obj in raw_json[:100]:

        name = obj.get(
            "OBJECT_NAME",
            "DEBRIS FRAGMENT"
        )

        norad_id = obj.get(
            "NORAD_CAT_ID",
            0
        )

        inclination = float(
            obj.get("INCLINATION", 0.0)
        )

        eccentricity = float(
            obj.get("ECCENTRICITY", 0.0)
        )

        mean_motion = float(
            obj.get("MEAN_MOTION", 15.0)
        )

        bstar = float(
            obj.get("BSTAR", 0.0001)
        )

        # -----------------------------------------
        # Calculate Semi-Major Axis
        # -----------------------------------------

        # Mean motion:
        # revolutions/day → radians/second

        n_rad = (
            mean_motion * (2 * np.pi)
        ) / 86400.0

        # Earth's gravitational parameter
        # km^3 / s^2

        mu = 398600.4418

        # Semi-major axis in km

        semi_major_axis = (
            mu / (n_rad ** 2)
        ) ** (1 / 3)

        # -----------------------------------------
        # Calculate Perigee and Apogee
        # -----------------------------------------

        earth_radius = 6371.0

        perigee_alt = (
            semi_major_axis *
            (1 - eccentricity)
        ) - earth_radius

        apogee_alt = (
            semi_major_axis *
            (1 + eccentricity)
        ) - earth_radius

        # -----------------------------------------
        # Estimated Object Size
        # -----------------------------------------

        # Temporary size proxy.
        # This should eventually be replaced
        # with real RCS/size data.

        rcs_estimate = np.round(
            np.random.uniform(0.1, 5.5),
            2
        )

        # -----------------------------------------
        # Store Object
        # -----------------------------------------

        parsed.append({
            "norad_id": norad_id,
            "name": name,
            "inclination": inclination,
            "eccentricity": eccentricity,
            "perigee_alt": perigee_alt,
            "apogee_alt": apogee_alt,
            "mean_motion": mean_motion,
            "bstar_drag": bstar,
            "rcs_size_m2": rcs_estimate
        })

    return pd.DataFrame(parsed)
