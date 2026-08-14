import pandas as pd

from modules.orbit_sim import generate_3d_orbit_plot


data = {

    "name": [
        "Debris-1",
        "Debris-2",
        "Debris-3",
        "Debris-4",
        "Debris-5"
    ],

    "perigee_alt": [
        500,
        800,
        1200,
        650,
        950
    ],

    "inclination": [
        30,
        60,
        45,
        20,
        75
    ],

    "norad_id": [
        12345,
        67890,
        11111,
        22222,
        33333
    ],

    "risk_score": [
        85,
        60,
        25,
        75,
        45
    ]
}


df = pd.DataFrame(data)


html = generate_3d_orbit_plot(df)


with open(
    "orbit_test.html",
    "w",
    encoding="utf-8"
) as file:

    file.write(html)


print("3D Orbit generated successfully!")