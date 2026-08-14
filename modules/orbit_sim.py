import numpy as np
import plotly.graph_objects as go


def generate_3d_orbit_plot(debris_df):

    fig = go.Figure()

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    r_earth = 6371

    x_e = r_earth * np.outer(
        np.cos(u),
        np.sin(v)
    )

    y_e = r_earth * np.outer(
        np.sin(u),
        np.sin(v)
    )

    z_e = r_earth * np.outer(
        np.ones(np.size(u)),
        np.cos(v)
    )

    fig.add_trace(
        go.Surface(
            x=x_e,
            y=y_e,
            z=z_e,
            colorscale="Blues",
            showscale=False,
            opacity=0.75,
            name="Earth"
        )
    )

    x_d = []
    y_d = []
    z_d = []

    colors = []
    hover_texts = []

    for _, row in debris_df.iterrows():

        r = r_earth + row["perigee_alt"]

        inc = np.radians(row["inclination"])

        raan = np.radians(
            row["norad_id"] % 360
        )

        x_d.append(
            r * np.cos(raan)
        )

        y_d.append(
            r * np.sin(raan) * np.cos(inc)
        )

        z_d.append(
            r * np.sin(raan) * np.sin(inc)
        )

        risk = row["risk_score"]

        if risk > 70:
            colors.append("#ff3333")

        elif risk > 40:
            colors.append("#ffbb00")

        else:
            colors.append("#00cc66")

        hover_texts.append(
            f"{row['name']}<br>"
            f"Alt: {round(row['perigee_alt'], 1)} km<br>"
            f"Risk: {risk}"
        )

    fig.add_trace(
        go.Scatter3d(
            x=x_d,
            y=y_d,
            z=z_d,
            mode="markers",
            marker=dict(
                size=5,
                color=colors,
                opacity=0.9
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Space Debris"
        )
    )

    fig.update_layout(
        template="plotly_dark",

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=20
        ),

        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )