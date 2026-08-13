from flask import Flask, render_template, Response
from modules.data_engine import fetch_orbital_data
from modules.ml_model import DebrisRiskClassifier
from modules.conjunction_engine import analyze_conjunctions
from modules.orbit_sim import generate_3d_orbit_plot
import cv2

app = Flask(__name__)

classifier = DebrisRiskClassifier()


@app.route("/")
def index():

    # 1. Fetch orbital data
    df = fetch_orbital_data()

    # 2. ML risk prediction
    df_classified = classifier.predict_risk(df)

    # 3. Conjunction analysis
    priority_df = analyze_conjunctions(df_classified)

    # 4. 3D orbital plot
    plot_html = generate_3d_orbit_plot(df_classified)

    # 5. Critical debris count
    critical_count = len(
        df_classified[
            df_classified["risk_score"] > 70
        ]
    )

    # 6. Top threats
    if len(priority_df) > 0:
        top_threats = priority_df.head(10).to_dict(
            orient="records"
        )
    else:
        top_threats = df_classified.head(10).to_dict(
            orient="records"
        )

    return render_template(
        "index.html",
        plot_html=plot_html,
        threats=top_threats,
        total_tracked=len(df_classified),
        critical_count=critical_count
    )


def gen_frames():

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        h, w, _ = frame.shape

        # Optical tracker circle
        cv2.circle(
            frame,
            (w // 2, h // 2),
            60,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "OPTICAL TRACKER: ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    camera.release()


@app.route("/video_feed")
def video_feed():

    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )