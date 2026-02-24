"""
Flask backend za Personality Predictor aplikaciju.
Komunicira s Azure ML endpointom i sprema rezultate u SQLite bazu.
Koristi Pandas za obradu podataka i Plotly Express za grafikone.

Instalacija:
    pip install flask flask-cors requests pandas plotly

Pokretanje:
    python server.py
"""

import os
import sqlite3
import json
import random
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Konfiguracija Azure ML endpointa ─────────────────────────────────────────
AZURE_ENDPOINT = os.environ.get(
    "AZURE_ENDPOINT",
    "http://1d80b764-3a0a-4c2c-b5a6-353e66c7ac64.polandcentral.azurecontainer.io/score" 
)
AZURE_API_KEY = os.environ.get(
    "AZURE_API_KEY",
    "swxfWqkq5R8zMKRiS1INkHS7gcBKCXSH"
)

DB_PATH = "predictions.db"


# ─── Inicijalizacija baze i seed s Kaggle-like podacima ───────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp             TEXT    NOT NULL,
            time_spent_alone      REAL,
            stage_fear            TEXT,
            social_event_attend   REAL,
            going_outside         REAL,
            drained_after_social  TEXT,
            friends_circle_size   REAL,
            post_frequency        REAL,
            prediction            TEXT,
            probability           REAL
        )
    """)
    conn.commit()
    count = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if count == 0:
        _seed_kaggle_data(c)
        conn.commit()
    conn.close()


def _seed_kaggle_data(cursor):
    """
    Generira 100 sintetičkih zapisa koji simuliraju distribuciju
    Kaggle dataseta 'Extrovert vs Introvert Behavior Data'.
    Pandas se koristi za strukturiranje i validaciju podataka prije unosa u bazu.
    """
    random.seed(2024)
    rows = []

    for _ in range(100):
        is_introvert = random.random() < 0.52

        if is_introvert:
            row = {
                "timestamp":            f"2025-{random.randint(1,9):02d}-{random.randint(1,28):02d}T{random.randint(8,22):02d}:{random.randint(0,59):02d}:00",
                "time_spent_alone":     round(random.uniform(5, 11), 1),
                "stage_fear":           "Yes" if random.random() < 0.72 else "No",
                "social_event_attend":  round(random.uniform(0, 3.5), 1),
                "going_outside":        round(random.uniform(0, 2.5), 1),
                "drained_after_social": "Yes" if random.random() < 0.80 else "No",
                "friends_circle_size":  round(random.uniform(0, 5), 1),
                "post_frequency":       round(random.uniform(0, 3), 1),
                "prediction":           "Introvert",
                "probability":          round(random.uniform(0.60, 0.97), 4),
            }
        else:
            row = {
                "timestamp":            f"2025-{random.randint(1,9):02d}-{random.randint(1,28):02d}T{random.randint(8,22):02d}:{random.randint(0,59):02d}:00",
                "time_spent_alone":     round(random.uniform(0, 3.5), 1),
                "stage_fear":           "Yes" if random.random() < 0.18 else "No",
                "social_event_attend":  round(random.uniform(5.5, 10), 1),
                "going_outside":        round(random.uniform(4, 7), 1),
                "drained_after_social": "Yes" if random.random() < 0.12 else "No",
                "friends_circle_size":  round(random.uniform(7, 15), 1),
                "post_frequency":       round(random.uniform(5, 10), 1),
                "prediction":           "Extrovert",
                "probability":          round(random.uniform(0.03, 0.38), 4),
            }
        rows.append(row)

    # Pandas DataFrame za validaciju i clip vrijednosti
    df = pd.DataFrame(rows)
    df["time_spent_alone"]    = df["time_spent_alone"].clip(0, 11)
    df["social_event_attend"] = df["social_event_attend"].clip(0, 10)
    df["going_outside"]       = df["going_outside"].clip(0, 7)
    df["friends_circle_size"] = df["friends_circle_size"].clip(0, 15)
    df["post_frequency"]      = df["post_frequency"].clip(0, 10)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, time_spent_alone, stage_fear, social_event_attend,
                going_outside, drained_after_social, friends_circle_size,
                post_frequency, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["timestamp"], row["time_spent_alone"], row["stage_fear"],
            row["social_event_attend"], row["going_outside"],
            row["drained_after_social"], row["friends_circle_size"],
            row["post_frequency"], row["prediction"], row["probability"]
        ))


init_db()


# ─── DB helper ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def to_yes_no(value):
    return "Yes" if int(value) == 1 else "No"


# ─── RUTA: Predikcija ──────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Nema podataka u zahtjevu"}), 400

    time_spent_alone     = float(data.get("time_spent_alone", 0))
    stage_fear           = to_yes_no(data.get("stage_fear", 0))
    social_event_attend  = float(data.get("social_event_attend", 0))
    going_outside        = float(data.get("going_outside", 0))
    drained_after_social = to_yes_no(data.get("drained_after_social", 0))
    friends_circle_size  = float(data.get("friends_circle_size", 0))
    post_frequency       = float(data.get("post_frequency", 0))

    # Azure payload format (identičan ispravnom primjeru)
    azure_payload = {
        "Inputs": {
            "input1": [
                {
                    "Time_spent_Alone":          time_spent_alone,
                    "Stage_fear":                stage_fear,
                    "Social_event_attendance":   social_event_attend,
                    "Going_outside":             going_outside,
                    "Drained_after_socializing": drained_after_social,
                    "Friends_circle_size":       friends_circle_size,
                    "Post_frequency":            post_frequency
                }
            ]
        },
        "GlobalParameters": {}
    }

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_API_KEY}"
        }
        response = requests.post(AZURE_ENDPOINT, json=azure_payload, headers=headers, timeout=20)
        response.raise_for_status()
        azure_result = response.json()

        # Parsiranje Azure odgovora
        key        = list(azure_result["Results"].keys())[0]
        first      = azure_result["Results"][key][0]
        prediction = first["PersonalityPrediction"]
        probability = float(first.get("Probability", 0.0))
        confidence  = probability if prediction == "Introvert" else 1.0 - probability

    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Ne mogu se spojiti na Azure endpoint: {str(e)}"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Azure endpoint nije odgovorio (timeout 20s)."}), 504
    except requests.exceptions.HTTPError as e:
        body = e.response.text[:300] if e.response else ""
        return jsonify({"error": f"Azure HTTP greška {e.response.status_code}: {body}"}), 502
    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Neočekivani format Azure odgovora: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Neočekivana greška: {str(e)}"}), 500

    # Spremi u SQLite
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO predictions (
                timestamp, time_spent_alone, stage_fear, social_event_attend,
                going_outside, drained_after_social, friends_circle_size,
                post_frequency, prediction, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            time_spent_alone, stage_fear, social_event_attend,
            going_outside, drained_after_social, friends_circle_size,
            post_frequency, prediction, confidence
        ))
        conn.commit()
    finally:
        conn.close()

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4)
    })


# ─── RUTA: Plotly grafikoni ────────────────────────────────────────────────────
@app.route("/api/charts", methods=["GET"])
def charts():
    """
    Dohvaća sve zapise iz SQLite baze, pretvara ih u Pandas DataFrame,
    te generira tri Plotly Express grafikona kao JSON za plotly.js.
    """
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
    finally:
        conn.close()

    if not rows:
        return jsonify({"error": "Nema podataka u bazi"}), 404

    # Pandas: sirovi SQLite zapisi → strukturirana tablica
    df = pd.DataFrame([dict(r) for r in rows])

    color_map = {"Introvert": "#1C3A2B", "Extrovert": "#C8872A"}

    # Graf 1: Pie chart — distribucija predikcija
    dist_df = df["prediction"].value_counts().reset_index()
    dist_df.columns = ["Tip", "Broj"]

    fig1 = px.pie(
        dist_df, names="Tip", values="Broj", color="Tip",
        color_discrete_map=color_map, hole=0.45,
        title="Distribucija predikcija"
    )
    fig1.update_traces(
        textposition="inside", textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Broj: %{value}<br>%{percent}<extra></extra>"
    )
    fig1.update_layout(
        font=dict(family="DM Sans, sans-serif", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    # Graf 2: Grouped Bar — prosječne značajke po tipu
    num_cols = ["time_spent_alone", "social_event_attend",
                "going_outside", "friends_circle_size", "post_frequency"]
    label_map = {
        "time_spent_alone":    "Samo/a",
        "social_event_attend": "Soc. eventi",
        "going_outside":       "Izlasci",
        "friends_circle_size": "Prijatelji",
        "post_frequency":      "Objave"
    }

    avg_df = df.groupby("prediction")[num_cols].mean().round(2).reset_index()
    avg_melted = avg_df.melt(id_vars="prediction", var_name="Značajka", value_name="Prosjek")
    avg_melted["Značajka"] = avg_melted["Značajka"].map(label_map)

    fig2 = px.bar(
        avg_melted, x="Značajka", y="Prosjek", color="prediction",
        barmode="group", color_discrete_map=color_map,
        title="Prosječne vrijednosti po tipu",
        labels={"prediction": "Tip", "Prosjek": "Prosjek"}
    )
    fig2.update_layout(
        font=dict(family="DM Sans, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45, b=10, l=10, r=10),
        yaxis=dict(range=[0, 13], gridcolor="rgba(0,0,0,0.06)", title=""),
        xaxis=dict(showgrid=False, title=""),
        legend=dict(title="Tip"), bargap=0.2, bargroupgap=0.08
    )

    # Graf 3: Line chart — zadnjih 30 predikcija
    tl = df.tail(30).copy().reset_index(drop=True)
    tl["Rbr"]  = range(1, len(tl) + 1)
    tl["Vrij"] = tl["prediction"].map({"Introvert": 0, "Extrovert": 1})
    tl["Boja"] = tl["prediction"].map({"Introvert": "#1C3A2B", "Extrovert": "#C8872A"})

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=tl["Rbr"], y=tl["Vrij"],
        mode="lines+markers",
        line=dict(color="#2E5E44", width=2),
        marker=dict(color=tl["Boja"].tolist(), size=9,
                    line=dict(width=1.5, color="white")),
        customdata=tl["prediction"],
        hovertemplate="<b>#%{x}</b> → %{customdata}<extra></extra>",
        fill="tozeroy", fillcolor="rgba(46,94,68,0.07)",
        name="Predikcija"
    ))
    fig3.update_layout(
        title="Zadnje predikcije (kronološki)",
        font=dict(family="DM Sans, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45, b=10, l=10, r=10),
        yaxis=dict(tickvals=[0, 1], ticktext=["Introvert", "Extrovert"],
                   range=[-0.3, 1.3], gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(title="Redni broj", showgrid=False),
        showlegend=False
    )

    return jsonify({
        "chart1":          json.loads(fig1.to_json()),
        "chart2":          json.loads(fig2.to_json()),
        "chart3":          json.loads(fig3.to_json()),
        "total":           len(df),
        "introvert_count": int((df["prediction"] == "Introvert").sum()),
        "extrovert_count": int((df["prediction"] == "Extrovert").sum())
    })


# ─── RUTA: Stats (backward compat s originalnim app.js) ───────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM predictions").fetchall()
    finally:
        conn.close()

    if not rows:
        return jsonify({"distribution": [], "averages": [], "timeline": [], "total": 0})

    df = pd.DataFrame([dict(r) for r in rows])

    dist = df["prediction"].value_counts().reset_index()
    dist.columns = ["prediction", "count"]

    num_cols = ["time_spent_alone", "social_event_attend",
                "going_outside", "friends_circle_size", "post_frequency"]
    avg_df = df.groupby("prediction")[num_cols].mean().round(2).reset_index()
    avg_df.rename(columns={
        "time_spent_alone":    "avg_alone",
        "social_event_attend": "avg_social",
        "going_outside":       "avg_outside",
        "friends_circle_size": "avg_friends",
        "post_frequency":      "avg_posts"
    }, inplace=True)

    tl = df.tail(30)[["timestamp", "prediction", "probability"]].iloc[::-1]

    return jsonify({
        "distribution": dist.to_dict(orient="records"),
        "averages":     avg_df.to_dict(orient="records"),
        "timeline":     tl.to_dict(orient="records"),
        "total":        len(df)
    })


# ─── RUTA: Debug records ───────────────────────────────────────────────────────
@app.route("/api/records", methods=["GET"])
def records():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100").fetchall()
        return jsonify([dict(r) for r in rows])
    finally:
        conn.close()


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PersonaLab — Personality Predictor Backend")
    print("=" * 60)
    print(f"  Azure Endpoint : {AZURE_ENDPOINT[:55]}...")
    print(f"  Baza podataka  : {DB_PATH}")
    print(f"  Server URL     : http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
