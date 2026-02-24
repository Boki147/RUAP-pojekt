# PersonaLab — Personality Predictor

Web aplikacija koja predviđa tip ličnosti (introvert/ekstrovert) na temelju
bihevioralnih podataka iz Kaggle dataseta, koristeći Azure ML endpoint, Flask backend,
**Pandas** za obradu podataka i **Plotly Express** za vizualizacije.

---

## Struktura projekta

```
/
├── index.html      ← Početna stranica (psihološki članak)
├── predict.html    ← Forma za predviđanje + 3 Plotly grafikona
├── style.css       ← Svi CSS stilovi
├── app.js          ← Frontend JS (forma, Plotly rendering)
├── server.py       ← Flask backend (Azure ML + SQLite + Pandas + Plotly)
└── README.md
```

---

## Postavljanje i pokretanje

### 1. Instaliraj Python dependencije

```bash
pip install flask flask-cors requests pandas plotly
```

### 2. Konfiguriraj Azure ML endpoint (opcionalno)

Endpoint i API ključ već su upisani u `server.py`. Ako imaš vlastiti endpoint:

```python
AZURE_ENDPOINT = "azure endpoint"
AZURE_API_KEY  = "API-KEY"
```

### 3. Pokreni Flask backend

```bash
python server.py
```

Server se pokreće na `http://localhost:5000`.


### 4. Otvori frontend

Otvori `index.html` ili `predict.html` u pregledniku (direktno ili putem Live Servera).

---

## Kako funkcionira

### Tok podataka

```
Korisnik popuni formu
       ↓
app.js → POST /api/predict
       ↓
server.py → Azure ML endpoint (REST API)
       ↓
Azure vraća: { "Results": { "WebServiceOutput0": [{ "PersonalityPrediction": "...", "Probability": 0.xx }] } }
       ↓
Flask sprema zapis u SQLite (predictions.db)
       ↓
app.js → GET /api/charts
       ↓
server.py → SQLite → Pandas DataFrame → Plotly Express → JSON
       ↓
app.js → Plotly.react() → 3 interaktivna grafikona
```

### Pandas uloga

Pandas se koristi u `server.py` za:
- **Seed data**: `pd.DataFrame` za validaciju i clip vrijednosti prije unosa
- **Distribucija**: `df["prediction"].value_counts()` → pie chart podaci
- **Prosjeci**: `df.groupby("prediction")[cols].mean()` + `.melt()` → bar chart
- **Timeline**: `df.tail(30)` → line chart

### Plotly Express uloga

Plotly Express generira tri grafikona kao JSON:
1. **`px.pie()`** — distribucija introverata vs ekstroverata (donut)
2. **`px.bar()`** — prosječne vrijednosti 5 značajki po tipu (grouped bar)
3. **`go.Scatter()`** — zadnjih 30 predikcija kronološki (line + fill)

Flask serializira grafikone s `fig.to_json()` → frontend renderira s `Plotly.react()`.

---

## API Endpointi

| Metoda | URL            | Opis                                              |
|--------|----------------|---------------------------------------------------|
| POST   | `/api/predict` | Prima 7 značajki, šalje Azure-u, sprema u SQLite  |
| GET    | `/api/charts`  | Pandas + Plotly → 3 grafikona kao JSON            |
| GET    | `/api/stats`   | Kompatibilnost unazad (isti podaci, drugačiji format) |
| GET    | `/api/records` | Debug — zadnjih 100 zapisa iz baze               |

### POST `/api/predict` — primjer zahtjeva

```json
{
  "time_spent_alone": 7,
  "stage_fear": 1,
  "social_event_attend": 2,
  "going_outside": 1,
  "drained_after_social": 1,
  "friends_circle_size": 3,
  "post_frequency": 1
}
```

### Odgovor

```json
{
  "prediction": "Introvert",
  "confidence": 0.92,
  "probability": 0.92
}
```

### Azure payload format (koji server.py šalje)

```json
{
  "Inputs": {
    "input1": [{
      "Time_spent_Alone": 7.0,
      "Stage_fear": "Yes",
      "Social_event_attendance": 2.0,
      "Going_outside": 1.0,
      "Drained_after_socializing": "Yes",
      "Friends_circle_size": 3.0,
      "Post_frequency": 1.0
    }]
  },
  "GlobalParameters": {}
}
```

### Azure odgovor (što server.py prima)

```json
{
  "Results": {
    "WebServiceOutput0": [{
      "PersonalityPrediction": "Introvert",
      "Probability": 0.92
    }]
  }
}
```

---

## Kaggle dataset

Dataset: [Extrovert vs Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)

| Značajka                    | Tip    | Opis                              |
|-----------------------------|--------|-----------------------------------|
| `Time_spent_Alone`          | Float  | Prosječni sati dnevno sami (0–11) |
| `Stage_fear`                | Yes/No | Strah od javnog nastupa           |
| `Social_event_attendance`   | Float  | Soc. događaji tjedno (0–10)       |
| `Going_outside`             | Float  | Izlasci tjedno (0–7)              |
| `Drained_after_socializing` | Yes/No | Iscrpljenost nakon socijalizacije |
| `Friends_circle_size`       | Float  | Broj bliskih prijatelja (0–15)    |
| `Post_frequency`            | Float  | Objave tjedno (0–10)              |

---

## Baza podataka

SQLite baza `predictions.db` automatski se kreira pri pokretanju.
Sadrži 100 seed zapisa + svaki novi korisnički unos.

| Kolona                | Tip     |
|-----------------------|---------|
| id                    | INTEGER |
| timestamp             | TEXT    |
| time_spent_alone      | REAL    |
| stage_fear            | TEXT    |
| social_event_attend   | REAL    |
| going_outside         | REAL    |
| drained_after_social  | TEXT    |
| friends_circle_size   | REAL    |
| post_frequency        | REAL    |
| prediction            | TEXT    |
| probability           | REAL    |
