# Titanic ML — Render Deployment Guide

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| FastAPI | 0.136.3 |
| uvicorn | 0.41.0 |
| scikit-learn | 1.8.0 |

**Model artifacts** (`models/final_pipeline.pkl`, `models/label_encoder.pkl`) are **not** stored in the repository. They are generated fresh on every Render build by running:
```
python src/preprocess.py && python src/train.py
```
The training dataset (`data/Titanic-Dataset.csv`) **is** committed to the repository so the build can run without any external data source.

---

## 2. Render Setup (5 steps)

1. Go to [https://dashboard.render.com](https://dashboard.render.com) and click **New → Web Service**.
2. Choose **Connect a GitHub repository** and select `ramleo/Titanic`.
3. Render will auto-detect `render.yaml` in the repo root and pre-fill the service settings:
   - **Name:** `titanic-ml`
   - **Build Command:** `pip install -r requirements.txt && python src/preprocess.py && python src/train.py`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.11
4. Select the **Free** tier (or paid for always-on).
5. Click **Create Web Service**. Render will build and deploy automatically.

Live URL: `https://titanic-ml.onrender.com`

> **Note on build time:** The build runs GridSearchCV across 4 models with 5-fold CV. Expect 3–6 minutes for the first deploy on Render's free tier CPU.

---

## 3. API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns model status and accuracy |
| POST | `/predict` | Predict survival for a single passenger |
| POST | `/predict/batch` | Predict survival for multiple passengers |
| GET | `/docs` | Interactive Swagger UI |

### Example Payloads

**POST /predict**
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S",
  "Name": "Mr. Owen Harris"
}
```

**POST /predict/batch**
```json
[
  {
    "Pclass": 1,
    "Sex": "female",
    "Age": 38,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.28,
    "Embarked": "C",
    "Name": "Mrs. John Bradley"
  },
  {
    "Pclass": 3,
    "Sex": "male",
    "Age": 26,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.90,
    "Embarked": "S",
    "Name": "Mr. James Moran"
  }
]
```

---

## 4. Test-It-Live curl Commands

Replace `https://titanic-ml.onrender.com` with `http://localhost:8000` for local testing.

### Health check
```bash
curl https://titanic-ml.onrender.com/health
```
Expected response:
```json
{"status": "ok", "model": "SVC", "accuracy": 0.8436}
```

### Single prediction
```bash
curl -X POST https://titanic-ml.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Name": "Mr. Owen Harris"
  }'
```
Expected response:
```json
{"prediction": 0, "label": "0", "probability": 0.9123}
```

### Batch prediction
```bash
curl -X POST https://titanic-ml.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C", "Name": "Mrs. John Bradley"},
    {"Pclass": 3, "Sex": "male", "Age": 26, "SibSp": 0, "Parch": 0, "Fare": 7.90, "Embarked": "S", "Name": "Mr. James Moran"}
  ]'
```

### Interactive Swagger UI
Open in your browser:
```
https://titanic-ml.onrender.com/docs
```

---

## 5. Run Locally

```bash
# Clone the repo
git clone https://github.com/ramleo/Titanic.git
cd Titanic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (generates models/*.pkl)
python src/preprocess.py
python src/train.py

# Start the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`

---

## 6. Input Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `Pclass` | int | Yes | Passenger class (1 = First, 2 = Second, 3 = Third) |
| `Sex` | str | Yes | `"male"` or `"female"` |
| `SibSp` | int | Yes | Number of siblings / spouses aboard |
| `Parch` | int | Yes | Number of parents / children aboard |
| `Age` | float | No | Age in years (imputed with median if omitted) |
| `Fare` | float | No | Ticket fare (imputed with median if omitted) |
| `Embarked` | str | No | Port of embarkation: `"S"`, `"C"`, or `"Q"` |
| `Name` | str | No | Full passenger name — used to extract title (Mr, Mrs, Miss, Master, Rare) |
| `PassengerId` | int | No | Ignored (pass-through only) |
| `Ticket` | str | No | Ignored (pass-through only) |
| `Cabin` | str | No | Ignored (pass-through only) |

---

## 7. Free Tier Cold-Start Note

Render's free tier spins down the service after 15 minutes of inactivity. The **first request** after a period of inactivity may take **30–60 seconds** to respond while the container restarts. Subsequent requests will be fast. Upgrade to a paid plan for always-on availability.

---

## 8. Build Process Note

The model is trained from scratch on every Render deploy using this sequence:

```
pip install -r requirements.txt
  ↓
python src/preprocess.py   # loads data/Titanic-Dataset.csv, engineers features,
                           # builds ColumnTransformer, saves models/*.pkl and *.npy
  ↓
python src/train.py        # runs GridSearchCV across 4 models (LR, RF, SVC, GBM),
                           # selects best by 5-fold CV accuracy, saves models/final_pipeline.pkl
  ↓
uvicorn app:app ...        # starts FastAPI, loads final_pipeline.pkl + label_encoder.pkl
```

This ensures full reproducibility — no pre-built binary artifacts are required in the repository.
