# Docker Guide — Titanic ML API

## Overview

This guide covers building, running, and testing the Titanic ML API in Docker.  
The image uses a **multi-stage build** (`python:3.11-slim`) to keep the runtime image lean.  
The API runs with `uvicorn` on port **8000**.

---

## Build

```bash
docker build -t titanic-ml:latest .
```

Expected output: `Successfully tagged titanic-ml:latest`  
Image size: ~869 MB (includes all ML libraries)

---

## Run

```bash
docker run -d -p 8000:8000 --name titanic-ml titanic-ml:latest
```

Flags:
- `-d` — detached (background)
- `-p 8000:8000` — map host port 8000 to container port 8000
- `--name titanic-ml` — assign a friendly name

---

## Curl Examples

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "model": "SVC", "accuracy": 0.8436}
```

---

### Single Prediction — 1st-class female passenger

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100.0,
    "Embarked": "S"
  }'
```

Expected response:
```json
{"prediction": 1, "label": "1", "probability": 0.8923}
```

---

### Batch Prediction — 2 passengers

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Pclass": 1,
      "Sex": "female",
      "Age": 29,
      "SibSp": 0,
      "Parch": 0,
      "Fare": 100.0,
      "Embarked": "S"
    },
    {
      "Pclass": 3,
      "Sex": "male",
      "Age": 22,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 7.25,
      "Embarked": "S"
    }
  ]'
```

Expected response:
```json
[
  {"prediction": 1, "label": "1", "probability": 0.8923},
  {"prediction": 0, "label": "0", "probability": 0.908}
]
```

---

## Swagger UI

Interactive API docs are available at:

```
http://localhost:8000/docs
```

ReDoc alternative:

```
http://localhost:8000/redoc
```

---

## Stop & Remove

```bash
docker stop titanic-ml && docker rm titanic-ml
```

---

## Post-Deploy Test Commands

Replace `localhost:8000` with your live service URL:

```bash
# Health check
curl https://<your-service-url>/health

# Single predict
curl -X POST https://<your-service-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"}'

# Batch predict
curl -X POST https://<your-service-url>/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"}, {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}]'
```

---

## Useful Docker Commands

| Command | Description |
|---|---|
| `docker logs titanic-ml` | Stream container logs |
| `docker logs -f titanic-ml` | Follow live logs |
| `docker exec -it titanic-ml /bin/bash` | Shell into running container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers (including stopped) |
| `docker images` | List local images |
| `docker rmi titanic-ml:latest` | Remove the image |
| `docker inspect titanic-ml` | Full container metadata |
| `docker stats titanic-ml` | Live CPU/memory usage |

---

## Image Details

| Property | Value |
|---|---|
| Base image | `python:3.11-slim` |
| Build strategy | Multi-stage (builder + runtime) |
| Exposed port | 8000 |
| Runtime user | `appuser` (non-root) |
| Image size | ~869 MB |
| Python version | 3.11 |
| Model artifact | `models/final_pipeline.pkl` |
| Label encoder | `models/label_encoder.pkl` |
