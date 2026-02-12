Build & Run (Docker)
---------------------

1. Build the Docker image:

```bash
docker build -t carprice-app .
```

2. Run the container and expose Streamlit on port 8501:

```bash
docker run --rm -p 8501:8501 carprice-app
```

Notes:
- The Dockerfile runs `detection_car_price/main.py` by default. If you want to run the MLflow variant, change the CMD to run `detection_car_price/main_mlflow.py`.
- Make sure `models/` and `artifacts/` are present in the repository before building the image.
