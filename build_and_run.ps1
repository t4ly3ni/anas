# Build and run the Docker image on Windows PowerShell
docker build -t carprice-app .
docker run --rm -p 8501:8501 carprice-app
