#!/usr/bin/env bash
set -euo pipefail

# Build and run the Docker image locally
docker build -t carprice-app .
docker run --rm -p 8501:8501 carprice-app
