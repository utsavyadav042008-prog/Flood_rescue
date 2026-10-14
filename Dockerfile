# FloodResQ dashboard + pipeline in one container.
#   docker build -t floodresq .                         (CPU, no PyTorch: non-AI fallback model)
#   docker build -t floodresq --build-arg WITH_ML=1 .   (adds CPU PyTorch for the U-Net)
#   docker run -p 8501:8501 --env-file .env -v floodresq-data:/app/data -v floodresq-out:/app/outputs floodresq
# ESA SNAP is NOT included (the cdse_snap backend); the container uses the Planetary Computer backend.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt

ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
      pip install segmentation-models-pytorch albumentations ; \
    fi

COPY . .
# A contract-valid synthetic run so "Load cached demo" works on first start.
RUN python scripts/make_mock_outputs.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
