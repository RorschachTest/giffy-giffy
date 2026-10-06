FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg does all the video work; build-essential is needed to compile insightface.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-faces.txt ./
RUN pip install -r requirements.txt

# Build with --build-arg WITH_FACES=0 (and set ENABLE_FACES=0) to skip face recognition.
ARG WITH_FACES=1
RUN if [ "$WITH_FACES" = "1" ]; then pip install -r requirements-faces.txt; fi

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
