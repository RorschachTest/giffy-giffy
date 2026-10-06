FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg does all the video work. build-essential compiles older insightface
# releases; libgl1 and libglib2.0-0 are needed by the OpenCV build it pulls in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-faces.txt ./
RUN pip install -r requirements.txt

# Build with --build-arg WITH_FACES=0 (and set ENABLE_FACES=0) to skip face recognition.
# Older insightface releases need numpy and cython present while they build,
# which is why those go in first and build isolation is switched off.
ARG WITH_FACES=1
RUN if [ "$WITH_FACES" = "1" ]; then \
      pip install "numpy<2" cython setuptools wheel \
      && pip install --no-build-isolation -r requirements-faces.txt; \
    fi

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
