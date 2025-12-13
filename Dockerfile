FROM python:3.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
      awscli \
      xvfb \
      xauth \
      libgl1 \
      libgl1-mesa-dri \
      libglu1-mesa \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# gym==0.21.0 has packaging metadata that breaks with newer build tooling.
RUN python -m pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1" \
    && pip install --no-build-isolation --no-use-pep517 -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/scripts/run_job.sh && mkdir -p /outputs

ENTRYPOINT ["/app/scripts/run_job.sh"]
CMD ["python3", "-m", "duckrace.lmpc.duckietown_compare", "--iterations", "15", "--quantum", "--diagnostics", "--plot", "--no-augment"]
