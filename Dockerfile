# TODO: M5/M6 で完成 — uv + CUDA base image でクラウド A100 訓練用 image を作る。
# 現状は placeholder (skeleton)。M5 (統合スモークのクラウド移行) 着手時に完成させる。
#
# 想定構成 (cu128 wheel に合わせる):
#   FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
#   COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#   WORKDIR /app
#   COPY pyproject.toml uv.lock .python-version ./
#   RUN uv sync --frozen --no-dev
#   COPY src/ ./src/
#   COPY scripts/ ./scripts/
#   COPY configs/ ./configs/
#   ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#   ENTRYPOINT ["uv", "run", "python", "-m", "wavenext2.train.train_gan"]
