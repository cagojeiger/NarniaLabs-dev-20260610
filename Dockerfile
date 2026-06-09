# ML 학습 워크플로우 PoC 재현용 이미지 (로컬 의존성 0).
#
#   docker build -t narnia-wf .
#   docker run --rm narnia-wf                      # 전체 파이프라인 (dvc repro)
#   docker run --rm narnia-wf dvc dag              # DAG 시각화
#   docker run --rm narnia-wf python -m mljob.train --fault nan   # 실패 시나리오
#
# Linux 이미지라 uv.lock 의 CPU 전용 torch(2.x+cpu)가 설치되어 가볍다.

FROM python:3.12-slim

# uv (정적 바이너리) + git (DVC 의 SCM 컨텍스트용)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# 1) 의존성 레이어 먼저 (소스 변경 시 재설치 안 하도록 캐시 최적화)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# 2) 소스 + 파이프라인 정의
COPY mljob/ ./mljob/
COPY dvc.yaml dvc.lock ./
COPY .dvc/ ./.dvc/

# 3) 프로젝트 설치
RUN uv sync --frozen

# DVC 가 SCM 루트를 찾도록 최소 git 저장소 초기화 (커밋은 하지 않아 이미지 비대화 방지)
RUN git init -q && git config user.email ci@narnia.local && git config user.name ci

ENTRYPOINT ["uv", "run", "--frozen"]
CMD ["dvc", "repro"]
