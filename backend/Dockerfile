FROM python:3.10-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code/app \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY app/pyproject.toml app/poetry.lock* /code/
COPY app/app/ app/alembic/ /code/app/

FROM base AS builder
ARG INSTALL_DEV=false
RUN poetry install --no-root && \
    poetry run alembic --help

FROM base AS final
ENV PATH="/code/.venv/bin:$PATH"

COPY --from=builder /code /code

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9090/health || exit 1

WORKDIR /code

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9090"]
