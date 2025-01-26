FROM python:3.12 as builder

WORKDIR /app
RUN pip install poetry
RUN poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY . .
RUN poetry install

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY --from=builder /app .

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
