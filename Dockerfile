FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

    
COPY .python-version pyproject.toml uv.lock ./


RUN uv sync --locked 


COPY src/predict.py ./src/
COPY src/transform.py ./src/
COPY models/model_XGB.bin ./models/



EXPOSE 9696
ENTRYPOINT ["uv", "run", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]