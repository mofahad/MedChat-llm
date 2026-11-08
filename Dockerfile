FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app
COPY requirements.txt requirements-hosting.txt ./
RUN pip install --no-cache-dir -r requirements-hosting.txt
COPY --chown=app:app app.py ./
COPY --chown=app:app data ./data
COPY --chown=app:app demo ./demo
USER app
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "app:create_app()"]
