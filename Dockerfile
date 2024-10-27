FROM python:3.11.10-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python model.py


FROM python:3.11.10-slim-bookworm

WORKDIR /app

COPY --from=0 /app/model.pkl /app/model.pkl
COPY requirements-build.txt app.py /app
RUN pip install -r requirements-build.txt  --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

