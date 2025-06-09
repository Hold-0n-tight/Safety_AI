# client1/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flwr numpy

CMD ["python", "client.py"]
