FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENV ACTION="predict"

CMD ["sh", "-c", "python ${ACTION}.py"]
