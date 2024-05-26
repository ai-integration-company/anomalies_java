
FROM python:3.9-slim


RUN apt-get update && \
    apt-get install -y gcc libfreetype6-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


WORKDIR /app

RUN chmod +x entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]
