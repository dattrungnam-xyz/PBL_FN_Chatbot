FROM python:3.13-slim

ARG PINECONE_API_KEY
ARG TOGETHER_API_KEY


ENV PINECONE_API_KEY=${PINECONE_API_KEY} \
    TOGETHER_API_KEY=${TOGETHER_API_KEY} \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "chatbot_api.py"]