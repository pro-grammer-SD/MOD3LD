FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080 8501
CMD bash -c "uvicorn app:app --host 0.0.0.0 --port 8080 & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"