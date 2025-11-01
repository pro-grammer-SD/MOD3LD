FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080 8501
CMD uvicorn app:app --host 0.0.0.0 --port 8080 & streamlit run ui.py --server.port=8501 --server.address=0.0.0.0
