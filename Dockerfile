FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app
COPY best_model.joblib /app/best_model.joblib

EXPOSE 5001
CMD ["python", "app.py"]
