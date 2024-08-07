FROM python:3.11.4-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas scikit-learn 

CMD ["python", "train_spam_detector.py"]