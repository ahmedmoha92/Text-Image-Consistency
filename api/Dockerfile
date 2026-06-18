FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p "$NLTK_DATA"

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/app.py .
# Note: Ces fichiers doivent etre presents ou montes via volume si non entraines
# COPY api/model.pkl .
# COPY api/vectorizer.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
