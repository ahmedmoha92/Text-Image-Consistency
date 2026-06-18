FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p "$NLTK_DATA"

COPY notebooks/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk, os; d=os.environ['NLTK_DATA']; [nltk.download(r, quiet=True, download_dir=d) for r in ['punkt','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','punkt_tab','maxent_ne_chunker','words','vader_lexicon']]"

# Copier le code source pour y avoir acces dans les notebooks
COPY src/ ./src/

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
