import nltk
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

NLTK_DATA_DIR = os.environ.get('NLTK_DATA', '/usr/local/share/nltk_data')
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

for resource in ['punkt', 'stopwords']:
    nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True, halt_on_error=False)

def extract_text_features(texts, max_features=5000, vectorizer=None):
    try:
        stop_words = list(stopwords.words('english'))
    except LookupError:
        stop_words = None
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    extra_features = []
    for doc in texts:
        words = nltk.word_tokenize(doc.lower())
        punct_ratio = sum(1 for c in doc if c in '!?.,;') / max(len(doc), 1)
        upper_count = sum(1 for w in words if w.isupper())
        extra_features.append([len(words), punct_ratio, upper_count])
    
    return tfidf_matrix, np.array(extra_features), vectorizer
