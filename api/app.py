from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import base64
import cv2
import os
import re
import nltk
from nltk.stem import SnowballStemmer
from skimage.feature import hog

# Prétraitement du texte : correspondre exactement au pipeline d'entraînement
TOKEN_PATTERN = re.compile(r'\w+')
stemmer = SnowballStemmer('english')

NLTK_DATA_DIR = os.environ.get('NLTK_DATA', '/usr/local/share/nltk_data')
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Télécharger les ressources NLTK nécessaires au démarrage si elles sont absentes.
for resource in ['punkt', 'wordnet', 'omw-1.4']:
    nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True, halt_on_error=False)

app = Flask(__name__)
CORS(app)

# Déterminer les chemins absolus des modèles
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger les composants du pipeline classique entraîné
try:
    best_model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca_model = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    kmeans_bovw = joblib.load(os.path.join(MODEL_DIR, 'kmeans_bovw.pkl'))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    print("Modèles et composants chargés avec succès !")
except Exception as e:
    best_model = None
    scaler = None
    pca_model = None
    kmeans_bovw = None
    vectorizer = None
    print(f"Erreur lors du chargement des modèles : {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if not all([best_model, scaler, pca_model, kmeans_bovw, vectorizer]):
        return jsonify({'error': 'Modèles non chargés. Veuillez d\'abord entraîner le pipeline.'}), 500

    data = request.get_json()
    text = data.get('text', '')
    image_b64 = data.get('image', '')
    
    if not text or not image_b64:
        return jsonify({'error': 'Texte et image requis'}), 400
    
    # 1. Traitement de l'image (Entièrement en mémoire)
    try:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Impossible de décoder l\'image.'}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur de décodage base64 : {str(e)}'}), 400
    
    # Extraction des caractéristiques image
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # - HOG (1764 features)
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    
    # - Histogramme couleur HSV (24 features)
    color_hist = np.concatenate([
        np.histogram(hsv[:, :, i], bins=8, range=(0, 256), density=True)[0] for i in range(3)
    ])
    
    # - Bag of Visual Words (100 features)
    bovw_feat = np.zeros(100)
    if kmeans_bovw is not None:
        orb = cv2.ORB_create(nfeatures=200)
        _, des = orb.detectAndCompute(gray, None)
        if des is not None:
            words = kmeans_bovw.predict(des)
            bovw_feat, _ = np.histogram(words, bins=np.arange(101), density=True)
            
    X_img = np.concatenate([hog_features, color_hist, bovw_feat])
    
    # 2. Traitement du texte (TF-IDF)
    tokens = TOKEN_PATTERN.findall(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    lemmatized = " ".join([stemmer.stem(t) for t in tokens])

    X_tfidf = vectorizer.transform([lemmatized]).toarray()
    
    # 3. Fusion, normalisation et PCA
    X_raw = np.hstack([X_tfidf, X_img.reshape(1, -1)])
    X_scaled = scaler.transform(X_raw)
    X_pca = pca_model.transform(X_scaled)
    
    # 4. Prédiction
    proba = best_model.predict_proba(X_pca)[0]
    pred = best_model.predict(X_pca)[0]
    
    return jsonify({
        'prediction': 'coherent' if pred == 1 else 'incoherent',
        'confidence': float(max(proba)),
        'probabilities': {'coherent': float(proba[1]), 'incoherent': float(proba[0])}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
