import joblib
import numpy as np
from src.features.text_features import extract_text_features
from src.features.image_features import extract_image_features
from src.features.fusion import fuse_features

class MultimodalPredictor:
    def __init__(self, model_path='api/model.pkl', vectorizer_path='api/vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    
    def predict(self, text, image_path):
        tfidf, extra, _ = extract_text_features([text], vectorizer=self.vectorizer)
        img_vec = extract_image_features(image_path)
        fused = fuse_features(tfidf[0], extra[0], img_vec)
        proba = self.model.predict_proba([fused])[0]
        pred = self.model.predict([fused])[0]
        return {'prediction': 'coherent' if pred == 1 else 'incoherent',
                'confidence': float(max(proba)),
                'probabilities': {'coherent': float(proba[1]), 'incoherent': float(proba[0])}}
