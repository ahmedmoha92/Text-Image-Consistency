"""
Script d'entraînement du pipeline complet.

Produit les artefacts suivants dans api/ :
  - best_model.pkl        (RandomForest entraîné)
  - scaler.pkl            (StandardScaler)
  - pca.pkl               (PCA)
  - kmeans_bovw.pkl       (KMeans BoVW codebook)
  - tfidf_vectorizer.pkl  (TfidfVectorizer)

Le pipeline de features correspond EXACTEMENT à celui de api/app.py.
"""

import joblib
import numpy as np
import os
import re
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------------------------
# Constantes (doivent correspondre exactement à api/app.py)
# ---------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r'\w+')
stemmer = SnowballStemmer('english')
IMAGE_SIZE = (128, 128)
N_BOVW_CLUSTERS = 100
ORB_N_FEATURES = 200


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def load_data(data_dir):
    """Lit les paires image/texte depuis les sous-dossiers coherent/ et incoherent/."""
    texts, image_paths, labels = [], [], []
    for label, category in enumerate(['incoherent', 'coherent']):
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith('.txt'):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                img_name = fname.replace('.txt', '.jpg')
                img_path = os.path.join(path, img_name)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(label)
                else:
                    texts.pop()  # Retirer le texte sans image correspondante
    return texts, image_paths, np.array(labels)


# ---------------------------------------------------------------------------
# Extraction de features (identique à api/app.py)
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Tokenise, filtre et stemmise un texte — même logique que app.py L89-L91."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    return " ".join([stemmer.stem(t) for t in tokens])


def extract_orb_descriptors(gray):
    """Extrait les descripteurs ORB d'une image en niveaux de gris."""
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    _, des = orb.detectAndCompute(gray, None)
    return des  # peut être None


def extract_image_features(image_path, kmeans_bovw=None):
    """Extrait les features image : HOG + HSV histogram + BoVW.

    Correspond exactement au pipeline dans api/app.py (lignes 64-86).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HOG
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

    # Histogramme couleur HSV (8 bins × 3 canaux = 24 features)
    color_hist = np.concatenate([
        np.histogram(hsv[:, :, i], bins=8, range=(0, 256), density=True)[0]
        for i in range(3)
    ])

    # Bag of Visual Words (100 features)
    bovw_feat = np.zeros(N_BOVW_CLUSTERS)
    if kmeans_bovw is not None:
        des = extract_orb_descriptors(gray)
        if des is not None:
            words = kmeans_bovw.predict(des)
            bovw_feat, _ = np.histogram(words, bins=np.arange(N_BOVW_CLUSTERS + 1), density=True)

    return np.concatenate([hog_features, color_hist, bovw_feat])


# ---------------------------------------------------------------------------
# Construction du codebook BoVW
# ---------------------------------------------------------------------------
def build_bovw_codebook(image_paths, n_clusters=N_BOVW_CLUSTERS):
    """Construit le codebook KMeans pour le BoVW à partir des images d'entraînement."""
    print("Construction du codebook BoVW...")
    all_descriptors = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        des = extract_orb_descriptors(gray)
        if des is not None:
            all_descriptors.append(des)
        if (i + 1) % 500 == 0:
            print(f"  Descripteurs extraits : {i + 1}/{len(image_paths)}")

    if not all_descriptors:
        print("  ⚠ Aucun descripteur ORB trouvé. Le BoVW sera vide.")
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    print(f"  Total descripteurs : {all_descriptors.shape[0]}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)
    print("  Codebook construit ✓")
    return kmeans


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/validation'

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Les répertoires de données n'existent pas. Veuillez générer les paires d'abord.")
        print("  python scripts/download_datasets.py --dataset mscoco")
        print("  python scripts/prepare_coco.py")
        print("  python scripts/generate_pairs.py --source data/raw/coco_prepared --output data/processed")
        return

    # --- Charger les données ---
    print("=== Chargement des données ===")
    texts_train, images_train, y_train = load_data(train_dir)
    texts_val, images_val, y_val = load_data(val_dir)

    if not texts_train or not texts_val:
        print("Pas de données trouvées dans les répertoires.")
        return

    print(f"  Train : {len(texts_train)} paires  |  Val : {len(texts_val)} paires")

    # --- 1. Features texte (TF-IDF avec stemming) ---
    print("\n=== Extraction des features texte ===")
    train_preprocessed = [preprocess_text(t) for t in texts_train]
    val_preprocessed = [preprocess_text(t) for t in texts_val]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(train_preprocessed).toarray()
    X_val_tfidf = vectorizer.transform(val_preprocessed).toarray()
    print(f"  TF-IDF dimension : {X_train_tfidf.shape[1]}")

    # --- 2. Codebook BoVW (construit sur les images d'entraînement) ---
    print("\n=== Construction du codebook BoVW ===")
    kmeans_bovw = build_bovw_codebook(images_train)

    # --- 3. Features image (HOG + HSV histogram + BoVW) ---
    print("\n=== Extraction des features image ===")
    X_train_img = []
    for i, path in enumerate(images_train):
        X_train_img.append(extract_image_features(path, kmeans_bovw))
        if (i + 1) % 500 == 0:
            print(f"  Train : {i + 1}/{len(images_train)}")
    X_train_img = np.array(X_train_img)

    X_val_img = []
    for i, path in enumerate(images_val):
        X_val_img.append(extract_image_features(path, kmeans_bovw))
        if (i + 1) % 500 == 0:
            print(f"  Val : {i + 1}/{len(images_val)}")
    X_val_img = np.array(X_val_img)
    print(f"  Image feature dimension : {X_train_img.shape[1]}")

    # --- 4. Fusion ---
    X_train_raw = np.hstack([X_train_tfidf, X_train_img])
    X_val_raw = np.hstack([X_val_tfidf, X_val_img])
    print(f"\n  Feature totale (avant PCA) : {X_train_raw.shape[1]}")

    # --- 5. Normalisation ---
    print("\n=== Normalisation (StandardScaler) ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # --- 6. PCA ---
    print("=== Réduction de dimension (PCA 0.95 variance) ===")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    print(f"  Composantes PCA retenues : {pca.n_components_}")

    # --- 7. GridSearch Random Forest ---
    print("\n=== Entraînement (GridSearchCV) ===")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train_pca, y_train)

    print(f"\n  Meilleurs paramètres : {grid.best_params_}")
    print(f"  F1 score CV moyen   : {grid.best_score_:.4f}")

    # --- 8. Évaluation sur la validation ---
    print("\n=== Évaluation sur la validation ===")
    y_val_pred = grid.best_estimator_.predict(X_val_pca)
    print(classification_report(y_val, y_val_pred, target_names=['incoherent', 'coherent']))
    print(f"  F1 validation : {f1_score(y_val, y_val_pred):.4f}")

    # --- 9. Sauvegarde des artefacts ---
    print("\n=== Sauvegarde des artefacts dans api/ ===")
    os.makedirs('api', exist_ok=True)
    artifacts = {
        'api/best_model.pkl': grid.best_estimator_,
        'api/scaler.pkl': scaler,
        'api/pca.pkl': pca,
        'api/kmeans_bovw.pkl': kmeans_bovw,
        'api/tfidf_vectorizer.pkl': vectorizer,
    }
    for path, obj in artifacts.items():
        joblib.dump(obj, path)
        print(f"  ✓ {path}")

    print("\n✅ Entraînement terminé. Les artefacts sont prêts pour api/app.py.")


if __name__ == '__main__':
    main()
