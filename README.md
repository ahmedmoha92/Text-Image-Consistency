# 🖼️ Détection d'Incohérence Texte-Image

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Système de détection d'incohérence sémantique entre une image et sa légende, utilisant le Machine Learning (Random Forest) et une architecture micro-services conteneurisée.

---

## 🛠️ Architecture & Services

Le projet est divisé en trois services principaux orchestrés par Docker :

*   **API Backend (Flask)** : `http://localhost:5009` - Gère les prédictions en temps réel.
*   **Interface Frontend (Web)** : `http://localhost:8081` - Interface utilisateur simple pour uploader une image et tester une légende.
*   **Jupyter Notebook** : `http://localhost:8891` - Environnement de développement pour l'expérimentation ML.

---

## 🚀 Installation & Configuration

### 1. Préparation des données (Exemple MSCOCO)

Suivez ces étapes dans l'ordre pour préparer votre dataset de manière optimale :

```bash
# A. Télécharger les données brutes
python3 scripts/download_datasets.py --dataset mscoco

# B. Préparer le format (Images + Textes individuels)
python3 scripts/prepare_coco.py

# C. Générer les paires (Cohérentes et Incohérentes pour l'entraînement)
python3 scripts/generate_pairs.py --source data/raw/coco_prepared --output data/processed
```

### 2. Entraînement du modèle

Une fois les données générées, lancez l'entraînement du modèle Random Forest :

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/models/train.py
```

### 3. Lancement avec Docker

Déployez l'application complète en une seule commande :

```bash
docker compose build
docker compose up -d
```

---

## 🖥️ Utilisation de l'API

Vous pouvez tester l'API directement via `curl` :

```bash
curl -X POST http://localhost:5009/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Un chat noir sur un tapis rouge",
    "image": "BASE64_ENCODED_IMAGE_STRING"
  }'
```

---

## 📂 Structure du Projet

```text
.
├── api/              # API Flask et logique de service
├── frontend/         # Interface utilisateur (HTML/JS/CSS)
├── src/              # Noyau du projet (extraction de features, modèles)
│   ├── features/     # Logique d'extraction (Image & Texte)
│   └── models/       # Script d'entraînement et stockage
├── scripts/          # Utilitaires de gestion des données
├── notebooks/        # Expérimentations Data Science
├── data/             # Données brutes et traitées (ignoré par Git)
└── docker-compose.yml # Orchestration des services
```

---

## 📄 Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.
