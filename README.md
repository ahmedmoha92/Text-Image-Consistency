# 🖼️ Text-Image Inconsistency Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** This is an academic project developed for a Machine Learning module.

A semantic inconsistency detection system between an image and its caption, using Machine Learning (Random Forest) and a containerized microservices architecture.

---

## 🛠️ Architecture & Services

The project is divided into three main services orchestrated by Docker:

*   **Backend API (Flask)**: `http://localhost:5009` - Handles real-time predictions.
*   **Frontend Interface (Web)**: `http://localhost:8081` - Simple user interface to upload an image and test a caption.
*   **Jupyter Notebook**: `http://localhost:8891` - Development environment for ML experimentation.

---

## 🚀 Installation & Configuration

### 1. Data Preparation (MSCOCO Example)

Follow these steps in order to optimally prepare your dataset:

```bash
# A. Download raw data
python3 scripts/download_datasets.py --dataset mscoco

# B. Prepare format (Images + Individual texts)
python3 scripts/prepare_coco.py

# C. Generate pairs (Coherent and Incoherent for training)
python3 scripts/generate_pairs.py --source data/raw/coco_prepared --output data/processed
```

### 2. Model Training

Once the data is generated, launch the training of the Random Forest model:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/models/train.py
```

### 3. Launch with Docker

Deploy the complete application with a single command:

```bash
docker compose build
docker compose up -d
```

---

## 🖥️ API Usage

You can test the API directly via `curl`:

```bash
curl -X POST http://localhost:5009/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A black cat on a red carpet",
    "image": "BASE64_ENCODED_IMAGE_STRING"
  }'
```

---

## 📂 Project Structure

```text
.
├── api/              # Flask API and service logic
├── frontend/         # User interface (HTML/JS/CSS)
├── src/              # Project core (feature extraction, models)
│   ├── features/     # Extraction logic (Image & Text)
│   └── models/       # Training script and storage
├── scripts/          # Data management utilities
├── notebooks/        # Data Science experiments
├── data/             # Raw and processed data (ignored by Git)
└── docker-compose.yml # Services orchestration
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
