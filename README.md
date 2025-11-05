# API Toxic Detection - SVM

API REST pour la détection automatique de commentaires toxiques.

## Architecture

```
API_Digital_Social_Score/
├── app.py                    # API Flask (production)
├── main.py                  # API FastAPI
├── Dockerfile               # Configuration Docker
├── k8s/deployment.yaml      # Configuration Kubernetes
├── model/
│   ├── svm_model.pkl        # Modèle SVM entraîné
│   ├── SVM.py               # Script d'entraînement SVM
│   └── BERT.py              # Script d'entraînement BERT
├── data/                    # Datasets d'entraînement
└── requirements.txt         # Dépendances Python
```

## Technologies

- **Backend**: FastAPI
- **ML Model**: SVM + TF-IDF (scikit-learn)
- **Container**: Docker
- **Orchestration**: Kubernetes
- **Cloud**: Google Cloud Platform (GCP)
- **Dataset**: Jigsaw Toxic Comment Classification

## Installation et Démarrage

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Démarrage local

**FastAPI avec interface web (recommandé)**

```bash
python app.py
# Interface web: http://localhost:8080/docs
```

### 3. Test de l'API

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I hate this stupid product"}'
```

## Déploiement Docker

### Construction de l'image

```bash
docker build -t toxic-detection-api .
docker tag toxic-detection-api gcr.io/PROJECT-ID/toxic-detection-api
```

### Push vers Container Registry

```bash
docker push gcr.io/PROJECT-ID/toxic-detection-api
```

## Déploiement Kubernetes sur GCP

### 1. Création du cluster GKE

```bash
gcloud container clusters create toxic-detection-cluster \
  --zone=us-central1-a \
  --num-nodes=3
```

### 2. Déploiement

```bash
kubectl apply -f k8s/deployment.yaml
```

### 3. Vérification

```bash
kubectl get services
kubectl get pods
```

## Utilisation de l'API

### Endpoints disponibles

#### GET /

Informations générales sur l'API

#### GET /health

Vérification de l'état du service

#### POST /predict

Analyse de toxicité d'un texte

**Corps de la requête :**

```json
{
  "text": "Votre commentaire à analyser"
}
```

**Réponse :**

```json
{
  "prediction": 1,
  "probability": 0.85,
  "label": "toxic",
  "confidence": "high",
  "text_length": 25,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Interface web (FastAPI)

Accédez à `http://localhost:8080/docs` pour une interface interactive Swagger UI permettant de tester l'API directement depuis le navigateur.

## Performance du modèle

- **Algorithme**: Support Vector Machine (SVM)
- **Vectorisation**: TF-IDF
- **Dataset**: Jigsaw Toxic Comment Classification (10k échantillons)
- **Précision**: ~85% sur les données de test
- **Temps de réponse**: ~50ms par prédiction

## Déploiement en production

**URL de production**: http://34.68.240.253:80

- Service déployé sur Google Kubernetes Engine
- 3 répliques avec load balancer
- Health checks configurés

## Développement

### Structure du projet

- `app.py`: API Flask optimisée pour la production
- `fastapi_app.py`: API FastAPI avec interface web pour le développement
- `main.py`: Version FastAPI complète avec validation avancée
- `model/SVM.py`: Script d'entraînement du modèle SVM
- `data/`: Datasets anonymisés conformes RGPD

### Conformité RGPD

Les datasets ont été anonymisés en supprimant:

- Noms de personnes (détection NER avec spaCy)
- Adresses emails (regex)
- Numéros de téléphone (regex)
- URLs personnelles (regex)
