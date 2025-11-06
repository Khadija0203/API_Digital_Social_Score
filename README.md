# API Toxic Detection

API REST haute performance pour la détection automatique de commentaires toxiques avec monitoring complet Prometheus/Grafana.

## Architecture Production

```
API_Digital_Social_Score/
├── app.py                           # API FastAPI Production + Métriques
├── docker-compose.monitoring.yml    # Stack Prometheus + Grafana
├── prometheus/
│   └── prometheus.yml              # Configuration monitoring
├── grafana/
│   ├── dashboards/                 # Dashboards personnalisés
│   └── datasources/               # Configuration auto Prometheus
├── k8s/deployment.yaml            # Déploiement Kubernetes
├── model/
│   ├── svm_model.pkl              # Modèle SVM entraîné (10k samples)
│   ├── SVM.py                     # Training pipeline SVM
│   └── BERT.py                    # Training pipeline BERT
├── data/                          # Datasets RGPD-anonymized
├── performance_test.py            # Tests de charge automatisés
└── requirements.txt               # Dependencies production
```

## Stack Technologique

### Core API

- **Backend**: FastAPI (production-optimized)
- **ML Pipeline**: SVM + TF-IDF (scikit-learn)
- **Performance**: < 50ms response time
- **Concurrency**: Async/await + thread-safe

### Infrastructure & Monitoring

- **Containerization**: Docker multi-stage builds
- **Orchestration**: Kubernetes (GKE)
- **Monitoring**: Prometheus + Grafana
- **Metrics**: 12+ production metrics
- **Alerting**: Configurable SLA alerts
- **Cloud**: Google Cloud Platform

### Data & Compliance

- **Dataset**: Jigsaw Toxic Comment (RGPD-compliant)
- **Privacy**: NER-based anonymization (spaCy)
- **Security**: Non-root containers + health checks

## Démarrage Rapide

### 1. Setup Environment

```bash
# Clone et setup
git clone <repository>
cd API_Digital_Social_Score

# Virtual environment (recommandé)
python -m venv .env
source .env/bin/activate  # Linux/Mac
.env\Scripts\activate     # Windows

# Installation dependencies
pip install -r requirements.txt
```

### 2. Lancement API + Monitoring

```bash
# Démarrage API FastAPI
python app.py
# API: http://localhost:8080/docs

# Démarrage stack monitoring (nouveau terminal)
docker-compose -f docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

### 3. Tests et Métriques

```bash
# Test de base
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"You are stupid and I hate you"}'

# Health check avancé
curl http://localhost:8080/health

# Métriques Prometheus
curl http://localhost:8080/metrics

# Tests de performance
python performance_test.py
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

## API Endpoints

### Core Endpoints

#### `GET /` - API Info

```json
{
  "message": "API de Détection de Commentaires Toxiques",
  "version": "1.0.0",
  "endpoints": {
    "/docs": "Interface Swagger UI",
    "/health": "Health check",
    "/predict": "Prédiction ML"
  }
}
```

#### `GET /health` - Health Check Avancé

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "checks": {
    "model_loaded": true,
    "model_working": true,
    "memory_ok": true,
    "response_time_ok": true
  },
  "response_time_ms": 15.2,
  "version": "1.0.0"
}
```

#### `POST /predict` - Prédiction ML

**Requête:**

```json
{ "text": "You are stupid and I hate you" }
```

**Réponse:**

```json
{
  "prediction": 1,
  "probability": 0.89,
  "label": "toxic",
  "text_length": 32,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `GET /metrics` - Métriques Prometheus

```prometheus
# HELP ml_predictions_total Nombre total de prédictions ML
ml_predictions_total{result="toxic",confidence_level="high"} 245
# HELP http_request_duration_seconds Durée des requêtes HTTP
http_request_duration_seconds_bucket{method="POST",endpoint="/predict",status="200",le="0.1"} 1023
```

### Interface Web Interactive

- **Swagger UI**: http://localhost:8080/docs
- **Tests en direct**: Interface graphique pour tester tous les endpoints
- **Documentation**: Schémas automatiques des requêtes/réponses

## Monitoring & Observabilité

### Métriques Production (Prometheus)

#### Business Metrics - Surveillance ML

```prometheus
ml_predictions_total{result="toxic|non_toxic", confidence_level="high|medium|low"}
ml_prediction_errors_total{error_type="model_load|prediction_failed|timeout"}
ml_confidence_distribution  # Distribution des scores de confiance
```

#### Performance Metrics - SLA Monitoring

```prometheus
http_request_duration_seconds{method, endpoint, status}  # Latence totale
ml_processing_duration_seconds{model_type}              # Temps ML pur
input_text_length_chars                                 # Distribution taille input
```

#### System Metrics - Résilience

```prometheus
app_memory_usage_bytes           # Consommation mémoire
http_requests_in_progress        # Requêtes simultanées
ml_model_status                  # État modèle (1=OK, 0=Erreur)
app_health_status               # Santé globale système
app_error_rate_percent          # Taux d'erreur 5min
```

### Dashboards Grafana

1. **Business Dashboard**: Prédictions, ratios toxic/non-toxic, confidence trends
2. **Performance Dashboard**: Latences P50/P95/P99, throughput, error rates
3. **System Dashboard**: Mémoire, CPU, requêtes concurrentes
4. **Alerting Dashboard**: SLA violations, model health, system alerts

### Alertes Configurables

```yaml
# Exemples d'alertes Prometheus/Grafana
- High Error Rate: > 5% sur 5min
- Slow Response: P95 > 500ms
- Model Down: ml_model_status == 0
- Memory Leak: Memory growth > 500MB/hour
- High Load: concurrent_requests > 100
```

## Performance & Benchmarks

### Model Performance

- **Algorithm**: Support Vector Machine (LinearSVC) + TF-IDF
- **Training Dataset**: Jigsaw Toxic Comment (10k samples, RGPD-anonymized)
- **Accuracy**: ~85% on test set
- **F1-Score**: 0.83 (toxic), 0.87 (non-toxic)

### API Performance

- **Response Time**: < 50ms P95 (including ML processing)
- **ML Processing**: < 20ms P95 (pure model inference)
- **Throughput**: > 200 RPS (single instance)
- **Concurrent Users**: 100+ simultaneous requests
- **Memory Usage**: ~150MB base + 50MB per 1000 requests

## Déploiement Production

### Production Environment

- **URL**: http://34.68.240.253:80
- **Platform**: Google Kubernetes Engine (GKE)
- **Replicas**: 3 instances + LoadBalancer
- **Health Checks**: Kubernetes liveness/readiness probes
- **Monitoring**: Prometheus scraping + Grafana dashboards

### CI/CD Pipeline

```bash
# 1. Build & Push Container
docker build -t toxic-detection-api .
docker tag toxic-detection-api gcr.io/simplifia-hackathon/toxic-detection-api
docker push gcr.io/simplifia-hackathon/toxic-detection-api

# 2. Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# 3. Verify Deployment
kubectl get pods,services
kubectl logs -f deployment/toxic-detection-api
```

### Production Monitoring Stack

```bash
# Monitoring déployé avec Docker Compose
docker-compose -f docker-compose.monitoring.yml up -d

# Accès interfaces
- API Production: http://34.68.240.253:80
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
```

## Tests & Validation

### Tests Automatisés

```bash
# Tests unitaires des endpoints
python -m pytest tests/

# Tests de performance et charge
python performance_test.py

# Tests de sanité rapides
bash quick_test.sh

# Validation métriques Prometheus
curl http://localhost:8080/metrics | grep ml_predictions_total
```

### Scénarios de Test Performance

1. **Load Testing**: 100 requêtes/seconde pendant 5 minutes
2. **Stress Testing**: Montée progressive jusqu'à 500 RPS
3. **Spike Testing**: Pics soudains de trafic
4. **Endurance Testing**: Charge constante pendant 2 heures
5. **Memory Leak Testing**: Surveillance mémoire long-terme

Voir: `PERFORMANCE_TEST.md` pour les détails complets.

## Développement

### Project Structure Production

```
├── app.py                 # API FastAPI + Monitoring complet
├── model/
│   ├── svm_model.pkl     # Modèle trained (excluded from git)
│   ├── SVM.py            # ML training pipeline
│   └── BERT.py           # Alternative BERT pipeline
├── monitoring/
│   ├── prometheus/       # Config Prometheus
│   ├── grafana/         # Dashboards & datasources
│   └── docker-compose.monitoring.yml
├── tests/
│   ├── test_api.py      # Tests unitaires
│   ├── test_model.py    # Tests ML
│   └── performance_test.py  # Tests charge
└── k8s/                 # Kubernetes manifests
```

### Development Workflow

1. **Local Development**: `python app.py` + monitoring stack
2. **Testing**: Automated tests + performance validation
3. **Docker Build**: Multi-stage optimized containers
4. **K8s Deploy**: Production deployment with monitoring
5. **Monitoring**: Real-time metrics + alerting

### RGPD Compliance & Privacy

**Anonymization Pipeline** (spaCy NER + regex):

- Personal names → `[PERSON]`
- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`
- URLs with personal data → `[URL]`
- Custom PII patterns → configurable removal

**Data Governance**:

- No personal data stored in logs
- Model trained on anonymized dataset only
- GDPR-compliant data processing pipeline
- Right to be forgotten: no user data retention

## Support & Maintenance

### Monitoring & Alerting

- **24/7 Monitoring**: Prometheus + Grafana dashboards
- **SLA Monitoring**: P95 < 500ms, availability > 99.9%
- **Error Tracking**: Detailed error metrics + alerting
- **Capacity Planning**: Resource usage trends + forecasting

### Troubleshooting

```bash
# API Health Check
curl http://localhost:8080/health

# Check Metrics
curl http://localhost:8080/metrics | grep -E "(error|duration|memory)"

# Kubernetes Logs
kubectl logs -f deployment/toxic-detection-api

# Prometheus Targets
curl http://localhost:9090/api/v1/targets
```

### Performance Optimization

- Async FastAPI for high concurrency
- ML model caching and optimization
- Connection pooling and resource limits
- Horizontal Pod Autoscaling (HPA) configured
