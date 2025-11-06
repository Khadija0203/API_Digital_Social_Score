import pickle
import logging
from datetime import datetime
from typing import List
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === MÉTRIQUES PROMETHEUS POUR PRODUCTION ===
import prometheus_client

# Vider complètement le registre au démarrage pour éviter les conflits
prometheus_client.REGISTRY._collector_to_names.clear()
prometheus_client.REGISTRY._names_to_collectors.clear()

# Surveillance du modèle ML
PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Nombre total de prédictions ML',
    ['result', 'confidence_level']  # toxic/non_toxic, high/medium/low
)

PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total',
    'Nombre d\'erreurs de prédiction',
    ['error_type']  # model_load, prediction_failed, timeout
)

# Latence et throughput
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'Durée des requêtes HTTP',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")]
)

ML_PROCESSING_TIME = Histogram(
    'ml_processing_duration_seconds',
    'Temps de traitement ML pur',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")]
)

# État de l'application
MODEL_STATUS = Gauge(
    'ml_model_status',
    'État du modèle ML (1=chargé, 0=erreur)'
)

MEMORY_USAGE = Gauge(
    'app_memory_usage_bytes',
    'Utilisation mémoire de l\'application'
)

CONCURRENT_REQUESTS = Gauge(
    'http_requests_in_progress',
    'Nombre de requêtes en cours de traitement'
)

# Distribution des données
TEXT_LENGTH_DISTRIBUTION = Histogram(
    'input_text_length_chars',
    'Distribution de la longueur des textes',
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")]
)

CONFIDENCE_DISTRIBUTION = Histogram(
    'ml_confidence_distribution',
    'Distribution des scores de confiance',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# Surveillance proactive
HEALTH_STATUS = Gauge(
    'app_health_status',
    'État de santé global (1=healthy, 0=unhealthy)'
)

ERROR_RATE = Gauge(
    'app_error_rate_percent',
    'Taux d\'erreur sur les 5 dernières minutes'
)

# Modèles Pydantic
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    text_length: int
    timestamp: str

# Application FastAPI
app = FastAPI(
    title=" API Détection Toxique",
    description="API pour détecter les commentaires toxiques",
    version="1.0.0"
)

# Configuration Prometheus Monitoring
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/docs", "/redoc", "/openapi.json", "/favicon.ico"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Initialisation du monitoring (s'active automatiquement)
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")

# Variable globale pour le modèle
model = None

def load_model():
    """Charger le modèle SVM"""
    global model
    try:
        model_path = './model/svm_model.pkl'
        logger.info(f"Chargement du modèle depuis: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(" Modèle SVM chargé avec succès")
        MODEL_STATUS.set(1)  # Métrique Prometheus
        return True
        
    except Exception as e:
        logger.error(f" Erreur lors du chargement du modèle: {str(e)}")
        MODEL_STATUS.set(0)  # Métrique Prometheus
        PREDICTION_ERRORS.labels(error_type="model_load").inc()
        return False

# Charger le modèle au démarrage
model_loaded = load_model()

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API de Détection de Commentaires Toxiques",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Interface Swagger UI",
            "/health": "Health check",
            "/predict": "Prédiction de toxicité"
        },
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(input_data: TextInput):
    """Prédiction de toxicité"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    text = input_data.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
    
    # === MÉTRIQUES PRODUCTION ===
    request_start = time.time()
    CONCURRENT_REQUESTS.inc()  # Requête en cours
    TEXT_LENGTH_DISTRIBUTION.observe(len(text))
    
    try:
        # Prédiction ML
        ml_start = time.time()
        logger.info(f"Prédiction pour: {text[:50]}...")
        
        prediction = int(model.predict([text])[0])
        
        # Calcul de probabilité
        try:
            # Essayer predict_proba d'abord
            probabilities = model.predict_proba([text])[0]
            probability_toxic = float(probabilities[1])
        except AttributeError:
            # Utiliser decision_function pour LinearSVC
            try:
                from scipy.special import expit
                decision_score = model.decision_function([text])[0]
                probability_toxic = float(expit(decision_score))
            except:
                # Fallback simple
                probability_toxic = 1.0 if prediction == 1 else 0.0
        
        # Temps ML pur
        ml_duration = time.time() - ml_start
        ML_PROCESSING_TIME.labels(model_type="svm").observe(ml_duration)
        
        label = "toxic" if prediction == 1 else "non_toxic"
        
        # Déterminer le niveau de confiance
        if probability_toxic > 0.8 or probability_toxic < 0.2:
            confidence_level = "high"
        elif probability_toxic > 0.6 or probability_toxic < 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # === MÉTRIQUES FINALES ===
        total_duration = time.time() - request_start
        
        # Métriques de succès
        PREDICTIONS_TOTAL.labels(result=label, confidence_level=confidence_level).inc()
        CONFIDENCE_DISTRIBUTION.observe(probability_toxic)
        REQUEST_DURATION.labels(method="POST", endpoint="/predict", status="200").observe(total_duration)
        HEALTH_STATUS.set(1)  # Système opérationnel
        
        logger.info(f"✅ Résultat: {label} (prob: {probability_toxic:.3f}, conf: {confidence_level})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_toxic,
            label=label,
            text_length=len(text),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        # Métriques d'erreur
        PREDICTION_ERRORS.labels(error_type="prediction_failed").inc()
        REQUEST_DURATION.labels(method="POST", endpoint="/predict", status="500").observe(time.time() - request_start)
        
        logger.error(f"❌ Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    finally:
        CONCURRENT_REQUESTS.dec()  # Requête terminée

# === MONITORING SYSTÈME ===
def update_system_metrics():
    """Met à jour les métriques système"""
    import psutil
    
    try:
        # Mémoire de l'application
        process = psutil.Process()
        MEMORY_USAGE.set(process.memory_info().rss)
        
        # État global de santé
        if model_loaded:
            HEALTH_STATUS.set(1)
        else:
            HEALTH_STATUS.set(0)
            
    except Exception as e:
        logger.warning(f"Erreur mise à jour métriques système: {e}")

@app.get("/health")
async def health_check():
    """Health check avancé avec métriques"""
    start_time = time.time()
    
    try:
        # Vérifications de santé
        checks = {
            "model_loaded": model_loaded,
            "memory_ok": True,
            "response_time_ok": True
        }
        
        # Test du modèle
        if model_loaded:
            try:
                test_prediction = model.predict(["test"])[0]
                checks["model_working"] = True
            except:
                checks["model_working"] = False
        else:
            checks["model_working"] = False
        
        # Mise à jour métriques système
        update_system_metrics()
        
        # Temps de réponse
        response_time = time.time() - start_time
        REQUEST_DURATION.labels(method="GET", endpoint="/health", status="200").observe(response_time)
        
        all_healthy = all(checks.values())
        status_code = 200 if all_healthy else 503
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "response_time_ms": round(response_time * 1000, 2),
            "version": "1.0.0"
        }
        
    except Exception as e:
        REQUEST_DURATION.labels(method="GET", endpoint="/health", status="500").observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Endpoint pour les métriques Prometheus"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 8080))
    
    print(f" Démarrage de l'API FastAPI sur le port {port}")
    print(f" Interface web: http://localhost:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )