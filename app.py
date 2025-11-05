import pickle
import logging
from datetime import datetime
from typing import List
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        return True
        
    except Exception as e:
        logger.error(f" Erreur lors du chargement du modèle: {str(e)}")
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
    
    try:
        # Prédiction
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
        
        label = "toxic" if prediction == 1 else "non_toxic"
        
        logger.info(f"Résultat: {label} (prob: {probability_toxic:.3f})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_toxic,
            label=label,
            text_length=len(text),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

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