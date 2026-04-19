import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from app.FakeNewsDetector import FakeNewsDetector
from app.scraper import async_scrape_article

logger = logging.getLogger(__name__)
class TextInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=500, example="Breaking news: scientists discover...")
    text: str  = Field(..., min_length=10, example="Researchers at MIT announced today...")


class UrlInput(BaseModel):
    url: HttpUrl = Field(..., example="https://www.bbc.com/news/article-xyz")


class PredictionResponse(BaseModel):
    label: str
    confidence: float | None
    input_preview: str
    latency_seconds: float


class AppState:
    def __init__(self):
        self.detector: FakeNewsDetector | None = None
        self.prediction_count: int = 0
        self.error_count: int = 0
        self.start_time: float = time.time()

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 1)


state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Chargement du modèle…")
    state.detector = FakeNewsDetector(
        model_path="models/b_model.pkl",
        vectorizer_path="models/vectorizer.pkl",
    )
    logger.info("[STARTUP] Modèle prêt → %s", state.detector)
    yield
    logger.info("[SHUTDOWN] API arrêtée proprement.")
    
app = FastAPI(
    title="Fake News Detection API",
    description=(
        "API de détection de fausses nouvelles basée sur un modèle NLP "
        "(LinearSVC + TF-IDF). Supporte texte brut et URL d'article."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Journalise chaque requête avec méthode, path et durée."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = round(time.perf_counter() - start, 4)
    logger.info("[HTTP] %s %s → %d (%.4fs)", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/", tags=["Health"])
async def root():
    """Vérification rapide que l'API est en ligne."""
    return {"status": "ok", "message": "Fake News Detection API v2 is running"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: TextInput):
    start = time.perf_counter()
    content = f"{data.title} {data.text}"

    try:
        result = state.detector.predict(content)
        state.prediction_count += 1
    except Exception as exc:
        state.error_count += 1
        logger.error("[PREDICT] Erreur : %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return {
        **result,
        "latency_seconds": round(time.perf_counter() - start, 4),
    }


@app.post("/predict-url", response_model=dict, tags=["Prediction"])
async def predict_from_url(data: UrlInput):
    url_str = str(data.url)

    try:
        article = await async_scrape_article(url_str)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    content = f"{article['title']} {article['text']}"

    try:
        result = state.detector.predict(content)
        state.prediction_count += 1
    except Exception as exc:
        state.error_count += 1
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return {
        **result,
        "source_url": url_str,
        "scraped_title": article["title"],
    }


@app.get("/metrics", tags=["MLOps"])
async def metrics():
    return {
        "total_predictions": state.prediction_count,
        "total_errors": state.error_count,
        "error_rate": (
            round(state.error_count / state.prediction_count, 4)
            if state.prediction_count > 0 else 0.0
        ),
        "uptime_seconds": state.uptime_seconds,
        "model": repr(state.detector),
        "status": "running",
    }