from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from ml_model import Model
from models import PredictableRecord, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    Model()
    yield
    Model.instance.model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_handler():
    return {"status": "ok"}


@app.post("/predict")
def predict_handler(
    data: PredictableRecord, model: Model = Depends(Model)
) -> PredictionResponse:
    return {
        "earnings_more_than_50k": model.predict(data),
    }


@app.post("/train")
def train_handler():
    Model.retrain_model()
    return {"message": "Training complete"}
