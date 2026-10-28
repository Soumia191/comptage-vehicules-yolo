from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI() # l’application
class Reading(BaseModel):
    temperature_c: float = Field(ge=-40, le=150) # le schéma
    vibration_mm_s: float = Field(ge=0, le=100)

@app.get("/health") # la route GET /health
def health():
    return {"status": "ok"}

@app.post("/v1/predict") # la route POST /v1/predict
def predict(reading: Reading):
    return {"alert": reading.temperature_c > 80} # reading : déjà validé
