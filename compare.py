from fastapi import APIRouter
from pydantic import BaseModel
import torch
from models import model

router = APIRouter()

class CompareInput(BaseModel):
    text_a: str
    text_b: str

@router.post("/")
def compare_texts(payload: CompareInput):
    with torch.no_grad():
        vectors, _ = model([payload.text_a, payload.text_b])
        a, b = vectors[0], vectors[1]

        cosine_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        euclidean = torch.norm(a - b).item()

    return {
        "cosine_similarity": round(cosine_sim, 4),
        "euclidean_distance": round(euclidean, 4)
    }
