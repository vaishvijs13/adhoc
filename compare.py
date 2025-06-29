from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import logging
from models import get_model

logger = logging.getLogger(__name__)
router = APIRouter()


class CompareInput(BaseModel):
    text_a: str
    text_b: str


class CompareResponse(BaseModel):
    cosine_similarity: float
    euclidean_distance: float
    similarity_interpretation: str
    ideology_analysis: dict


@router.post("/", response_model=CompareResponse)
async def compare_texts(payload: CompareInput):
    try:
        if not payload.text_a.strip() or not payload.text_b.strip():
            raise HTTPException(status_code=400, detail="Both texts must be non-empty")

        model = get_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        with torch.no_grad():
            results_a = model([payload.text_a.strip()], return_attention=False)
            results_b = model([payload.text_b.strip()], return_attention=False)

            embedding_a = results_a["embeddings"][0]
            embedding_b = results_b["embeddings"][0]

            cosine_sim = torch.nn.functional.cosine_similarity(
                embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)
            ).item()

            euclidean_dist = torch.norm(embedding_a - embedding_b).item()

            political_a = torch.argmax(results_a["political_logits"][0]).item()
            political_b = torch.argmax(results_b["political_logits"][0]).item()

            social_a = torch.argmax(results_a["social_logits"][0]).item()
            social_b = torch.argmax(results_b["social_logits"][0]).item()

            economic_a = torch.argmax(results_a["economic_logits"][0]).item()
            economic_b = torch.argmax(results_b["economic_logits"][0]).item()

            foreign_a = torch.argmax(results_a["foreign_logits"][0]).item()
            foreign_b = torch.argmax(results_b["foreign_logits"][0]).item()

        if cosine_sim > 0.8:
            similarity_interpretation = "Very similar"
        elif cosine_sim > 0.6:
            similarity_interpretation = "Moderately similar"
        elif cosine_sim > 0.4:
            similarity_interpretation = "Somewhat similar"
        elif cosine_sim > 0.2:
            similarity_interpretation = "Somewhat different"
        else:
            similarity_interpretation = "Very different"

        political_labels = ["left", "center", "right"]
        social_labels = ["liberal", "moderate", "conservative"]
        economic_labels = ["left", "center", "right"]
        foreign_labels = ["isolationist", "neutral", "interventionist"]

        ideology_analysis = {
            "text_a_ideology": {
                "political": political_labels[political_a],
                "social": social_labels[social_a],
                "economic": economic_labels[economic_a],
                "foreign": foreign_labels[foreign_a],
            },
            "text_b_ideology": {
                "political": political_labels[political_b],
                "social": social_labels[social_b],
                "economic": economic_labels[economic_b],
                "foreign": foreign_labels[foreign_b],
            },
            "ideology_agreement": {
                "political": political_a == political_b,
                "social": social_a == social_b,
                "economic": economic_a == economic_b,
                "foreign": foreign_a == foreign_b,
            },
        }

        return CompareResponse(
            cosine_similarity=round(cosine_sim, 4),
            euclidean_distance=round(euclidean_dist, 4),
            similarity_interpretation=similarity_interpretation,
            ideology_analysis=ideology_analysis,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")
