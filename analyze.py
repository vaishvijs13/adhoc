from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import re
from models import get_model
import faiss
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
REFERENCE_DATA_PATH = "data/reference_embeddings.csv"
REFERENCE_INDEX_PATH = "data/reference_index.faiss"
MODEL_PATH = "data/trained_model.pt"


class AnalyzeRequest(BaseModel):
    text: str
    include_neighbors: Optional[bool] = True
    top_k: Optional[int] = 5
    return_attention: Optional[bool] = True


class NeighborMatch(BaseModel):
    politician: str
    similarity_score: float
    sample_text: str
    date: Optional[str] = None
    political_lean: Optional[str] = None


class IdeologyScores(BaseModel):
    social: str
    social_confidence: float
    economic: str
    economic_confidence: float
    foreign: str
    foreign_confidence: float
    axis_scores: List[float]


class TokenImportance(BaseModel):
    token: str
    importance_score: float
    position: int


class AnalyzeResponse(BaseModel):
    ideology_vector: List[float]
    predicted_label: str
    label_confidence: float
    multi_label_ideology: IdeologyScores
    important_tokens: List[TokenImportance]
    neighbors: Optional[List[NeighborMatch]] = None
    processing_info: Dict[str, Any]


class ReferenceDatabase:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self.index = None
        self.load_or_create()

    def load_or_create(self):
        if os.path.exists(REFERENCE_DATA_PATH):
            try:
                self.load_reference_data()
                logger.info(f"Loaded {len(self.df)} reference embeddings")
            except Exception as e:
                logger.warning(f"Error loading reference data: {e}")
                self.create_sample_reference_data()
        else:
            self.create_sample_reference_data()

    def load_reference_data(self):
        self.df = pd.read_csv(REFERENCE_DATA_PATH)

        if "embedding" in self.df.columns:
            embeddings_list = []
            for emb_str in self.df["embedding"]:
                if isinstance(emb_str, str):
                    emb = np.array(eval(emb_str))
                else:
                    emb = np.array(emb_str)
                embeddings_list.append(emb)
            self.embeddings = np.stack(embeddings_list)
        else:
            self.generate_embeddings()

        # normalize and build index
        self.embeddings = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )
        self.build_index()

    def generate_embeddings(self):
        model = get_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)
        texts = self.df["text"].tolist()

        embeddings_list = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            with torch.no_grad():
                results = model(batch_texts, return_attention=False)
                batch_embeddings = results["embeddings"].cpu().numpy()
                embeddings_list.extend(batch_embeddings)

        self.embeddings = np.array(embeddings_list)

        # save embeddings back to CSV
        self.df["embedding"] = [emb.tolist() for emb in self.embeddings]
        self.df.to_csv(REFERENCE_DATA_PATH, index=False)

    def build_index(self):
        if self.embeddings is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype(np.float32))

            faiss.write_index(self.index, REFERENCE_INDEX_PATH)

    def create_sample_reference_data(self):
        sample_data = {
            "politician": [
                "Bernie Sanders",
                "AOC",
                "Elizabeth Warren",
                "Joe Biden",
                "Kamala Harris",
                "Chuck Schumer",
                "Donald Trump",
                "Ron DeSantis",
                "Ted Cruz",
                "Mitt Romney",
                "Susan Collins",
                "John McCain",
            ],
            "text": [
                "Healthcare is a human right and we need Medicare for All",
                "The Green New Deal will create millions of jobs while fighting climate change",
                "We need to break up big tech monopolies and protect workers",
                "We must work across the aisle to rebuild our infrastructure",
                "Criminal justice reform is essential for a fair society",
                "Protecting voting rights is fundamental to our democracy",
                "America First policies will bring back manufacturing jobs",
                "We need strong borders and merit-based immigration",
                "Conservative values and limited government are our foundation",
                "Fiscal responsibility and bipartisan cooperation matter",
                "We must find common ground on important issues",
                "National security requires strong military and alliances",
            ],
            "political_lean": [
                "left",
                "left",
                "left",
                "center",
                "center",
                "center",
                "right",
                "right",
                "right",
                "center",
                "center",
                "center",
            ],
            "date": [
                "2023-01-15",
                "2023-02-20",
                "2023-03-10",
                "2023-01-25",
                "2023-02-15",
                "2023-03-05",
                "2023-01-30",
                "2023-02-25",
                "2023-03-15",
                "2023-01-20",
                "2023-02-10",
                "2023-03-01",
            ],
        }

        self.df = pd.DataFrame(sample_data)
        self.df.to_csv(REFERENCE_DATA_PATH, index=False)
        self.generate_embeddings()
        self.build_index()
        logger.info(f"created sample reference data with {len(self.df)} entries")

    def find_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None:
            return []

        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for i in range(k):
            if i < len(indices[0]):
                idx = indices[0][i]
                similarity = float(similarities[0][i])
                row = self.df.iloc[idx]

                results.append(
                    {
                        "politician": row["politician"],
                        "similarity_score": similarity,
                        "sample_text": (
                            row["text"][:200] + "..."
                            if len(row["text"]) > 200
                            else row["text"]
                        ),
                        "date": row.get("date", "N/A"),
                        "political_lean": row.get("political_lean", "unknown"),
                    }
                )

        return results


reference_db = ReferenceDatabase()


def clean_and_preprocess_text(text: str) -> str:
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"[@#](\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\!\?\,\;\:]", " ", text)
    text = " ".join(text.split())
    return text.strip()


def extract_key_phrases(text: str, top_k: int = 5) -> List[str]:
    cleaned_text = clean_and_preprocess_text(text)

    try:
        tokens = word_tokenize(cleaned_text.lower())
        stop_words = set(stopwords.words("english"))

        filtered_tokens = [
            token
            for token in tokens
            if token.isalpha() and len(token) > 2 and token not in stop_words
        ]

        # simple frequency-based extraction
        from collections import Counter

        word_freq = Counter(filtered_tokens)
        key_phrases = [word for word, freq in word_freq.most_common(top_k)]

        return key_phrases
    except Exception as e:
        logger.warning(f"Error in phrase extraction: {e}")
        return text.split()[:top_k]


def classify_ideology_scores(logits: torch.Tensor, axis_name: str) -> tuple:
    probabilities = torch.softmax(logits, dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = float(probabilities[predicted_class])

    if axis_name == "social":
        labels = ["liberal", "moderate", "conservative"]
    elif axis_name == "economic":
        labels = ["left", "center", "right"]
    elif axis_name == "foreign":
        labels = ["isolationist", "neutral", "interventionist"]
    else:
        labels = ["left", "center", "right"]

    return labels[predicted_class], confidence


@router.post("/", response_model=AnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty input text")

        if len(text) > 10000:
            raise HTTPException(
                status_code=400, detail="Text too long (max 10,000 characters)"
            )

        model = get_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

        cleaned_text = clean_and_preprocess_text(text)

        # run model inference
        with torch.no_grad():
            results = model([cleaned_text], return_attention=payload.return_attention)

        embedding = results["embeddings"][0].cpu().numpy()
        political_logits = results["political_logits"][0]
        social_logits = results["social_logits"][0]
        economic_logits = results["economic_logits"][0]
        foreign_logits = results["foreign_logits"][0]
        axis_scores = results["ideology_axes_scores"][0].cpu().numpy()

        political_label, political_confidence = classify_ideology_scores(
            political_logits, "political"
        )
        social_label, social_confidence = classify_ideology_scores(
            social_logits, "social"
        )
        economic_label, economic_confidence = classify_ideology_scores(
            economic_logits, "economic"
        )
        foreign_label, foreign_confidence = classify_ideology_scores(
            foreign_logits, "foreign"
        )

        important_tokens = []
        if payload.return_attention:
            try:
                token_importance = model.get_token_importance(cleaned_text, top_k=10)
                important_tokens = [
                    TokenImportance(
                        token=token.replace("##", ""),  # Remove BERT subword markers
                        importance_score=float(score),
                        position=i,
                    )
                    for i, (token, score) in enumerate(token_importance)
                    if not token.startswith("[") and not token.startswith("#")
                ][:5]
            except Exception as e:
                logger.warning(f"Error extracting token importance: {e}")
                key_phrases = extract_key_phrases(cleaned_text)
                important_tokens = [
                    TokenImportance(token=phrase, importance_score=1.0, position=i)
                    for i, phrase in enumerate(key_phrases)
                ]

        neighbors = None
        if payload.include_neighbors:
            try:
                neighbors = reference_db.find_similar(embedding, payload.top_k)
                neighbors = [NeighborMatch(**neighbor) for neighbor in neighbors]
            except Exception as e:
                logger.warning(f"Error finding neighbors: {e}")
                neighbors = []

        response = AnalyzeResponse(
            ideology_vector=embedding.tolist(),
            predicted_label=political_label,
            label_confidence=political_confidence,
            multi_label_ideology=IdeologyScores(
                social=social_label,
                social_confidence=social_confidence,
                economic=economic_label,
                economic_confidence=economic_confidence,
                foreign=foreign_label,
                foreign_confidence=foreign_confidence,
                axis_scores=axis_scores.tolist(),
            ),
            important_tokens=important_tokens,
            neighbors=neighbors,
            processing_info={
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "model_used": "IdeologyEmbeddingModel",
                "embedding_dim": len(embedding),
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    try:
        from models import model as global_model

        return {
            "status": "healthy",
            "model_loaded": global_model is not None,
            "reference_db_size": (
                len(reference_db.df) if reference_db.df is not None else 0
            ),
        }
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        return {"status": "healthy", "model_loaded": False, "reference_db_size": 0}
