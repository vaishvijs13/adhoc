from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import re
import string
from models import model
from sklearn.preprocessing import MultiLabelBinarizer
import faiss
import pandas as pd

router = APIRouter()

REFERENCE_DATA_PATH = "data/reference_embeddings.csv"
REFERENCE_INDEX_PATH = "data/reference_index.faiss"

def load_reference_embeddings():
    df = pd.read_csv(REFERENCE_DATA_PATH)
    embeddings = np.stack(df['embedding'].apply(eval).values)
    return df, embeddings

REFERENCE_DF, REFERENCE_EMBEDDINGS = load_reference_embeddings()

REFERENCE_EMBEDDINGS = REFERENCE_EMBEDDINGS / np.linalg.norm(REFERENCE_EMBEDDINGS, axis=1, keepdims=True)
INDEX = faiss.IndexFlatIP(REFERENCE_EMBEDDINGS.shape[1])
INDEX.add(REFERENCE_EMBEDDINGS)

class AnalyzeRequest(BaseModel):
    text: str
    include_neighbors: Optional[bool] = True
    top_k: Optional[int] = 5

class NeighborMatch(BaseModel):
    politician: str
    distance: float
    sample_text: str
    date: Optional[str]

class IdeologyScores(BaseModel):
    social: str
    economic: str
    foreign: str
    probabilities: List[float]

class AnalyzeResponse(BaseModel):
    ideology_vector: List[float]
    predicted_label: str
    multi_label_ideology: IdeologyScores
    explanation_phrases: List[str]
    neighbors: Optional[List[NeighborMatch]]

def clean_text(text: str) -> str:
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[\\r\\n]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def get_token_importance(text: str) -> List[str]:
    tokens = model.tokenizer.tokenize(text)
    encoding = model.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.encoder(**encoding, output_attentions=True)
        attentions = outputs.attentions[-1].squeeze(0).mean(dim=0)  # [seq, seq]
        importance = attentions[0].numpy()
    ranked = sorted(zip(tokens, importance), key=lambda x: x[1], reverse=True)
    return [tok for tok, _ in ranked[:5]]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify_logits(logits: torch.Tensor) -> str:
    probs = softmax(logits.numpy())
    labels = ["left", "center", "right"]
    return labels[np.argmax(probs)]

def multilabel_classify(vector: np.ndarray) -> IdeologyScores:
    #will replace with learned axes
    SOCIAL_AXIS = np.random.normal(size=vector.shape)
    ECON_AXIS = np.random.normal(size=vector.shape)
    FOREIGN_AXIS = np.random.normal(size=vector.shape)

    SOCIAL_AXIS /= np.linalg.norm(SOCIAL_AXIS)
    ECON_AXIS /= np.linalg.norm(ECON_AXIS)
    FOREIGN_AXIS /= np.linalg.norm(FOREIGN_AXIS)

    social_score = np.dot(vector, SOCIAL_AXIS)
    econ_score = np.dot(vector, ECON_AXIS)
    foreign_score = np.dot(vector, FOREIGN_AXIS)

    def interpret(val, axis):
        if axis == "social":
            return "liberal" if val < -0.2 else "conservative" if val > 0.2 else "moderate"
        elif axis == "economic":
            return "left" if val < -0.2 else "right" if val > 0.2 else "center"
        elif axis == "foreign":
            return "isolationist" if val < -0.2 else "interventionist" if val > 0.2 else "neutral"

    return IdeologyScores(
        social=interpret(social_score, "social"),
        economic=interpret(econ_score, "economic"),
        foreign=interpret(foreign_score, "foreign"),
        probabilities=[round(float(x), 4) for x in [social_score, econ_score, foreign_score]]
    )

def find_neighbors(vec: np.ndarray, k: int) -> List[dict]:
    query = vec / np.linalg.norm(vec)
    dists, idxs = INDEX.search(np.array([query]), k)
    results = []
    for i in range(k):
        row = REFERENCE_DF.iloc[idxs[0][i]]
        results.append({
            "politician": row["politician"],
            "distance": round(float(dists[0][i]), 4),
            "sample_text": row["text"][:200],
            "date": row.get("date", "N/A")
        })
    return results

@router.post("/", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input.")

    cleaned = clean_text(text)
    with torch.no_grad():
        emb, logits = model([cleaned])
        vec_np = emb[0].detach().numpy()
        logit = logits[0]

    pred_label = classify_logits(logit)
    top_phrases = get_token_importance(cleaned)
    multi_label = multilabel_classify(vec_np)

    result = {
        "ideology_vector": vec_np.tolist(),
        "predicted_label": pred_label,
        "explanation_phrases": top_phrases,
        "multi_label_ideology": multi_label
    }

    if payload.include_neighbors:
        result["neighbors"] = find_neighbors(vec_np, payload.top_k)

    return result
