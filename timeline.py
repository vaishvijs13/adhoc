# backend/timeline.py
from fastapi import APIRouter
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from models import model

router = APIRouter()

DATA_PATH = "data/stmts.csv"


def load_political_statements(politician: str):
    df = pd.read_csv(DATA_PATH)
    df = df[df["politician"].str.lower() == politician.lower()]
    df = df.sort_values("date")
    return df.to_dict(orient="records")


@router.get("/{politician}/trajectory")
def get_trajectory(politician: str):
    records = load_political_statements(politician)
    if not records:
        return {"error": "Politician not found"}

    texts = [r["text"] for r in records]
    dates = [r["date"] for r in records]

    with torch.no_grad():
        embeddings, _ = model(texts)
        vectors = embeddings.detach().cpu().numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    return {
        "politician": politician,
        "trajectory": [
            {"date": d, "x": float(x), "y": float(y), "text": t}
            for d, (x, y), t in zip(dates, reduced, texts)
        ],
    }
