from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models import get_model
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_PATH = "data/trained_model.pt"
REFERENCE_DATA_PATH = "data/reference_embeddings.csv"


class VisualizationRequest(BaseModel):
    texts: List[str]
    method: Optional[str] = "umap"
    color_by: Optional[str] = "ideology"
    include_references: Optional[bool] = True


class ClusterAnalysisRequest(BaseModel):
    texts: List[str]
    n_clusters: Optional[int] = 3
    method: Optional[str] = "kmeans"


class IdeologySpaceRequest(BaseModel):
    texts: List[str]
    politicians: Optional[List[str]] = None


class VisualizationResponse(BaseModel):
    plot_data: Dict[str, Any]
    plot_html: Optional[str] = None
    clusters: Optional[List[int]] = None
    analysis: Dict[str, Any]


def load_reference_data():
    if os.path.exists(REFERENCE_DATA_PATH):
        df = pd.read_csv(REFERENCE_DATA_PATH)

        if "embedding" in df.columns:
            embeddings_list = []
            for emb_str in df["embedding"]:
                if isinstance(emb_str, str):
                    emb = np.array(eval(emb_str))
                else:
                    emb = np.array(emb_str)
                embeddings_list.append(emb)
            embeddings = np.stack(embeddings_list)
            return df, embeddings

    return None, None


def generate_embeddings(texts: List[str]) -> np.ndarray:
    model = get_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

    embeddings_list = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        with torch.no_grad():
            results = model(batch_texts, return_attention=False)
            batch_embeddings = results["embeddings"].cpu().numpy()
            embeddings_list.extend(batch_embeddings)

    return np.array(embeddings_list)


def apply_dimensionality_reduction(
    embeddings: np.ndarray, method: str = "umap", n_components: int = 2
) -> np.ndarray:
    if method.lower() == "umap":
        reducer = umap.UMAP(
            n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1
        )
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, len(embeddings) - 1),
        )
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)


def classify_texts(texts: List[str]) -> List[Dict[str, Any]]:
    model = get_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

    classifications = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        with torch.no_grad():
            results = model(batch_texts, return_attention=False)

            for j, text in enumerate(batch_texts):
                political_logits = results["political_logits"][j]
                social_logits = results["social_logits"][j]
                economic_logits = results["economic_logits"][j]
                foreign_logits = results["foreign_logits"][j]

                political_pred = torch.argmax(political_logits).item()
                social_pred = torch.argmax(social_logits).item()
                economic_pred = torch.argmax(economic_logits).item()
                foreign_pred = torch.argmax(foreign_logits).item()

                political_labels = ["left", "center", "right"]
                social_labels = ["liberal", "moderate", "conservative"]
                economic_labels = ["left", "center", "right"]
                foreign_labels = ["isolationist", "neutral", "interventionist"]

                classifications.append(
                    {
                        "text": text,
                        "political": political_labels[political_pred],
                        "social": social_labels[social_pred],
                        "economic": economic_labels[economic_pred],
                        "foreign": foreign_labels[foreign_pred],
                        "political_confidence": float(
                            torch.softmax(political_logits, dim=0)[political_pred]
                        ),
                        "social_confidence": float(
                            torch.softmax(social_logits, dim=0)[social_pred]
                        ),
                        "economic_confidence": float(
                            torch.softmax(economic_logits, dim=0)[economic_pred]
                        ),
                        "foreign_confidence": float(
                            torch.softmax(foreign_logits, dim=0)[foreign_pred]
                        ),
                    }
                )

    return classifications


@router.post("/ideology-space", response_model=VisualizationResponse)
async def visualize_ideology_space(payload: IdeologySpaceRequest):
    try:
        texts = payload.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        embeddings = generate_embeddings(texts)

        ref_df, ref_embeddings = load_reference_data()

        all_embeddings = embeddings
        all_texts = texts
        text_types = ["input"] * len(texts)
        politicians = ["User Input"] * len(texts)

        if ref_embeddings is not None:
            all_embeddings = np.vstack([embeddings, ref_embeddings])
            all_texts = texts + ref_df["text"].tolist()
            text_types = ["input"] * len(texts) + ["reference"] * len(ref_df)
            politicians = ["User Input"] * len(texts) + ref_df["politician"].tolist()

        # apply UMAP
        reduced_embeddings = apply_dimensionality_reduction(all_embeddings, "umap")

        classifications = classify_texts(texts)
        if ref_df is not None:
            ref_classifications = [
                {"political": pol} for pol in ref_df["political_lean"]
            ]
        else:
            ref_classifications = []

        all_classifications = classifications + ref_classifications

        # create da plot
        colors = []
        for i, classification in enumerate(all_classifications):
            if classification["political"] == "left":
                colors.append("blue")
            elif classification["political"] == "right":
                colors.append("red")
            else:
                colors.append("green")

        fig = go.Figure()

        # add input points
        input_indices = [i for i, t in enumerate(text_types) if t == "input"]
        if input_indices:
            fig.add_trace(
                go.Scatter(
                    x=reduced_embeddings[input_indices, 0],
                    y=reduced_embeddings[input_indices, 1],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=[colors[i] for i in input_indices],
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    text=[f"Input: {all_texts[i][:50]}..." for i in input_indices],
                    name="Input Texts",
                    hovertemplate="<b>%{text}</b><br>Political: %{customdata}<extra></extra>",
                    customdata=[
                        all_classifications[i]["political"] for i in input_indices
                    ],
                )
            )

        # ref points
        ref_indices = [i for i, t in enumerate(text_types) if t == "reference"]
        if ref_indices:
            fig.add_trace(
                go.Scatter(
                    x=reduced_embeddings[ref_indices, 0],
                    y=reduced_embeddings[ref_indices, 1],
                    mode="markers",
                    marker=dict(
                        size=8, color=[colors[i] for i in ref_indices], opacity=0.6
                    ),
                    text=[
                        f"{politicians[i]}: {all_texts[i][:50]}..." for i in ref_indices
                    ],
                    name="Reference Politicians",
                    hovertemplate="<b>%{text}</b><br>Political: %{customdata}<extra></extra>",
                    customdata=[
                        all_classifications[i]["political"] for i in ref_indices
                    ],
                )
            )

        fig.update_layout(
            title="Political Ideology Space",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode="closest",
            showlegend=True,
            width=800,
            height=600,
        )

        plot_html = fig.to_html(include_plotlyjs=True)

        analysis = {
            "total_points": len(all_embeddings),
            "input_points": len(texts),
            "reference_points": (
                len(ref_embeddings) if ref_embeddings is not None else 0
            ),
            "ideology_distribution": {
                "left": sum(1 for c in all_classifications if c["political"] == "left"),
                "center": sum(
                    1 for c in all_classifications if c["political"] == "center"
                ),
                "right": sum(
                    1 for c in all_classifications if c["political"] == "right"
                ),
            },
        }

        return VisualizationResponse(
            plot_data={
                "x": reduced_embeddings[:, 0].tolist(),
                "y": reduced_embeddings[:, 1].tolist(),
                "colors": colors,
                "texts": all_texts,
                "politicians": politicians,
                "types": text_types,
            },
            plot_html=plot_html,
            analysis=analysis,
        )

    except Exception as e:
        logger.error(f"Error in ideology space visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@router.post("/cluster-analysis", response_model=VisualizationResponse)
async def cluster_analysis(payload: ClusterAnalysisRequest):
    try:
        texts = payload.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        embeddings = generate_embeddings(texts)

        # perform clustering
        if payload.method.lower() == "kmeans":
            clusterer = KMeans(n_clusters=payload.n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering method: {payload.method}")

        cluster_labels = clusterer.fit_predict(embeddings)

        # apply dimensionality reduction for visualization
        reduced_embeddings = apply_dimensionality_reduction(embeddings, "umap")

        classifications = classify_texts(texts)

        fig = go.Figure()

        colors = px.colors.qualitative.Set1[: payload.n_clusters]

        for cluster_id in range(payload.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            fig.add_trace(
                go.Scatter(
                    x=reduced_embeddings[cluster_indices, 0],
                    y=reduced_embeddings[cluster_indices, 1],
                    mode="markers",
                    marker=dict(size=10, color=colors[cluster_id], opacity=0.7),
                    text=[
                        f"Cluster {cluster_id}: {texts[i][:50]}..."
                        for i in cluster_indices
                    ],
                    name=f"Cluster {cluster_id}",
                    hovertemplate="<b>%{text}</b><br>Political: %{customdata}<extra></extra>",
                    customdata=[
                        classifications[i]["political"] for i in cluster_indices
                    ],
                )
            )

        centers_2d = np.array(
            [
                reduced_embeddings[cluster_labels == i].mean(axis=0)
                for i in range(payload.n_clusters)
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=centers_2d[:, 0],
                y=centers_2d[:, 1],
                mode="markers",
                marker=dict(size=15, color="black", symbol="x", line=dict(width=2)),
                name="Cluster Centers",
                hovertemplate="Cluster Center %{pointNumber}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Political Text Clustering ({payload.method.upper()}, k={payload.n_clusters})",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode="closest",
            showlegend=True,
            width=800,
            height=600,
        )

        plot_html = fig.to_html(include_plotlyjs=True)

        cluster_analysis = {}
        for cluster_id in range(payload.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_classifications = [classifications[i] for i in cluster_indices]

            # count ideologies in cluster
            ideology_counts = {}
            for c in cluster_classifications:
                ideology = c["political"]
                ideology_counts[ideology] = ideology_counts.get(ideology, 0) + 1

            cluster_analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_indices),
                "ideology_distribution": ideology_counts,
                "dominant_ideology": (
                    max(ideology_counts.items(), key=lambda x: x[1])[0]
                    if ideology_counts
                    else "unknown"
                ),
                "sample_texts": cluster_texts[:3], # show first 3 as examples
            }

        return VisualizationResponse(
            plot_data={
                "x": reduced_embeddings[:, 0].tolist(),
                "y": reduced_embeddings[:, 1].tolist(),
                "clusters": cluster_labels.tolist(),
                "texts": texts,
            },
            plot_html=plot_html,
            clusters=cluster_labels.tolist(),
            analysis={
                "n_clusters": payload.n_clusters,
                "method": payload.method,
                "cluster_analysis": cluster_analysis,
                "total_texts": len(texts),
            },
        )

    except Exception as e:
        logger.error(f"Error in cluster analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@router.post("/dimensional-analysis")
async def dimensional_analysis(payload: VisualizationRequest):
    try:
        texts = payload.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        classifications = classify_texts(texts)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Social vs Economic",
                "Social vs Foreign",
                "Economic vs Foreign",
                "Overall Political Distribution",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        social_map = {"liberal": -1, "moderate": 0, "conservative": 1}
        economic_map = {"left": -1, "center": 0, "right": 1}

        social_scores = [social_map[c["social"]] for c in classifications]
        economic_scores = [economic_map[c["economic"]] for c in classifications]

        fig.add_trace(
            go.Scatter(
                x=social_scores,
                y=economic_scores,
                mode="markers",
                marker=dict(size=10, opacity=0.7),
                text=[f"{text[:50]}..." for text in texts],
                name="Texts",
            ),
            row=1,
            col=1,
        )

        foreign_map = {"isolationist": -1, "neutral": 0, "interventionist": 1}
        foreign_scores = [foreign_map[c["foreign"]] for c in classifications]

        fig.add_trace(
            go.Scatter(
                x=social_scores,
                y=foreign_scores,
                mode="markers",
                marker=dict(size=10, opacity=0.7),
                text=[f"{text[:50]}..." for text in texts],
                name="Texts",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=economic_scores,
                y=foreign_scores,
                mode="markers",
                marker=dict(size=10, opacity=0.7),
                text=[f"{text[:50]}..." for text in texts],
                name="Texts",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # overall political distribution
        political_counts = {"left": 0, "center": 0, "right": 0}
        for c in classifications:
            political_counts[c["political"]] += 1

        fig.add_trace(
            go.Bar(
                x=list(political_counts.keys()),
                y=list(political_counts.values()),
                marker=dict(color=["blue", "green", "red"]),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(
            title_text="Social Axis (Liberal ← → Conservative)", row=1, col=1
        )
        fig.update_yaxes(title_text="Economic Axis (Left ← → Right)", row=1, col=1)

        fig.update_xaxes(
            title_text="Social Axis (Liberal ← → Conservative)", row=1, col=2
        )
        fig.update_yaxes(
            title_text="Foreign Policy (Isolationist ← → Interventionist)", row=1, col=2
        )

        fig.update_xaxes(title_text="Economic Axis (Left ← → Right)", row=2, col=1)
        fig.update_yaxes(
            title_text="Foreign Policy (Isolationist ← → Interventionist)", row=2, col=1
        )

        fig.update_xaxes(title_text="Political Orientation", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.update_layout(
            title="Multi-Dimensional Political Analysis", height=800, width=1000
        )

        plot_html = fig.to_html(include_plotlyjs=True)

        analysis = {
            "ideology_distributions": {
                "social": {
                    label: sum(1 for c in classifications if c["social"] == label)
                    for label in ["liberal", "moderate", "conservative"]
                },
                "economic": {
                    label: sum(1 for c in classifications if c["economic"] == label)
                    for label in ["left", "center", "right"]
                },
                "foreign": {
                    label: sum(1 for c in classifications if c["foreign"] == label)
                    for label in ["isolationist", "neutral", "interventionist"]
                },
                "political": political_counts,
            },
            "correlations": {
                "social_economic": np.corrcoef(social_scores, economic_scores)[0, 1],
                "social_foreign": np.corrcoef(social_scores, foreign_scores)[0, 1],
                "economic_foreign": np.corrcoef(economic_scores, foreign_scores)[0, 1],
            },
            "average_scores": {
                "social": np.mean(social_scores),
                "economic": np.mean(economic_scores),
                "foreign": np.mean(foreign_scores),
            },
        }

        return VisualizationResponse(
            plot_data={
                "social_scores": social_scores,
                "economic_scores": economic_scores,
                "foreign_scores": foreign_scores,
                "political_counts": political_counts,
                "texts": texts,
            },
            plot_html=plot_html,
            analysis=analysis,
        )

    except Exception as e:
        logger.error(f"Error in dimensional analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/reference-data")
async def get_reference_data():
    try:
        ref_df, ref_embeddings = load_reference_data()

        if ref_df is None:
            return {"error": "No reference data available"}

        stats = {
            "total_statements": len(ref_df),
            "unique_politicians": (
                ref_df["politician"].nunique() if "politician" in ref_df.columns else 0
            ),
            "ideology_distribution": (
                ref_df["political_lean"].value_counts().to_dict()
                if "political_lean" in ref_df.columns
                else {}
            ),
            "date_range": (
                {
                    "earliest": (
                        ref_df["date"].min() if "date" in ref_df.columns else None
                    ),
                    "latest": (
                        ref_df["date"].max() if "date" in ref_df.columns else None
                    ),
                }
                if "date" in ref_df.columns
                else None
            ),
            "politicians": (
                ref_df["politician"].unique().tolist()
                if "politician" in ref_df.columns
                else []
            ),
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting reference data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
