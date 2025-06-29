import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import os


class IdeologyEmbeddingModel(nn.Module):
    def __init__(
        self, model_name="bert-base-uncased", projection_dim=256, num_political_labels=3
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, projection_dim),
            nn.Tanh(),
        )

        self.attention_head = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        self.political_classifier = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_political_labels),
        )

        self.social_classifier = nn.Sequential(
            nn.Linear(projection_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

        self.economic_classifier = nn.Sequential(
            nn.Linear(projection_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

        self.foreign_classifier = nn.Sequential(
            nn.Linear(projection_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

        self.ideology_axes = nn.Parameter(torch.randn(3, projection_dim))

    def forward(self, input_texts: List[str], return_attention: bool = True):
        encoding = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        outputs = self.encoder(**encoding, output_attentions=return_attention)
        hidden_states = outputs.last_hidden_state

        attended_output, attention_weights = self.attention_head(
            hidden_states, hidden_states, hidden_states
        )

        attention_mask = encoding["attention_mask"].unsqueeze(-1)
        masked_output = attended_output * attention_mask
        pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1)

        embeddings = self.projection_head(pooled_output)

        political_logits = self.political_classifier(embeddings)

        social_logits = self.social_classifier(embeddings)
        economic_logits = self.economic_classifier(embeddings)
        foreign_logits = self.foreign_classifier(embeddings)

        results = {
            "embeddings": embeddings,
            "political_logits": political_logits,
            "social_logits": social_logits,
            "economic_logits": economic_logits,
            "foreign_logits": foreign_logits,
            "ideology_axes_scores": torch.matmul(embeddings, self.ideology_axes.T),
        }

        if return_attention:
            results["attention_weights"] = attention_weights
            results["tokens"] = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in encoding["input_ids"]
            ]

        return results

    def get_token_importance(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        with torch.no_grad():
            results = self.forward([text], return_attention=True)
            attention = results["attention_weights"][0].mean(
                dim=0
            )  # Average across heads
            tokens = results["tokens"][0]

            token_scores = []
            for i, token in enumerate(tokens):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    score = attention[0, i].item()
                    token_scores.append((token, score))

            token_scores.sort(key=lambda x: x[1], reverse=True)
            return token_scores[:top_k]

    def save_model(self, path: str):
        torch.save(
            {"model_state_dict": self.state_dict(), "tokenizer": self.tokenizer}, path
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        positive_mask.fill_diagonal_(0)

        positive_sim = similarity_matrix * positive_mask
        negative_sim = similarity_matrix * negative_mask

        pos_loss = -torch.log(torch.exp(positive_sim).sum(dim=1) + 1e-8)
        neg_loss = -torch.log(1 - torch.exp(negative_sim).sum(dim=1) + 1e-8)

        return (pos_loss + neg_loss).mean()


model = None


def get_model(model_path: Optional[str] = None) -> IdeologyEmbeddingModel:
    global model
    if model is None:
        model = IdeologyEmbeddingModel()
        if model_path and os.path.exists(model_path):
            model.load_model(model_path)
        model.eval()
    return model


def initialize_model(model_path: Optional[str] = None):
    global model
    model = get_model(model_path)
    return model


model = initialize_model()
