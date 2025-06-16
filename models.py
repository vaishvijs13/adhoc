import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List

class IdeologyEmbeddingModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", projection_dim=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, projection_dim)
        )

        self.classifier = nn.Linear(projection_dim, 3) # for usa (left, cntr, right)

    def forward(self, input_texts: List[str]):
        encoding = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.encoder(**encoding)
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = self.projection(pooled)
        logits = self.classifier(projected)
        return projected, logits

model = IdeologyEmbeddingModel()
model.eval()
