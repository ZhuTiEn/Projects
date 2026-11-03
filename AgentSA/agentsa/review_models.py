"""Explicit review-aware victim models; engineering adaptations, not author code."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import torch
from torch import nn

from .models import normal_embedding


class ReviewTextEncoder:
    def __init__(self, settings, mock=False):
        self.mock = mock or settings["encoder"] == "hash"
        self.dimension = int(settings.get("hash_dimensions", 64))
        if self.mock:
            self.model = None
            self.identity = {"type": "hash", "dimensions": self.dimension}
        else:
            from sentence_transformers import SentenceTransformer
            kwargs = {"local_files_only": settings.get("encoder_local_only", False)}
            if settings.get("encoder_revision"):
                kwargs["revision"] = settings["encoder_revision"]
            self.model = SentenceTransformer(settings["encoder_model"], **kwargs)
            self.dimension = int(self.model.get_sentence_embedding_dimension())
            self.identity = {"type": "sentence_transformer", "model": settings["encoder_model"],
                             "revision": settings.get("encoder_revision"),
                             "dimensions": self.dimension}

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                                show_progress_bar=False), dtype=np.float32)
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", str(text).casefold()):
                raw = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(raw[:4], "big") % self.dimension
                sign = 1.0 if raw[4] & 1 else -1.0
                output[row, index] += sign
            norm = np.linalg.norm(output[row])
            if norm:
                output[row] /= norm
        return output


def aggregate_features(rows, users, items, encoder, base=None):
    if base is None:
        user_sum = np.zeros((users, encoder.dimension), dtype=np.float32)
        item_sum = np.zeros((items, encoder.dimension), dtype=np.float32)
        user_count = np.zeros(users, dtype=np.int64)
        item_count = np.zeros(items, dtype=np.int64)
    else:
        user_sum, item_sum, user_count, item_count = [x.copy() for x in base]
        if users > len(user_count):
            user_sum = np.pad(user_sum, ((0, users-len(user_count)), (0, 0)))
            user_count = np.pad(user_count, (0, users-len(user_count)))
    texts = [(row.get("summary", "") + " " + row.get("review", "")).strip() for row in rows]
    vectors = encoder.encode(texts)
    for row, vector in zip(rows, vectors):
        user, item = int(row["user_id"]), int(row["item_id"])
        user_sum[user] += vector
        item_sum[item] += vector
        user_count[user] += 1
        item_count[item] += 1
    return user_sum, item_sum, user_count, item_count


def means(aggregates):
    user_sum, item_sum, user_count, item_count = aggregates
    users = np.divide(user_sum, np.maximum(user_count[:, None], 1),
                      out=np.zeros_like(user_sum), where=user_count[:, None] > 0)
    items = np.divide(item_sum, np.maximum(item_count[:, None], 1),
                      out=np.zeros_like(item_sum), where=item_count[:, None] > 0)
    return users, items


class DualTowerEarly(nn.Module):
    def __init__(self, users, items, text_features, settings, mean, seed):
        super().__init__()
        dim = settings["id_dimensions"]
        self.user = normal_embedding(users, dim, seed + 501)
        self.item = normal_embedding(items, dim, seed + 502)
        user_text, item_text = text_features
        self.register_buffer("user_text", torch.as_tensor(user_text, dtype=torch.float32))
        self.register_buffer("item_text", torch.as_tensor(item_text, dtype=torch.float32))
        text_dim = user_text.shape[1]
        hidden = settings["hidden_dimensions"]
        self.user_tower = nn.Sequential(nn.Linear(dim + text_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden))
        self.item_tower = nn.Sequential(nn.Linear(dim + text_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden))
        self.bias = nn.Parameter(torch.tensor(float(mean)))

    def forward(self, users, items):
        u = self.user_tower(torch.cat([self.user(users), self.user_text[users]], -1))
        i = self.item_tower(torch.cat([self.item(items), self.item_text[items]], -1))
        return (u * i).sum(-1) / np.sqrt(u.shape[-1]) + self.bias


class DualTowerLate(nn.Module):
    def __init__(self, users, items, text_features, settings, mean, seed):
        super().__init__()
        dim = settings["id_dimensions"]
        self.user = normal_embedding(users, dim, seed + 601)
        self.item = normal_embedding(items, dim, seed + 602)
        user_text, item_text = text_features
        self.register_buffer("user_text", torch.as_tensor(user_text, dtype=torch.float32))
        self.register_buffer("item_text", torch.as_tensor(item_text, dtype=torch.float32))
        text_dim = user_text.shape[1]
        self.user_projection = nn.Linear(text_dim, dim, bias=False)
        self.item_projection = nn.Linear(text_dim, dim, bias=False)
        self.text_weight = nn.Parameter(torch.tensor(float(settings.get("late_text_weight", 1.0))))
        self.bias = nn.Parameter(torch.tensor(float(mean)))

    def forward(self, users, items):
        id_score = (self.user(users) * self.item(items)).sum(-1)
        ut = self.user_projection(self.user_text[users])
        it = self.item_projection(self.item_text[items])
        text_score = (ut * it).sum(-1) / np.sqrt(ut.shape[-1])
        return id_score + self.text_weight * text_score + self.bias


def build_review_model(name, users, items, text_features, settings, mean, seed):
    models = {"dual_tower_early": DualTowerEarly, "dual_tower_late": DualTowerLate}
    if name not in models:
        raise ValueError(f"Unknown review-aware model: {name}")
    torch.manual_seed(seed)
    return models[name](users, items, text_features, settings, mean, seed)
