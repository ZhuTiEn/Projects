from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .review_models import build_review_model
from .training import predict
from .utils import digest, file_hash, seed_everything, write_json


def fit_review_model(name, train, validation, users, items, text_features,
                     config, mean, seed, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["reviews"]["training"]
    device = config["training"]["device"]
    feature_hash = hashlib.sha256()
    for value in text_features:
        feature_hash.update(str(value.shape).encode("ascii"))
        feature_hash.update(np.ascontiguousarray(value).tobytes())
    fingerprint = digest({"identity": identity, "model": name, "settings": settings,
                          "users": users, "items": items, "mean": mean, "seed": seed,
                          "text_features_sha256": feature_hash.hexdigest(),
                          "source_hashes": {name: file_hash(Path(__file__).with_name(name))
                                            for name in ("review_models.py", "review_training.py")}})
    seed_everything(seed, config["training"].get("threads", 4))
    model = build_review_model(name, users, items, text_features, settings, mean, seed).to(device)
    checkpoint = directory / "model.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Existing review-model checkpoint has different settings.")
        model.load_state_dict(saved["state_dict"])
        return model
    tensor = torch.as_tensor(train, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    rng = np.random.default_rng(seed)
    best, state, stale, best_epoch = float("inf"), None, 0, 0
    history, started = [], time.monotonic()
    for epoch in range(settings["epochs"]):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), settings["batch_size"]):
            ids = torch.as_tensor(order[start:start + settings["batch_size"]], device=device)
            batch = tensor[ids]
            optimizer.zero_grad(set_to_none=True)
            estimate = model(batch[:, 0].long(), batch[:, 1].long())
            loss = torch.mean((estimate - batch[:, 2]) ** 2)
            loss = loss + settings["l2"] * sum(p.square().sum() for p in model.parameters()) / len(train)
            loss.backward()
            optimizer.step()
        estimate = predict(model, validation, device, config["experiment"]["prediction_batch"], False)
        rmse = float(np.sqrt(np.mean((estimate - validation[:, 2]) ** 2)))
        history.append({"epoch": epoch + 1, "validation_rmse_raw": rmse})
        if rmse < best - 1e-7:
            best, state, stale, best_epoch = rmse, copy.deepcopy(model.state_dict()), 0, epoch + 1
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    torch.save({"fingerprint": fingerprint, "state_dict": state}, checkpoint)
    write_json(directory / "training.json", {"fingerprint": fingerprint, "identity": identity,
        "model": name, "seed": seed, "best_epoch": best_epoch, "best_validation_rmse_raw": best,
        "elapsed_seconds": time.monotonic()-started, "history": history, "settings": settings})
    return model
