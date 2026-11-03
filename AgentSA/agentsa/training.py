from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .models import build_model
from .utils import digest, file_hash, seed_everything, write_json


@torch.no_grad()
def predict(model, pairs, device="cpu", batch_size=8192, clip=False):
    model.eval()
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        users = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
        items = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
        value = model(users, items).cpu().numpy()
        if clip:
            value = np.clip(value, 1, 5)
        output.append(value)
    return np.concatenate(output) if output else np.empty(0)


def fit_model(name, train, validation, users, items, config, mean, seed, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    device = settings["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set training.device: cpu.")
    seed_everything(seed, settings.get("threads", 4))
    model = build_model(name, users, items, settings, mean, seed).to(device)
    fingerprint = digest({"identity": identity, "model": name, "settings": settings,
                          "users": users, "items": items, "mean": mean, "seed": seed,
                          "source_hashes": {name: file_hash(Path(__file__).with_name(name))
                                            for name in ["models.py", "training.py"]}})
    checkpoint = directory / "model.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Existing checkpoint uses different settings: {directory}; choose a new output.")
        model.load_state_dict(saved["state_dict"])
        return model
    latents, network = model.latent_parameters(), model.network_parameters()
    lr = settings["learning_rate"]
    alternating = name == "neunmf" and settings["neunmf_optimization"] == "alternating"
    if name == "neunmf" and settings["neunmf_optimizer"] == "rmsprop":
        optimizer_type = torch.optim.RMSprop
    else:
        optimizer_type = torch.optim.Adam
    if alternating:
        optimizers = [optimizer_type(latents, lr=lr), optimizer_type(network, lr=lr)]
    else:
        optimizers = [optimizer_type(model.parameters(), lr=lr)]
    tensor = torch.as_tensor(train, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    start_time = time.monotonic()
    for epoch in range(settings["epochs"]):
        model.train()
        for phase, optimizer in enumerate(optimizers):
            if alternating:
                for param in latents:
                    param.requires_grad_(phase == 0)
                for param in network:
                    param.requires_grad_(phase == 1)
            order = rng.permutation(len(train))
            for start in range(0, len(order), settings["batch_size"]):
                ids = torch.as_tensor(order[start:start + settings["batch_size"]], device=device)
                batch = tensor[ids]
                optimizer.zero_grad(set_to_none=True)
                estimate = model(batch[:, 0].long(), batch[:, 1].long())
                loss = torch.mean((estimate - batch[:, 2]) ** 2)
                # Sum-square objective divided by number of observed train entries.
                regularization = sum(param.square().sum() for param in latents)
                loss = loss + settings["latent_l2"] * regularization / len(train)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss: {name}, epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
                model.project()
        for param in model.parameters():
            param.requires_grad_(True)
        estimate = predict(model, validation, device, config["experiment"]["prediction_batch"], False)
        rmse = float(np.sqrt(np.mean((estimate - validation[:, 2]) ** 2)))
        if not np.isfinite(rmse):
            raise ValueError("Non-finite validation score.")
        history.append({"epoch": epoch + 1, "validation_rmse_raw": rmse})
        if rmse < best - 1e-7:
            best = rmse
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            best_epoch, stale = epoch + 1, 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "fingerprint": fingerprint}, checkpoint)
    write_json(directory / "training.json", {
        "fingerprint": fingerprint, "identity": identity, "seed": seed, "model": name,
        "best_epoch": best_epoch, "best_validation_rmse_raw": best,
        "elapsed_seconds": time.monotonic() - start_time, "history": history,
        "torch_version": str(torch.__version__), "device": device, "settings": settings,
    })
    return model
