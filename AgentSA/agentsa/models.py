"""NMF and Dziugaite--Roy NNMF (called NeuNMF in AgentSA).

Architecture follows the cited NNMF paper: [U, V, U' * V'] for K=1.
AgentSA does not report its exact victim hyperparameters; our choices are explicit.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def normal_embedding(rows, columns, seed, std=0.1):
    # Separate streams ensure adding fake users does not change real/item initialization.
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator) * std
    return nn.Embedding.from_pretrained(values, freeze=False)


class NMF(nn.Module):
    def __init__(self, users, items, config, mean, seed):
        super().__init__()
        dim = config["nmf_dimensions"]
        center = math.sqrt(mean / dim)
        jitter = config.get("nmf_initial_jitter", 0.15)
        self.user = normal_embedding(users, dim, seed + 101, jitter * center)
        self.item = normal_embedding(items, dim, seed + 102, jitter * center)
        with torch.no_grad():
            self.user.weight.add_(center).clamp_(min=1e-8)
            self.item.weight.add_(center).clamp_(min=1e-8)

    def forward(self, users, items):
        return (self.user(users) * self.item(items)).sum(-1)

    def latent_parameters(self):
        return list(self.parameters())

    def network_parameters(self):
        return []

    def project(self):
        with torch.no_grad():
            self.user.weight.clamp_(min=1e-8)
            self.item.weight.clamp_(min=1e-8)


class NeuNMF(nn.Module):
    def __init__(self, users, items, config, mean, seed):
        super().__init__()
        d, product_d = config["neunmf_dimensions"], config["neunmf_product_dimensions"]
        self.user = normal_embedding(users, d, seed + 201)
        self.item = normal_embedding(items, d, seed + 202)
        self.user_product = normal_embedding(users, product_d, seed + 203)
        self.item_product = normal_embedding(items, product_d, seed + 204)
        widths = [2 * d + product_d, *config["neunmf_hidden"], 1]
        layers = []
        generator = torch.Generator().manual_seed(seed + 205)
        for idx, (n_in, n_out) in enumerate(zip(widths, widths[1:])):
            linear = nn.Linear(n_in, n_out)
            bound = 4 * math.sqrt(6 / (n_in + n_out))
            with torch.no_grad():
                linear.weight.uniform_(-bound, bound, generator=generator)
                linear.bias.zero_()
            layers.append(linear)
            if idx < len(widths) - 2:
                layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        # Fixed real-train mean for both clean and poison, documented added initialization.
        with torch.no_grad():
            self.network[-1].bias.fill_(mean)

    def forward(self, users, items):
        inputs = torch.cat([self.user(users), self.item(items),
                            self.user_product(users) * self.item_product(items)], dim=-1)
        return self.network(inputs).squeeze(-1)

    def latent_parameters(self):
        return [self.user.weight, self.item.weight,
                self.user_product.weight, self.item_product.weight]

    def network_parameters(self):
        return list(self.network.parameters())

    def project(self):
        pass


def build_model(name, users, items, config, mean, seed):
    if name not in {"nmf", "neunmf"}:
        raise ValueError(f"Unknown model: {name}")
    return {"nmf": NMF, "neunmf": NeuNMF}[name](users, items, config, mean, seed)
