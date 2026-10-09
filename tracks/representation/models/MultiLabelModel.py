from __future__ import annotations

import torch.nn as nn


class MultiLabelModel(nn.Module):
    """Multi-label classifier: logistic regression or N-layer MLP.

    hidden_dims controls the architecture:
        []       → logistic regression (default, fastest)
        [64]     → 1-hidden-layer MLP
        [64, 32] → 2-hidden-layer MLP

    State-dict key convention
    -------------------------
    hidden_dims=[] (logistic regression):
        Weights stored under ``linear.weight`` / ``linear.bias``.

    hidden_dims=[...] (MLP):
        Weights stored under ``net.{i}.weight`` / ``net.{i}.bias``
        following nn.Sequential insertion order.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims else []

        if not hidden_dims:
            self.linear = nn.Linear(n_features, n_classes)
            self._mlp = False
        else:
            act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU
            layers: list[nn.Module] = []
            in_dim = n_features
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim))
                layers.append(act_cls())
                layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, n_classes))
            self.net = nn.Sequential(*layers)
            self._mlp = True

    def forward(self, x):
        return self.net(x) if self._mlp else self.linear(x)
