import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.file_management import processed_split_path

torch.set_default_dtype(torch.float64)

ACTIVATIONS = {
    "relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU,
    "gelu": nn.GELU, "swish": nn.SiLU, "mish": nn.Mish,
}

AGGREGATIONS = {
    "max": lambda x: x.max(dim=1)[0],
    "mean": lambda x: x.mean(dim=1),
    "sum": lambda x: x.sum(dim=1),
    "min": lambda x: x.min(dim=1)[0],
    "hadamard_prod": lambda x: F.relu(x).prod(dim=1),
}


def build_mlp(dims, activation, dropout=0.0):
    """Linear layers over `dims` with activation (and dropout) between them, none after the last."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def build_head(input_dim, hidden_dims, activation):
    """MLP head with a single output; the output layer is appended when `hidden_dims` doesn't end in 1."""
    if hidden_dims and hidden_dims[-1] != 1:
        hidden_dims = hidden_dims + [1]
    return build_mlp([input_dim] + (hidden_dims or [1]), activation)


def postprocess(output, task, volume_scale_factor):
    """Raw network output -> predictions: 0/1 status and/or volume in original units."""
    if task == "IntersectionStatus":
        return (output > 0).int().squeeze()
    if task == "IntersectionVolume":
        return output.squeeze() / volume_scale_factor
    if task == "IntersectionStatus_IntersectionVolume":
        return torch.stack([(output[:, 0] > 0).int(), output[:, 1] / volume_scale_factor], dim=1)
    raise ValueError(f"Unknown task: {task}")


class BaseNet(nn.Module):
    """Feature extractor (built by subclasses) followed by a classification head and a regression head."""

    def __init__(self, config):
        super().__init__()
        self.task = config["task"]
        self.volume_scale_factor = config["volume_scale_factor"]
        self.input_dim = config["input_dim"]
        self.activation = ACTIVATIONS[config["activation"].lower()]()
        self.dropout = config.get("dropout_rate", 0.0)
        self.feature_extractor, feature_dim = self._build_features(config)
        self.classification_head = build_head(feature_dim, config.get("classification_head", []), self.activation)
        self.regression_head = build_head(feature_dim, config.get("regression_head", []), self.activation)

    def _build_features(self, config):
        """Return (feature extractor module, its output dimension)."""
        raise NotImplementedError

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.extract_features(x)
        if self.task == "IntersectionStatus":
            return self.classification_head(features)
        if self.task == "IntersectionVolume":
            return F.relu(self.regression_head(features))
        if self.task == "IntersectionStatus_IntersectionVolume":
            return torch.cat([self.classification_head(features), F.relu(self.regression_head(features))], dim=1)
        raise ValueError(f"Unknown task: {self.task}")

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return postprocess(self(x), self.task, self.volume_scale_factor)


class SimpleMLP(BaseNet):
    def _build_features(self, config):
        layers = config.get("shared_layers", [])
        if not layers:
            return nn.Identity(), config["input_dim"]
        return build_mlp([config["input_dim"]] + layers, self.activation, self.dropout), layers[-1]


class TetrahedronPairNet(BaseNet):
    """Per-vertex -> per-tetrahedron -> per-pair network with residual shortcuts.

    Vertices (and the two tetrahedra) are combined with a symmetric aggregation, so the result
    does not depend on vertex order within a tetrahedron or on tetrahedron order.
    """

    def _build_features(self, config):
        vertex_layers = config["per_vertex_layers"]
        tet_layers = config["per_tetrahedron_layers"]
        pair_layers = config["per_two_tetrahedra_layers"]
        self.vertex_agg = config["vertices_aggregation_function"]
        self.tet_agg = config["tetrahedra_aggregation_function"]

        self.vertex_net = build_mlp([3] + vertex_layers, nn.ReLU(), self.dropout)
        self.vertex_shortcuts = nn.ModuleList([nn.Linear(3, vertex_layers[-1]) for _ in range(8)])
        self.tet_net = build_mlp([vertex_layers[-1]] + tet_layers, self.activation, self.dropout)
        self.tet_shortcut_1 = nn.Linear(12, tet_layers[-1])
        self.tet_shortcut_2 = nn.Linear(12, tet_layers[-1])
        extractor = nn.ModuleDict({
            "pair_net": build_mlp([tet_layers[-1]] + pair_layers, self.activation, self.dropout),
            "global_shortcut": nn.Linear(config["input_dim"], pair_layers[-1]),
        })
        return extractor, pair_layers[-1]

    def extract_features(self, x):
        if x.size(1) not in (12, 24):
            raise ValueError(f"Input must be 12 or 24 dims, got {x.size(1)}")

        tets = self._tetrahedron(x[:, :12], 0)
        if x.size(1) == 24:
            tets = AGGREGATIONS[self.tet_agg](torch.stack([tets, self._tetrahedron(x[:, 12:], 1)], dim=1))
        return self.feature_extractor["pair_net"](tets) + self.feature_extractor["global_shortcut"](x)

    def _tetrahedron(self, coords, index):
        vertices = coords.view(-1, 4, 3)
        features = torch.stack([
            self.vertex_net(vertices[:, i]) + self.vertex_shortcuts[index * 4 + i](vertices[:, i])
            for i in range(4)
        ], dim=1)
        shortcut = (self.tet_shortcut_1, self.tet_shortcut_2)[index]
        return self.tet_net(AGGREGATIONS[self.vertex_agg](features)) + shortcut(coords)


ARCHITECTURES = {"mlp": SimpleMLP, "tetrahedronpairnet": TetrahedronPairNet}


def build_model(model_config):
    """Build the architecture selected in `model_config` (the config.yaml `model_config` section)."""
    arch = model_config["architecture"]["use_model"].lower()
    common = model_config["common_parameters"]
    train_header = pd.read_csv(processed_split_path(model_config["processed_data_path"], "train"), nrows=0)
    return ARCHITECTURES[arch]({
        "input_dim": len(train_header.columns) - 2,  # last two columns are the targets
        "task": common["task"],
        "activation": common["activation_function"],
        "volume_scale_factor": common["volume_scale_factor"],
        "dropout_rate": common.get("dropout_rate", 0.0),
        **model_config["architecture"].get(arch, {}),
    })


class TorchScriptModel(nn.Module):
    """A loaded TorchScript model that also offers `predict`, like `BaseNet`."""

    def __init__(self, model, task, volume_scale_factor):
        super().__init__()
        self.model = model
        self.task = task
        self.volume_scale_factor = volume_scale_factor

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return postprocess(self.model(x), self.task, self.volume_scale_factor)
