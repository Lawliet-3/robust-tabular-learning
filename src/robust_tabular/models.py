from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class ModelUnavailableError(ImportError):
    pass


def _columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = [name for name in X.columns if pd.api.types.is_numeric_dtype(X[name])]
    categorical = [name for name in X.columns if name not in numerical]
    return numerical, categorical


def _preprocessor(X: pd.DataFrame, *, one_hot: bool) -> ColumnTransformer:
    numerical, categorical = _columns(X)
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if not one_hot:
        numeric_steps.append(("scale", StandardScaler()))
    if one_hot:
        category_encoder: Any = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
    else:
        category_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
        )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numerical),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", category_encoder),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _optional_class(module: str, class_name: str) -> type:
    try:
        return getattr(importlib.import_module(module), class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelUnavailableError(
            f"{class_name} is unavailable; install the corresponding optional dependency"
        ) from exc


def build_model(
    name: str,
    task: str,
    X: pd.DataFrame,
    seed: int,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    params = dict(params or {})
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task: {task}")
    if name == "xgboost":
        class_name = "XGBClassifier" if task == "classification" else "XGBRegressor"
        cls = _optional_class("xgboost", class_name)
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_jobs": 1,
            "random_state": seed,
        }
        if task == "classification":
            defaults["eval_metric"] = "logloss"
    elif name == "lightgbm":
        class_name = "LGBMClassifier" if task == "classification" else "LGBMRegressor"
        cls = _optional_class("lightgbm", class_name)
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "n_jobs": 1,
            "random_state": seed,
            "verbosity": -1,
        }
    elif name == "catboost":
        class_name = "CatBoostClassifier" if task == "classification" else "CatBoostRegressor"
        cls = _optional_class("catboost", class_name)
        defaults = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "thread_count": 1,
            "random_seed": seed,
            "verbose": False,
        }
    elif name == "ft_transformer":
        cls = FTTransformerClassifier if task == "classification" else FTTransformerRegressor
        defaults = {"random_state": seed}
    elif name == "tabpfn":
        class_name = "TabPFNClassifier" if task == "classification" else "TabPFNRegressor"
        cls = _optional_class("tabpfn", class_name)
        defaults = {"random_state": seed}
    elif name == "hist_gradient_boosting":
        cls = (
            HistGradientBoostingClassifier
            if task == "classification"
            else HistGradientBoostingRegressor
        )
        defaults = {"max_iter": 150, "random_state": seed}
    else:
        raise ValueError(f"Unknown model: {name}")
    defaults.update(params)
    estimator = cls(**defaults)
    return Pipeline(
        [
            ("preprocess", _preprocessor(X, one_hot=name in {"xgboost", "lightgbm", "catboost"})),
            ("model", estimator),
        ]
    )


class _FTTransformerBase(BaseEstimator):
    def __init__(
        self,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 30,
        weight_decay: float = 1e-5,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _fit_torch(
        self, X: np.ndarray, y: np.ndarray, classification: bool
    ) -> _FTTransformerBase:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ModelUnavailableError(
                "FT-Transformer requires `pip install -e '.[neural]'`"
            ) from exc
        if self.d_token % self.n_heads:
            raise ValueError("d_token must be divisible by n_heads")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array) if classification else None
        if classification:
            class_to_index = {value: index for index, value in enumerate(self.classes_)}
            y_array = np.asarray([class_to_index[value] for value in y_array], dtype=np.int64)
            output_dim = len(self.classes_)
            y_tensor = torch.tensor(y_array, dtype=torch.long)
        else:
            output_dim = 1
            self.y_mean_ = float(np.mean(y_array))
            self.y_scale_ = float(np.std(y_array)) or 1.0
            normalized_y = (y_array.astype(np.float32) - self.y_mean_) / self.y_scale_
            y_tensor = torch.tensor(normalized_y, dtype=torch.float32).reshape(-1, 1)

        class Network(nn.Module):
            def __init__(network_self: Any) -> None:
                super().__init__()
                network_self.weight = nn.Parameter(torch.empty(X_array.shape[1], self.d_token))
                network_self.bias = nn.Parameter(torch.empty(X_array.shape[1], self.d_token))
                network_self.cls = nn.Parameter(torch.zeros(1, 1, self.d_token))
                nn.init.xavier_uniform_(network_self.weight)
                nn.init.normal_(network_self.bias, std=0.01)
                layer = nn.TransformerEncoderLayer(
                    d_model=self.d_token,
                    nhead=self.n_heads,
                    dim_feedforward=self.d_token * 4,
                    dropout=self.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                network_self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
                network_self.norm = nn.LayerNorm(self.d_token)
                network_self.head = nn.Linear(self.d_token, output_dim)

            def forward(network_self: Any, features: Any) -> Any:
                tokens = features.unsqueeze(-1) * network_self.weight + network_self.bias
                cls_token = network_self.cls.expand(features.shape[0], -1, -1)
                encoded = network_self.encoder(torch.cat([cls_token, tokens], dim=1))
                return network_self.head(network_self.norm(encoded[:, 0]))

        self.model_ = Network().to(device)
        self.device_ = device
        loader = DataLoader(
            TensorDataset(torch.tensor(X_array), y_tensor),
            batch_size=min(self.batch_size, len(X_array)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss() if classification else nn.MSELoss()
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch_X)
            if self.verbose:
                print(f"epoch={epoch + 1} loss={total_loss / len(X_array):.6f}")
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        with torch.inference_mode():
            tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
            return self.model_(tensor).cpu().numpy()


class FTTransformerClassifier(_FTTransformerBase, ClassifierMixin):
    def fit(self, X: np.ndarray, y: np.ndarray) -> FTTransformerClassifier:
        return self._fit_torch(X, y, classification=True)  # type: ignore[return-value]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class FTTransformerRegressor(_FTTransformerBase, RegressorMixin):
    def fit(self, X: np.ndarray, y: np.ndarray) -> FTTransformerRegressor:
        return self._fit_torch(X, y, classification=False)  # type: ignore[return-value]

    def predict(self, X: np.ndarray) -> np.ndarray:
        normalized = self._logits(X).reshape(-1)
        return normalized * self.y_scale_ + self.y_mean_
