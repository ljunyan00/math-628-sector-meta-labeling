"""Small matched-budget convolutional models and tabular comparators."""

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier

from .sequence_data import CONTEXT_COLUMNS, fit_pair_scaling, return_maps, standardize

MODELS = ("logistic", "xgboost", "cnn_1d", "cnn_2d")


class TradeCNN(nn.Module):
    def __init__(self, dimensions, context_size):
        super().__init__()
        if dimensions == 1:
            self.encoder = nn.Sequential(
                nn.Conv1d(2, 8, 3, padding=1), nn.ReLU(),
                nn.Conv1d(8, 12, 3, dilation=2, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            )
        self.head = nn.Sequential(nn.Linear(48 + context_size, 16), nn.ReLU(), nn.Dropout(0.15), nn.Linear(16, 1))

    def forward(self, sequences, context):
        return self.head(torch.cat([self.encoder(sequences), context], dim=1)).squeeze(-1)


def context_array(events, sectors):
    numeric = events[[column for column in CONTEXT_COLUMNS if column != "ticker"]].to_numpy(float)
    identity = np.stack([events.ticker.eq(ticker).to_numpy() for ticker in sectors], axis=1)
    return np.concatenate([numeric, identity], axis=1)


def cnn_predict(train_x, train_context, train_y, test_x, test_context, dimensions, spec):
    def tensor(values):
        return torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32))

    x, context, y = tensor(train_x), tensor(train_context), tensor(train_y)
    expected_x, expected_context = tensor(test_x), tensor(test_context)
    predictions = []
    for seed in spec["cnn_seeds"]:
        torch.manual_seed(seed)
        model = TradeCNN(dimensions, context.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec["cnn_learning_rate"], weight_decay=spec["cnn_weight_decay"],
        )
        objective = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(spec["cnn_epochs"]):
            for indices in torch.randperm(len(y)).split(spec["cnn_batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(x[indices], context[indices]), y[indices])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            probabilities = model(expected_x, expected_context).sigmoid().numpy()
        predictions.append(probabilities)
    return np.mean(predictions, axis=0)


def predict_fold(events, windows, returns, peers, train, validation, cfg, spec):
    last_train = events.signal_date.iloc[train].max()
    if last_train >= events.signal_date.iloc[validation].min():
        raise ValueError("Training signal dates must precede scoring dates.")
    scaling = fit_pair_scaling(returns, peers, cfg["sample_start"], last_train)
    joined = np.concatenate([train, validation])
    tickers = returns.columns.get_indexer(events.ticker.iloc[joined])
    sequences = standardize(windows[joined], tickers, scaling, spec["standardized_return_clip"])
    labels = events.label.iloc[train].to_numpy()
    context = context_array(events.iloc[joined], cfg["sectors"])
    split = len(train)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(context[:split])
    context = imputer.transform(context)
    scaling_context = StandardScaler().fit(context[:split])
    context = scaling_context.transform(context)
    train_context, validation_context = context[:split], context[split:]

    logistic = LogisticRegression(C=0.1, max_iter=500).fit(train_context, labels)
    xgboost = XGBClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.05, min_child_weight=10,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=5, tree_method="hist",
        random_state=cfg["seed"], n_jobs=2,
    ).fit(train_context, labels)
    maps = return_maps(sequences)
    return {
        "logistic": logistic.predict_proba(validation_context)[:, 1],
        "xgboost": xgboost.predict_proba(validation_context)[:, 1],
        "cnn_1d": cnn_predict(sequences[:split], train_context, labels, sequences[split:], validation_context, 1, spec),
        "cnn_2d": cnn_predict(maps[:split], train_context, labels, maps[split:], validation_context, 2, spec),
    }
