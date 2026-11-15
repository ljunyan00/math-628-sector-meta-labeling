"""Render the development revision diagnostics from published aggregate results."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "cnn_1d": "1D CNN",
    "cnn_2d": "Original 2D CNN",
    "cnn_2d_signed": "Signed 2D CNN",
    "ridge_return": "Net-return ridge",
}


def render(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(directory.parents[1] / "output/research/.mplconfig"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    forecast = pd.read_csv(directory / "forecast_metrics.csv")
    loss = forecast.pivot(index="model", columns="probabilities", values="log_loss")
    prior = loss.at["training_prior", "raw"]
    scores = loss.loc[list(LABELS)[:-1]] - prior
    trading = pd.read_csv(directory / "portfolio_metrics.csv").pivot(index="model", columns="policy", values="coverage")
    trading = trading.loc[list(LABELS)]
    fig = Figure(figsize=(11.7, 5.1), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    colors = ("#225C78", "#BD683D")
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Development replay  ·  2016–2021", fontsize=17, fontweight="bold")

    y = np.arange(len(scores))
    axes[0].scatter(scores.raw, y - .11, c=colors[0], marker="o", s=51, label="Raw")
    axes[0].scatter(scores.shrunk, y + .11, c=colors[1], marker="D", s=35, label="Shrunk toward prior")
    axes[0].set_yticks(y, [LABELS[key] for key in scores.index])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#65747B", lw=1, ls="--")
    axes[0].set_title("Probability forecasts", loc="left", fontsize=13, pad=11)
    axes[0].set_xlabel("Log loss minus training prior (lower is better)")
    axes[0].xaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncols=2, frameon=False)

    y = np.arange(len(trading))
    axes[1].scatter(trading.historical_cutoff, y - .11, c=colors[0], marker="o", s=51, label="Prior-years cutoff")
    axes[1].scatter(trading.matched_retrospective, y + .11, c=colors[1], marker="D", s=35, label="Retrospective match")
    axes[1].set_yticks(y, [LABELS[key] for key in trading.index])
    axes[1].invert_yaxis()
    axes[1].axvline(.30, color="#65747B", lw=1, ls="--")
    axes[1].set_title("Candidate coverage", loc="left", fontsize=13, pad=11)
    axes[1].set_xlabel("Accepted opportunities as a share of candidates")
    axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncols=2, frameon=False)
    fig.savefig(directory / "development_diagnostics.png", dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    render(parser.parse_args().root.resolve() / "reports/revision")
