"""Create a research illustration from past-only development inputs."""

import os

import numpy as np
import pandas as pd

from .sequence_data import event_inputs, fit_pair_scaling, return_maps, standardize
from .splits import purged_train


def create_input_figure(root, prices, calendar, vix, cfg, spec):
    out = root / "output/research"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    events, windows, returns, peers = event_inputs(
        prices, calendar, vix, cfg, spec, cfg["development_end_exclusive"],
    )
    train, _ = purged_train(events, returns.index, cfg["initial_train_end"], cfg["boundary_gap_sessions"])
    scaling = fit_pair_scaling(returns, peers, cfg["sample_start"], events.signal_date.iloc[train].max())
    # Use a fixed reference date and select the latest observed XLK opportunity at or before it.
    eligible = events.loc[(events.ticker == "XLK") & (events.signal_date <= pd.Timestamp("2020-04-01"))]
    index = eligible.index[-1]
    example = events.loc[index]
    scaled = standardize(windows[[index]], [returns.columns.get_loc("XLK")], scaling, spec["standardized_return_clip"])
    maps = return_maps(scaled)[0]
    correlation = returns.loc[:example.signal_date].tail(60).corr().to_numpy()

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    panels = [
        ("Absolute difference", maps[0], "viridis", 0, 8),
        ("Signed product", maps[1], "RdBu_r", -16, 16),
        ("Nine-sector correlation", correlation, "RdBu_r", -1, 1),
    ]
    for i, (title, values, cmap, lo, hi) in enumerate(panels):
        artist = axes[i].imshow(values, cmap=cmap, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        axes[i].set_title(title)
        fig.colorbar(artist, ax=axes[i], shrink=0.73, extend="both" if i == 1 else ("max" if i == 0 else "neither"))
        if i < 2:
            length = spec["window_sessions"]
            ticks = sorted({0, length // 2 - 1, length - 1})
            axes[i].set_xticks(ticks, [t + 1 for t in ticks])
            axes[i].set_yticks(ticks, [t + 1 for t in ticks])
            axes[i].set_xlabel("Peer basket session")
            axes[i].set_ylabel("XLK session")
        else:
            ticks = np.arange(len(returns.columns))
            axes[i].set_xticks(ticks, returns.columns, rotation=45, ha="right", fontsize=8)
            axes[i].set_yticks(ticks, returns.columns, fontsize=8)
    fig.suptitle(f"XLK opportunity at the {example.signal_date:%Y-%m-%d} close", fontsize=15)
    path = out / "cnn_input_examples.png"
    fig.savefig(path, dpi=160)
    print(f"{len(events):,} development events. Example figure: {path}")
    return path
