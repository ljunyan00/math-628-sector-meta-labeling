"""Global chronological splits with known exits and an eleven-session gap."""

import numpy as np
import pandas as pd


def purged_train(events, calendar, boundary, gap_sessions=11):
    boundary = pd.Timestamp(boundary)
    i = int(calendar.searchsorted(boundary))
    cutoff = calendar[max(0, i - gap_sessions)]
    # The gap applies to signal dates; actual label exits must also precede the refit.
    mask = (events.signal_date < cutoff) & (events.exit_date < boundary)
    return events.index[mask].to_numpy(), cutoff


def make_splits(events, calendar, cfg):
    result, metadata = {}, []
    for year in cfg["validation_years"]:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        train, cutoff = purged_train(events, calendar, start, cfg["boundary_gap_sessions"])
        # Development validation labels must resolve before the next annual boundary.
        # This keeps the 2021 development objective clear of final-test outcomes.
        validation = events.index[(events.signal_date >= start) & (events.signal_date < end) & (events.exit_date < end)].to_numpy()
        result[f"train_{year}"] = train
        result[f"validation_{year}"] = validation
        metadata.append(dict(fold=year, train_events=len(train), validation_events=len(validation),
                             train_signal_before=str(cutoff.date()), train_exit_before=str(start.date()),
                             validation_exit_before=str(end.date())))
    train, cutoff = purged_train(events, calendar, cfg["test_start"], cfg["boundary_gap_sessions"])
    result["final_train"] = train
    result["final_test"] = events.index[events.signal_date >= cfg["test_start"]].to_numpy()
    return result, pd.DataFrame(metadata)
