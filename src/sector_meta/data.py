"""Acquisition, fixed-vintage provenance, and strict input checks."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from io import StringIO
import json
from pathlib import Path
import platform
import shutil
from urllib.request import urlopen

import numpy as np
import pandas as pd


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "yfinance", "matplotlib", "pytest"]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def acquire(root, cfg, factors=None, refresh=False):
    import yfinance as yf

    raw = Path(root) / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    settings = dict(start=cfg["download_start"], end=cfg["end_exclusive"],
                    interval="1d", auto_adjust=False, back_adjust=False,
                    actions=True, repair=False, keepna=True)
    if manifest.get("download_settings") not in (None, settings) and not refresh:
        raise ValueError("Download settings changed; use --refresh in a separate data vintage.")

    def record(path, source, **extra):
        manifest["files"][path.name] = dict(
            sha256=checksum(path), source=source,
            acquired_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        manifest.update(download_settings=settings, versions=versions())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for ticker in cfg["sectors"] + [cfg["market"]]:
        path = raw / f"{ticker}.csv"
        if path.exists() and not refresh:
            if path.name not in manifest["files"] or checksum(path) != manifest["files"][path.name]["sha256"]:
                raise ValueError(f"Untracked or changed cache: {path}")
            continue
        frame = yf.Ticker(ticker).history(**settings, raise_errors=True)
        if frame.empty or "Adj Close" not in frame:
            raise ValueError(f"No adjusted price history for {ticker}; no synthetic fallback is used.")
        frame.index = frame.index.tz_localize(None).normalize()
        frame.index.name = "Date"
        frame.to_csv(path)
        record(path, f"Yahoo Finance via yfinance: {ticker}")
    path = raw / "VIXCLS.csv"
    if not path.exists() or refresh:
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
               f"&cosd={cfg['download_start']}&coed=2024-12-31")
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        pd.read_csv(StringIO(payload.decode())).to_csv(path, index=False)
        record(path, url)
    if factors is not None:
        path = raw / "F-F_Research_Data_Factors_weekly.csv"
        if not path.exists() or refresh:
            shutil.copyfile(factors, path)
            record(path, str(Path(factors).resolve()), vintage="202412 CRSP database")
    return manifest


def read_prices(raw, ticker):
    frame = pd.read_csv(Path(raw) / f"{ticker}.csv", index_col="Date", parse_dates=True)
    # All execution OHLC use the SAME total-return adjustment factor. Never add dividends again.
    factor = frame["Adj Close"] / frame["Close"]
    for col in ["Open", "High", "Low", "Close"]:
        frame[f"adj_{col.lower()}"] = frame[col] * factor
    return frame


def audit_and_load(root, cfg):
    raw = Path(root) / "data/raw"
    manifest = json.loads((raw / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        if checksum(raw / name) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    tickers = cfg["sectors"] + [cfg["market"]]
    required = [f"{t}.csv" for t in tickers] + ["VIXCLS.csv"]
    if set(required) - manifest["files"].keys():
        raise ValueError("Manifest does not cover all required data files.")
    prices = {t: read_prices(raw, t) for t in tickers}
    calendar = prices[cfg["market"]].index
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("SPY calendar must be unique and sorted.")
    calendar = calendar[(calendar >= cfg["download_start"]) & (calendar < cfg["end_exclusive"])]
    rows = []
    for ticker, frame in prices.items():
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted dates: {ticker}")
        missing = len(calendar.difference(frame.index))
        extra = len(frame.index.difference(calendar))
        frame = frame.reindex(calendar)
        columns = ["Open", "High", "Low", "Close", "Adj Close", "adj_open", "adj_close", "Volume"]
        values = frame[columns]
        bad = (~np.isfinite(values) | values.le(0)).any(axis=1)
        invalid_ohlc = (frame.High < frame[["Open", "Close", "Low"]].max(axis=1)) | (
            frame.Low > frame[["Open", "Close", "High"]].min(axis=1))
        rows.append(dict(ticker=ticker, first_date=str(frame.index.min().date()),
                         last_date=str(frame.index.max().date()), sessions=len(frame),
                         missing_spy_sessions=missing, extra_sessions=extra,
                         invalid_price_volume_rows=int(bad.sum()),
                         invalid_ohlc_rows=int(invalid_ohlc.sum()),
                         dividend_rows=int(frame.get("Dividends", pd.Series(dtype=float)).gt(0).sum()),
                         split_rows=int(frame.get("Stock Splits", pd.Series(dtype=float)).gt(0).sum())))
        if missing or extra or bad.any() or invalid_ohlc.any():
            raise ValueError(f"Price audit failed: {rows[-1]}")
        prices[ticker] = frame
    # Catch truncated downloads even if all tickers were truncated identically.
    if calendar.min() > pd.Timestamp(cfg["download_start"]) + pd.Timedelta(days=7):
        raise ValueError("Missing requested warm-up history.")
    if calendar.max() < pd.Timestamp(cfg["end_exclusive"]) - pd.Timedelta(days=7):
        raise ValueError("Missing requested sample endpoint.")
    vix_raw = pd.read_csv(raw / "VIXCLS.csv")
    vix = pd.Series(pd.to_numeric(vix_raw["VIXCLS"], errors="coerce").to_numpy(),
                    index=pd.to_datetime(vix_raw.iloc[:, 0]), name="vix")
    if vix.index.has_duplicates or not vix.index.is_monotonic_increasing:
        raise ValueError("VIX dates must be unique and sorted.")
    return prices, calendar, vix, pd.DataFrame(rows), manifest


def read_factors(path):
    """Parse the actual weekly endpoints; never synthesize Friday dates."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 5 and len(cells[0]) == 8 and cells[0].isdigit():
            rows.append([pd.to_datetime(cells[0], format="%Y%m%d")] + [float(x) / 100 for x in cells[1:]])
    frame = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"]).set_index("Date")
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Invalid weekly factor file.")
    return frame
