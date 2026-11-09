"""Nine capital sleeves, cash at zero, adjusted-unit mark-to-market accounting."""

import numpy as np
import pandas as pd


def run_portfolio(prices, events, sectors, start, end_exclusive, cost_bps=5, accepted=None):
    calendar = prices[sectors[0]].index
    dates = calendar[(calendar >= start) & (calendar < end_exclusive)]
    if len(dates) == 0:
        raise ValueError("Empty portfolio evaluation window.")
    # Never enter a trade whose full exit is outside the reported period.
    candidates = events[(events.signal_date >= start) & (events.exit_date < end_exclusive)].copy()
    if accepted is not None:
        candidates = candidates[candidates.event_id.isin(set(accepted))]
    if candidates.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("More than one event per sleeve and decision date.")
    decisions = {(r.signal_date, r.ticker): r for r in candidates.itertuples()}
    cash = {t: 1 / len(sectors) for t in sectors}
    shares = {t: 0.0 for t in sectors}
    active, pending, ledger, daily = {}, {}, [], []
    cost, previous_equity = cost_bps / 10_000, 1.0
    for date in dates:
        traded_notional = 0.0
        for ticker in sectors:
            price = prices[ticker]
            opening = float(price.at[date, "adj_open"])
            if ticker in active and active[ticker]["event"].exit_date == date:
                item = active.pop(ticker)
                notional = shares[ticker] * opening
                cash[ticker] = notional * (1 - cost)
                traded_notional += notional
                ledger.append(dict(event_id=item["event"].event_id, ticker=ticker,
                                   signal_date=item["event"].signal_date, entry_date=item["event"].entry_date,
                                   exit_date=date, barrier=item["event"].barrier,
                                   net_return=cash[ticker] / item["initial_cash"] - 1))
                shares[ticker] = 0.0
            if ticker in pending:
                event = pending.pop(ticker)
                if event.entry_date != date or ticker in active:
                    raise AssertionError("Invalid entry schedule or overlapping position.")
                initial = cash[ticker]
                shares[ticker] = initial / (opening * (1 + cost))
                traded_notional += shares[ticker] * opening
                cash[ticker] = 0.0
                active[ticker] = dict(event=event, initial_cash=initial)
        invested = sum(shares[t] * prices[t].at[date, "adj_close"] for t in sectors)
        equity = sum(cash.values()) + invested
        daily.append(dict(Date=date, equity=equity, daily_return=equity / previous_equity - 1,
                          exposure=invested / equity, active_sleeves=len(active),
                          turnover=traded_notional / previous_equity))
        previous_equity = equity
        for ticker in sectors:
            # A signal observed while a sleeve is open is skipped even if it exits tomorrow.
            event = decisions.get((date, ticker))
            if event is not None and ticker not in active:
                pending[ticker] = event
    if active or pending:
        raise AssertionError("Unclosed portfolio positions at evaluation endpoint.")
    return pd.DataFrame(daily).set_index("Date"), pd.DataFrame(ledger)


def buy_and_hold(prices, tickers, start, end_exclusive, cost_bps=5):
    calendar = prices[tickers[0]].index
    dates = calendar[(calendar >= start) & (calendar < end_exclusive)]
    # Equal initial allocations, no rebalancing, entry fee and terminal liquidation fee.
    values = pd.DataFrame({t: prices[t].loc[dates, "adj_close"] / prices[t].at[dates[0], "adj_open"] for t in tickers})
    cost = cost_bps / 10_000
    equity = values.mean(axis=1) / (1 + cost)
    terminal_notional = float(equity.iloc[-1])
    previous_terminal_equity = 1.0 if len(equity) == 1 else float(equity.iloc[-2])
    equity.iloc[-1] *= 1 - cost
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] - 1
    result = pd.DataFrame({"equity": equity, "daily_return": returns, "exposure": 1.0, "turnover": 0.0}, index=dates)
    result.iloc[0, result.columns.get_loc("turnover")] = 1 / (1 + cost)
    result.iloc[-1, result.columns.get_loc("turnover")] += terminal_notional / previous_terminal_equity
    return result


def performance(daily, trades=None):
    returns = daily.daily_return
    std = returns.std(ddof=1)
    equity = daily.equity
    # Initial capital is included in the high-water mark.
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    return dict(total_return=float(equity.iloc[-1] - 1),
                annualized_return=float(equity.iloc[-1] ** (252 / len(equity)) - 1),
                sharpe=float(np.sqrt(252) * returns.mean() / std) if std > 0 else None,
                max_drawdown=float(drawdown.min()), mean_exposure=float(daily.exposure.mean()),
                time_invested=float(daily.exposure.gt(0).mean()),
                annualized_turnover=float(daily.turnover.sum() * 252 / len(daily)) if "turnover" in daily else None,
                trades=0 if trades is None else len(trades),
                mean_net_trade_return=None if trades is None or trades.empty else float(trades.net_return.mean()))
