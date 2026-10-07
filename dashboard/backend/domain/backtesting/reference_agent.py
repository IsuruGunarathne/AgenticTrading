"""Deterministic rule-based reference trading agent.

Extracted (Phase 2C1) from ``PortfolioManager.make_trading_decision`` in
``dashboard/scripts/backtest_hourly_agent.py``. This is pure, deterministic,
domain-level decision logic over an explicit portfolio-state snapshot plus the
current positions/cash. The legacy method now delegates here.

Behavior is byte-for-byte identical to the original method. In particular:

* position sizing, BUY/SELL thresholds, reason strings, and the action schema
  are unchanged;
* symbols are iterated in ``portfolio_state["market_signals"]`` insertion order;
* a symbol is skipped when ``pd.isna([rsi, sma20]).any()`` (missing/None/NaN
  ``rsi`` or ``sma20``);
* BUY requires no existing position, ``rsi < 30`` and ``price < sma20``; size is
  ``int(total_equity * 0.02 / price)``, only emitted when ``shares_to_buy > 0``
  and ``shares_to_buy * price <= cash``;
* SELL (an ``elif``, so mutually exclusive with BUY) requires an existing
  position and (``rsi > 70`` or (``sma50`` truthy and ``price > sma50 * 1.02``));
  it sells the full held quantity;
* the return value is ``{"actions": [...]}`` and inputs are not mutated.

No thresholds, formulas, strings, or default quantities are changed. The LLM
decision workflow is intentionally NOT extracted here.

Fractional sizing (opt-in): stocks trade in whole shares, but crypto trades in
fractional units, and ``int(total_equity * 0.02 / price)`` is 0 whenever a unit
costs more than 2% of equity (e.g. BTC or ETH with a $10k portfolio) — so the
agent could never buy. With ``fractional=True`` the size is floored to
``FRACTIONAL_DECIMALS`` places instead of to a whole unit. The default
(``False``) keeps the original whole-share behavior exactly.

Position size (opt-in): each BUY spends ``position_fraction`` of total equity,
defaulting to ``DEFAULT_POSITION_FRACTION`` (0.02, the original 2%). 2% suits a
30-stock universe, but on a one- or few-pair crypto plan it caps exposure at
~2% per pair and leaves the rest idle in cash. Callers (the crypto backtester)
can pass a larger slot, e.g. ``1 / len(pairs)``. In fractional mode the spend is
also capped at available cash, so a slot isn't skipped just because prices
moved since the other slots were filled.

This module is domain-only: it must not import Anthropic, Alpaca, the database,
FastAPI, API routers, or scripts.
"""

import math
from typing import Dict, List

import pandas as pd

# Share of total equity each rule-based BUY spends (the original sizing).
DEFAULT_POSITION_FRACTION = 0.02

# Precision for fractional (crypto) order sizes. Flooring, not rounding, so the
# order can never cost more than the 2% risk budget.
FRACTIONAL_DECIMALS = 8


def _position_size(risk_amount: float, price: float, fractional: bool):
    if not fractional:
        return int(risk_amount / price)
    scale = 10 ** FRACTIONAL_DECIMALS
    return math.floor(risk_amount / price * scale) / scale


def make_rule_based_decision(
    *,
    portfolio_state: Dict,
    positions: Dict,
    cash: float,
    fractional: bool = False,
    position_fraction: float = DEFAULT_POSITION_FRACTION,
) -> Dict:
    """Produce rule-based trading actions for the given portfolio state.

    ``portfolio_state`` must provide ``total_equity`` and ``market_signals`` (a
    mapping of symbol -> indicator dict). ``positions`` and ``cash`` reflect the
    current holdings and available cash. ``fractional`` sizes buys in fractional
    units (crypto) instead of whole shares; ``position_fraction`` is the share of
    total equity each buy spends (default 2%). Inputs are read only, never mutated.
    Returns ``{"actions": [...]}`` with the same action dictionaries the original
    method produced.
    """
    if not 0 < position_fraction <= 1:
        raise ValueError(f"position_fraction must be in (0, 1], got {position_fraction}")

    actions: List[Dict] = []

    # Calculate total portfolio equity for consistent position sizing
    total_equity = portfolio_state["total_equity"]

    for symbol, signal in portfolio_state["market_signals"].items():
        rsi = signal.get("rsi")
        price = signal.get("price")
        sma20 = signal.get("sma20")
        sma50 = signal.get("sma50")

        # Skip if indicators not ready
        if pd.isna([rsi, sma20]).any():
            continue

        has_position = symbol in positions and positions[symbol] > 0

        # BUY logic: RSI < 30 (oversold)
        if not has_position and rsi < 30 and price < sma20:
            # Size: position_fraction (default 2%) of TOTAL PORTFOLIO per trade
            risk_amount = total_equity * position_fraction
            if fractional:
                risk_amount = min(risk_amount, cash)
            shares_to_buy = _position_size(risk_amount, price, fractional)
            if shares_to_buy > 0 and shares_to_buy * price <= cash:
                actions.append({
                    "symbol": symbol,
                    "action": "buy",
                    "shares": shares_to_buy,
                    "reason": f"RSI oversold ({rsi:.0f}), price below MA"
                })

        # SELL logic: RSI > 70 (overbought) or price above SMA50
        elif has_position and (rsi > 70 or (sma50 and price > sma50 * 1.02)):
            actions.append({
                "symbol": symbol,
                "action": "sell",
                "shares": positions[symbol],
                "reason": f"RSI overbought ({rsi:.0f})" if rsi > 70 else "Price above MA50"
            })

    return {"actions": actions}
