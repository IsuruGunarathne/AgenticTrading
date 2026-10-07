"""Characterization tests for the extracted rule-based agent (Phase 2C1).

Locks in the exact behavior of
``dashboard.backend.domain.backtesting.reference_agent.make_rule_based_decision``
and the legacy ``PortfolioManager.make_trading_decision`` that delegates to it.
Imports use the canonical package path; no external services are touched.
"""

import copy

import numpy as np

from dashboard.backend.domain.backtesting.reference_agent import (
    make_rule_based_decision,
)
from dashboard.scripts import backtest_hourly_agent as bha


def _signal(price=None, rsi=None, sma20=None, sma50=None):
    return {"price": price, "rsi": rsi, "sma20": sma20, "sma50": sma50}


def _state(total_equity=100000, signals=None):
    return {
        "total_equity": total_equity,
        "market_signals": signals or {},
    }


def _decide(state, positions=None, cash=100000):
    return make_rule_based_decision(
        portfolio_state=state,
        positions=dict(positions or {}),
        cash=cash,
    )


# ---------------------------------------------------------------------------
# Empty / incomplete input
# ---------------------------------------------------------------------------

def test_empty_market_data():
    assert _decide(_state(signals={})) == {"actions": []}


def test_none_indicators_skipped():
    state = _state(signals={"AAPL": _signal(price=100, rsi=None, sma20=None)})
    assert _decide(state) == {"actions": []}


def test_nan_indicators_skipped():
    state = _state(signals={"AAPL": _signal(price=100, rsi=np.nan, sma20=120)})
    assert _decide(state) == {"actions": []}


def test_missing_sma20_skipped():
    # sma20 None -> pd.isna true -> skipped
    state = _state(signals={"AAPL": _signal(price=100, rsi=20, sma20=None)})
    assert _decide(state) == {"actions": []}


def test_rsi_present_sma20_present_but_no_signal_holds():
    # neutral indicators -> no action (HOLD == empty action list)
    state = _state(signals={"AAPL": _signal(price=100, rsi=50, sma20=100, sma50=100)})
    assert _decide(state) == {"actions": []}


# ---------------------------------------------------------------------------
# BUY behavior
# ---------------------------------------------------------------------------

def test_buy_triggered_exact_action():
    # rsi < 30 and price < sma20, no position
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),
    })
    out = _decide(state, positions={}, cash=100000)
    # shares = int(100000 * 0.02 / 100) = int(20) = 20 ; 20*100=2000 <= cash
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "buy",
        "shares": 20,
        "reason": "RSI oversold (25), price below MA",
    }]}


def test_buy_fails_when_price_not_below_sma20():
    state = _state(signals={"AAPL": _signal(price=120, rsi=25, sma20=110)})
    assert _decide(state) == {"actions": []}


def test_buy_fails_when_rsi_not_oversold():
    state = _state(signals={"AAPL": _signal(price=100, rsi=35, sma20=110)})
    assert _decide(state) == {"actions": []}


def test_buy_skipped_when_existing_position():
    state = _state(signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    # has_position True -> BUY branch not taken; SELL branch needs rsi>70 / price>sma50*1.02
    assert _decide(state, positions={"AAPL": 5}) == {"actions": []}


def test_buy_skipped_when_zero_shares_to_buy():
    # tiny equity -> int(risk/price) == 0 -> no action
    state = _state(total_equity=100, signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    assert _decide(state, cash=100000) == {"actions": []}


def test_buy_skipped_when_insufficient_cash():
    state = _state(total_equity=100000, signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    # shares=20, cost=2000 > cash=1000 -> skipped
    assert _decide(state, cash=1000) == {"actions": []}


def test_buy_share_quantity_uses_int_truncation():
    # total_equity*0.02/price = 100000*0.02/150 = 13.33 -> int -> 13
    state = _state(total_equity=100000, signals={"AAPL": _signal(price=150, rsi=10, sma20=200)})
    out = _decide(state, cash=100000)
    assert out["actions"][0]["shares"] == 13
    assert isinstance(out["actions"][0]["shares"], int)


# ---------------------------------------------------------------------------
# SELL behavior
# ---------------------------------------------------------------------------

def test_sell_triggered_by_overbought_rsi():
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    out = _decide(state, positions={"AAPL": 7})
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "sell",
        "shares": 7,
        "reason": "RSI overbought (75)",
    }]}


def test_sell_triggered_by_price_above_sma50():
    # rsi not > 70, but price > sma50 * 1.02
    state = _state(signals={"AAPL": _signal(price=105, rsi=50, sma20=90, sma50=100)})
    out = _decide(state, positions={"AAPL": 3})
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "sell",
        "shares": 3,
        "reason": "Price above MA50",
    }]}


def test_sell_not_triggered_when_price_just_below_threshold():
    # price == sma50 * 1.02 exactly -> not strictly greater -> no sell
    state = _state(signals={"AAPL": _signal(price=102, rsi=50, sma20=90, sma50=100)})
    assert _decide(state, positions={"AAPL": 3}) == {"actions": []}


def test_sell_requires_position():
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    assert _decide(state, positions={}) == {"actions": []}


def test_sell_zero_position_treated_as_no_position():
    # positions[symbol] == 0 -> has_position False -> not a sell; BUY needs rsi<30
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    assert _decide(state, positions={"AAPL": 0}) == {"actions": []}


def test_sell_sma50_falsy_only_rsi_path():
    # sma50 == 0 (falsy) -> price>sma50*1.02 branch short-circuits; needs rsi>70
    state = _state(signals={"AAPL": _signal(price=100, rsi=50, sma20=90, sma50=0)})
    assert _decide(state, positions={"AAPL": 3}) == {"actions": []}


# ---------------------------------------------------------------------------
# Multiple symbols / ordering / determinism
# ---------------------------------------------------------------------------

def test_multiple_symbols_buy_and_sell_order_preserved():
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),  # buy
        "MSFT": _signal(price=100, rsi=80, sma20=90, sma50=80),    # sell (held)
    })
    out = _decide(state, positions={"MSFT": 4}, cash=100000)
    assert [a["symbol"] for a in out["actions"]] == ["AAPL", "MSFT"]
    assert out["actions"][0]["action"] == "buy"
    assert out["actions"][1]["action"] == "sell"


def test_one_valid_one_incomplete_symbol():
    state = _state(signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110),     # buy
        "MSFT": _signal(price=100, rsi=None, sma20=None),  # skipped
    })
    out = _decide(state, cash=100000)
    assert [a["symbol"] for a in out["actions"]] == ["AAPL"]


def test_deterministic_repeated_calls():
    state = _state(signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    a = _decide(state, cash=100000)
    b = _decide(state, cash=100000)
    assert a == b


# ---------------------------------------------------------------------------
# Inputs not mutated
# ---------------------------------------------------------------------------

def test_inputs_not_mutated():
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),
    })
    positions = {"MSFT": 5}
    state_copy = copy.deepcopy(state)
    positions_copy = copy.deepcopy(positions)
    make_rule_based_decision(portfolio_state=state, positions=positions, cash=100000)
    assert state == state_copy
    assert positions == positions_copy


# ---------------------------------------------------------------------------
# Legacy equivalence + subclass compatibility
# ---------------------------------------------------------------------------

def _golden_state():
    return _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),  # buy
        "MSFT": _signal(price=105, rsi=50, sma20=90, sma50=100),   # sell if held
        "JPM": _signal(price=100, rsi=None, sma20=None),           # skipped
    })


def test_legacy_matches_canonical():
    state = _golden_state()
    pm = bha.PortfolioManager(100000)
    pm.positions = {"MSFT": 4}
    pm.cash = 100000

    legacy = pm.make_trading_decision(copy.deepcopy(state))
    canon = make_rule_based_decision(
        portfolio_state=copy.deepcopy(state),
        positions={"MSFT": 4},
        cash=100000,
    )
    assert legacy == canon


def test_legacy_golden_exact():
    pm = bha.PortfolioManager(100000)
    pm.positions = {"MSFT": 4}
    out = pm.make_trading_decision(_golden_state())
    assert out == {"actions": [
        {"symbol": "AAPL", "action": "buy", "shares": 20,
         "reason": "RSI oversold (25), price below MA"},
        {"symbol": "MSFT", "action": "sell", "shares": 4,
         "reason": "Price above MA50"},
    ]}


def test_subclass_inherits_make_trading_decision():
    class MyPM(bha.PortfolioManager):
        def custom_method(self):
            return "ok"

    pm = MyPM(100000)
    out = pm.make_trading_decision(_state(signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110),
    }))
    assert out["actions"][0]["action"] == "buy"
    assert pm.custom_method() == "ok"
    assert MyPM.make_trading_decision is bha.PortfolioManager.make_trading_decision


# ---------------------------------------------------------------------------
# Fractional sizing (crypto)
# ---------------------------------------------------------------------------

def _decide_fractional(state, positions=None, cash=10000):
    return make_rule_based_decision(
        portfolio_state=state,
        positions=dict(positions or {}),
        cash=cash,
        fractional=True,
    )


def test_whole_share_default_cannot_buy_expensive_unit():
    # The bug: $10k equity -> $200 budget; a BTC-priced unit floors to 0 shares.
    state = _state(total_equity=10000, signals={
        "BTCUSDT": _signal(price=86614.0, rsi=25, sma20=90000),
    })
    assert _decide(state, cash=10000) == {"actions": []}


def test_fractional_buys_expensive_unit():
    state = _state(total_equity=10000, signals={
        "BTCUSDT": _signal(price=86614.0, rsi=25, sma20=90000),
    })
    actions = _decide_fractional(state)["actions"]
    assert len(actions) == 1
    a = actions[0]
    assert a["symbol"] == "BTCUSDT" and a["action"] == "buy"
    assert a["reason"] == "RSI oversold (25), price below MA"
    assert a["shares"] == 0.00230909  # floor(200 / 86614, 8 dp)
    assert 0 < a["shares"] * 86614.0 <= 200


def test_fractional_floors_never_exceeds_budget():
    for price in (3.0, 7.0, 0.3333, 86614.0, 123456.789):
        state = _state(total_equity=10000, signals={
            "X": _signal(price=price, rsi=10, sma20=price * 2),
        })
        shares = _decide_fractional(state)["actions"][0]["shares"]
        assert shares * price <= 200 + 1e-9
        assert round(shares, 8) == shares


def test_fractional_still_respects_cash():
    # Fractional mode caps the spend at available cash: $100 left against a
    # $200 budget buys $100 worth (never more than cash); no cash, no buy.
    state = _state(total_equity=10000, signals={
        "BTCUSDT": _signal(price=86614.0, rsi=25, sma20=90000),
    })
    shares = _decide_fractional(state, cash=100)["actions"][0]["shares"]
    assert shares == 0.00115454 and shares * 86614.0 <= 100
    assert _decide_fractional(state, cash=0) == {"actions": []}


def test_fractional_sell_logic_unchanged():
    state = _state(total_equity=10000, signals={
        "BTCUSDT": _signal(price=90000.0, rsi=75, sma20=85000),
    })
    out = _decide_fractional(state, positions={"BTCUSDT": 0.0023})
    assert out == {"actions": [{
        "symbol": "BTCUSDT", "action": "sell", "shares": 0.0023,
        "reason": "RSI overbought (75)",
    }]}


def test_portfolio_manager_fractional_flag():
    state = _state(total_equity=10000, signals={
        "BTCUSDT": _signal(price=86614.0, rsi=25, sma20=90000),
    })
    whole = bha.PortfolioManager(10000)
    frac = bha.PortfolioManager(10000, fractional_units=True)
    assert whole.make_trading_decision(state) == {"actions": []}
    assert frac.make_trading_decision(state)["actions"][0]["shares"] == 0.00230909


def test_crypto_backtester_uses_fractional_units(monkeypatch):
    # The crypto backtester must build its PortfolioManager with
    # fractional_units=True (the stock backtester keeps whole shares).
    from dashboard.scripts import backtest_crypto_agent as bca

    seen = {}
    real_pm = bca.PortfolioManager

    class SpyPM(real_pm):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bca, "PortfolioManager", SpyPM)
    bt = bca.CryptoBacktester("2026-01-01", "2026-01-02", pairs=["BTCUSDT"])
    bt.all_data = {}
    try:
        bt.run_agent_backtest()
    except Exception:
        pass  # empty data is fine — only the constructor call matters here
    assert seen.get("fractional_units") is True


# ---------------------------------------------------------------------------
# Position size (position_fraction)
# ---------------------------------------------------------------------------

import pytest


def _btc_state(equity=10000, price=80000.0):
    return _state(total_equity=equity, signals={
        "BTCUSDT": _signal(price=price, rsi=25, sma20=price * 1.1),
    })


def test_position_fraction_default_is_original_two_percent():
    out = make_rule_based_decision(
        portfolio_state=_btc_state(), positions={}, cash=10000, fractional=True,
    )
    assert out["actions"][0]["shares"] == 0.0025  # $200 / $80,000


def test_position_fraction_scales_order():
    out = make_rule_based_decision(
        portfolio_state=_btc_state(), positions={}, cash=10000,
        fractional=True, position_fraction=0.5,
    )
    assert out["actions"][0]["shares"] == 0.0625  # $5,000 / $80,000


def test_full_allocation_single_pair():
    out = make_rule_based_decision(
        portfolio_state=_btc_state(), positions={}, cash=10000,
        fractional=True, position_fraction=1.0,
    )
    shares = out["actions"][0]["shares"]
    assert shares == 0.125 and shares * 80000.0 <= 10000


def test_fractional_slot_capped_at_cash():
    # Equity 10k but only 6k cash left: the 50% slot ($5k) fits; a 100% slot
    # ($10k) is capped to the $6k available instead of being skipped.
    out = make_rule_based_decision(
        portfolio_state=_btc_state(), positions={}, cash=6000,
        fractional=True, position_fraction=1.0,
    )
    shares = out["actions"][0]["shares"]
    assert shares == 0.075 and shares * 80000.0 <= 6000


def test_whole_share_mode_not_cash_capped():
    # Stock (whole-share) behavior is unchanged: an order that costs more than
    # cash is skipped, not shrunk.
    state = _state(total_equity=100000, signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    out = make_rule_based_decision(
        portfolio_state=state, positions={}, cash=1000, position_fraction=0.5,
    )
    assert out == {"actions": []}


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_position_fraction_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        make_rule_based_decision(
            portfolio_state=_btc_state(), positions={}, cash=10000, position_fraction=bad,
        )


def test_portfolio_manager_passes_position_fraction():
    pm = bha.PortfolioManager(10000, fractional_units=True, position_fraction=0.5)
    assert pm.make_trading_decision(_btc_state())["actions"][0]["shares"] == 0.0625
    assert bha.PortfolioManager(10000).position_fraction == 0.02


def test_crypto_backtester_position_fraction_defaults_to_equal_slots():
    from dashboard.scripts import backtest_crypto_agent as bca

    one = bca.CryptoBacktester("2026-01-01", "2026-01-02", pairs=["BTCUSDT"])
    four = bca.CryptoBacktester("2026-01-01", "2026-01-02",
                                pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    custom = bca.CryptoBacktester("2026-01-01", "2026-01-02", pairs=["BTCUSDT"],
                                  position_fraction=0.3)
    assert one.position_fraction == 1.0
    assert four.position_fraction == 0.25
    assert custom.position_fraction == 0.3
    with pytest.raises(ValueError):
        bca.CryptoBacktester("2026-01-01", "2026-01-02", pairs=["BTCUSDT"], position_fraction=2)


def test_crypto_backtester_passes_position_fraction(monkeypatch):
    from dashboard.scripts import backtest_crypto_agent as bca

    seen = {}
    real_pm = bca.PortfolioManager

    class SpyPM(real_pm):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bca, "PortfolioManager", SpyPM)
    bt = bca.CryptoBacktester("2026-01-01", "2026-01-02", pairs=["BTCUSDT", "ETHUSDT"])
    bt.all_data = {}
    try:
        bt.run_agent_backtest()
    except Exception:
        pass  # empty data is fine — only the constructor call matters here
    assert seen.get("position_fraction") == 0.5
    assert seen.get("fractional_units") is True
