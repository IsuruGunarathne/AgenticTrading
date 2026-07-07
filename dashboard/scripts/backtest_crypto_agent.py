#!/usr/bin/env python3
"""
Crypto Backtest with Agent Decision Making (Binance data)

The crypto counterpart to ``backtest_hourly_agent.py``. An agent manages a
$10,000 USDT portfolio across the top-10 Binance USDT pairs. Each bar the agent
analyzes market data + technical indicators and decides buy/sell/hold.

Differences from the DJIA stock backtest:
- Data comes from :class:`BinanceDataLoader` (public REST API, no credentials).
- The tradable universe is ``CRYPTO_PAIRS`` (imported from the validator), not DJIA.
- Crypto trades 24/7, so there is no market-hours filter and intervals shorter
  than an hour are allowed.
- Default capital is $10,000 USDT (crypto-appropriate size).

Runs stored in the same SQLite database as the stock backtests, tagged as crypto
runs (agent name ``Crypto Agent`` / mode ``backtest``, ``crypto_...`` run ids).

Usage:
    python3 dashboard/scripts/backtest_crypto_agent.py --start 2026-06-01 --end 2026-07-01 --interval 1h
"""

import sys
import argparse
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Bootstrap for non-package execution (mirrors backtest_hourly_agent.py): when
# run directly as a file the repo root is not on sys.path, so the canonical
# ``dashboard.backend.*`` imports below would fail. As part of the package this
# is a harmless no-op.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.database import db
import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.infrastructure.llm.validator import CRYPTO_PAIRS
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.metrics import (
    calculate_sharpe,
    calculate_max_drawdown,
)
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.market_data.binance_bars import (
    BinanceDataLoader,
    MarketDataUnavailableError,
)
from dashboard.backend.infrastructure.llm.backtest_harness import (
    HAS_ANTHROPIC,
    default_model_name,
    make_llm_client,
)

# ============================================================================
# Configuration
# ============================================================================

# Default crypto universe — the top-10 Binance USDT pairs (kept in sync with
# validator.CRYPTO_PAIRS, imported above).
DEFAULT_PAIRS = CRYPTO_PAIRS

DEFAULT_START = "2026-06-01"
DEFAULT_END = "2026-07-01"
DEFAULT_INTERVAL = "1h"
# Crypto-appropriate portfolio size.
INITIAL_CAPITAL = 10000.0


# ============================================================================
# Crypto Backtester
# ============================================================================

class CryptoBacktester:
    """Runs a crypto backtest with an agent plus a buy-and-hold baseline.

    Mirrors ``HourlyBacktester`` but simplified for crypto: Binance data,
    24/7 trading (no market-hours filter), and the ``CRYPTO_PAIRS`` universe.
    """

    def __init__(self, start_date: str, end_date: str, interval: str = DEFAULT_INTERVAL,
                 session_id: str = "crypto-demo-session", use_llm: bool = False,
                 pairs: Optional[List[str]] = None, model: str = None):
        # Swap backwards dates, matching the stock backtester's behavior.
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start > end:
                print(f"⚠️  Dates were backwards ({start_date} > {end_date}). Swapping...")
                start_date, end_date = end_date, start_date
        except ValueError:
            pass

        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.session_id = session_id
        self.pairs = pairs or DEFAULT_PAIRS
        self.model = model or default_model_name()
        self.data_loader = BinanceDataLoader()
        self.all_data: Dict = {}
        self.use_llm = use_llm and HAS_ANTHROPIC
        self.llm_client = None

        if self.use_llm:
            self.llm_client = make_llm_client()
            if self.llm_client is None:
                print(
                    "⚠️  No LLM key (COMMONSTACK_API_KEY / ANTHROPIC_API_KEY) set. "
                    "Running without LLM."
                )
                self.use_llm = False
            else:
                print(f"✅ LLM initialized (model={self.model})")

    def load_data(self):
        """Fetch OHLCV bars from Binance."""
        self.all_data = self.data_loader.fetch_bars(
            self.pairs, self.start_date, self.end_date, interval=self.interval
        )
        if not self.all_data:
            print("❌ No data fetched.")
            raise MarketDataUnavailableError(
                f"No crypto data available for {self.start_date}..{self.end_date}"
            )

    def calculate_indicators(self):
        """Calculate technical indicators for all pairs."""
        print("\n📈 Calculating technical indicators...")
        count = 0
        for symbol, df in self.all_data.items():
            self.all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
            count += 1
            if count % 5 == 0:
                print(f"  ✅ {count}/{len(self.all_data)} pairs...")
        print(f"  ✅ All indicators calculated\n")

    def run_agent_backtest(self) -> Tuple[str, List[Dict]]:
        """Run backtest with the agent making per-bar decisions (24/7, no market-hours filter)."""
        print("🤖 Running Crypto Agent backtest...\n")

        llm_calls_count = 0
        llm_model = "rule-based"

        manager = PortfolioManager(initial_capital=INITIAL_CAPITAL)

        # Collect all timestamps across pairs and keep bars with data for 80%+ pairs.
        all_timestamps = set()
        for df in self.all_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)

        min_required = int(len(self.all_data) * 0.8)
        filtered = []
        for ts in all_timestamps:
            real_data_count = sum(1 for df in self.all_data.values() if ts in df.index)
            if real_data_count >= min_required:
                filtered.append(ts)
        all_timestamps = filtered if filtered else all_timestamps

        # Crypto trades around the clock — unlike the stock backtest there is NO
        # market-hours filter here.
        print(f"   Trading {len(all_timestamps)} {self.interval} bars (24/7 crypto)...\n")

        # Forward-filled price cache to handle any missing bars.
        print("   Pre-computing forward-filled price cache...")
        price_cache = {}
        for symbol, df in self.all_data.items():
            price_cache[symbol] = {}
            last_price = None
            for timestamp in all_timestamps:
                if timestamp in df.index:
                    last_price = df.loc[timestamp, "close"]
                    price_cache[symbol][timestamp] = last_price
                elif last_price is not None:
                    price_cache[symbol][timestamp] = last_price
        print("   ✅ Cache ready\n")

        for i, timestamp in enumerate(all_timestamps):
            market_data = {}
            for symbol in self.pairs:
                if symbol not in self.all_data:
                    continue
                df = self.all_data[symbol]
                if timestamp not in df.index:
                    continue
                market_data[symbol] = df.loc[timestamp]

            state = manager.get_portfolio_state(market_data, price_cache, timestamp)
            state["timestamp"] = timestamp

            if self.use_llm and self.llm_client:
                decision = manager.make_trading_decision_with_llm(
                    state,
                    self.llm_client,
                    mode="safe_trading",
                    model=self.model,
                )
                llm_calls_count += 1
                if llm_calls_count == 1:
                    llm_model = self.model
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision["actions"], market_data, timestamp)
            manager.update_equity(market_data, price_cache, timestamp)

            if (i + 1) % 100 == 0:
                equity = manager.equity_history[-1]["equity"]
                pct_return = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                print(f"   Bar {i+1}/{len(all_timestamps)}: Equity ${equity:,.0f} ({pct_return:+.1f}%)")

        equity_curve = manager.get_equity_curve()
        for entry in equity_curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()

        run_id = f"crypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_curve[0]["equity"] if equity_curve else INITIAL_CAPITAL
        final_eq = equity_curve[-1]["equity"] if equity_curve else INITIAL_CAPITAL
        total_return = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL

        est_cost = token_cost.estimate_cost_usd(
            llm_model, manager.input_tokens, manager.output_tokens
        )

        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name="Crypto Agent",
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=calculate_sharpe(equity_curve),
            max_drawdown=calculate_max_drawdown(equity_curve),
            num_trades=len(manager.trades),
            llm_model=llm_model,
            llm_calls=manager.llm_calls,
            input_tokens=manager.input_tokens,
            output_tokens=manager.output_tokens,
            est_cost_usd=est_cost,
        )
        db.insert_equity_points(run_id, equity_curve)

        print(f"\n  ✅ Crypto Agent backtest complete")
        print(f"     • Run ID: {run_id}")
        print(f"     • Model: {llm_model}")
        print(f"     • Trades: {len(manager.trades)}")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")

        return run_id, equity_curve

    def _equal_weight_buyhold(self) -> List[Dict]:
        """Equal-weight buy-and-hold equity curve over the loaded crypto bars.

        Computed inline rather than via ``baseline_generator.generate_baselines``:
        that helper requires Alpaca credentials in its constructor and applies a
        9:30-4pm ET market-hours filter + contest-window trimming, both of which
        are wrong for 24/7 crypto. Here we simply buy an equal USDT slice of each
        pair at the first bar and forward-fill valuation across every bar.
        """
        if not self.all_data:
            return []

        all_timestamps = set()
        for df in self.all_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)
        if not all_timestamps:
            return []

        first_ts = all_timestamps[0]
        num_symbols = len(self.all_data)
        allocation = INITIAL_CAPITAL / num_symbols

        positions: Dict[str, float] = {}
        cash = INITIAL_CAPITAL
        # Crypto is fractional — buy fractional quantities rather than whole units.
        for symbol, df in self.all_data.items():
            if first_ts not in df.index:
                continue
            price = df.loc[first_ts, "close"]
            if price <= 0:
                continue
            qty = allocation / price
            positions[symbol] = qty
            cash -= qty * price

        # Forward-filled price cache for smooth valuation.
        price_cache: Dict[str, Dict] = {}
        for symbol, df in self.all_data.items():
            if symbol not in positions:
                continue
            price_cache[symbol] = {}
            last_price = df.loc[first_ts, "close"] if first_ts in df.index else None
            for ts in all_timestamps:
                if ts in df.index:
                    last_price = df.loc[ts, "close"]
                if last_price is not None:
                    price_cache[symbol][ts] = last_price

        equity_curve = []
        for ts in all_timestamps:
            positions_value = sum(
                qty * price_cache[symbol].get(ts, 0)
                for symbol, qty in positions.items()
                if symbol in price_cache
            )
            equity = cash + positions_value
            equity_curve.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts,
                "equity": equity,
                "cash": cash,
                "positions_value": positions_value,
            })
        return equity_curve

    def run_buyhold_baseline(self) -> Tuple[Optional[str], List[Dict]]:
        """Equal-weight buy-and-hold baseline across all crypto pairs."""
        print("📊 Running Crypto Buy & Hold baseline...\n")

        equity_history = self._equal_weight_buyhold()
        if not equity_history:
            return None, []

        run_id = f"crypto_buyhold_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_history[0]["equity"]
        final_eq = equity_history[-1]["equity"]
        total_return = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL

        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name="Crypto Buy & Hold",
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=calculate_sharpe(equity_history),
            max_drawdown=calculate_max_drawdown(equity_history),
            num_trades=1,
        )
        db.insert_equity_points(run_id, equity_history)

        print(f"  ✅ Crypto Buy & Hold baseline complete")
        print(f"     • Run ID: {run_id}")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")

        return run_id, equity_history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crypto backtest with Agent vs Buy & Hold (Binance data)"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Candle interval (default: 1h)")
    parser.add_argument("--session-id", default="crypto-demo-session", help="Session ID for isolation")
    parser.add_argument("--clear", action="store_true", help="Clear all data first")
    parser.add_argument("--use-llm", action="store_true", default=False,
                        help="Use LLM for trading decisions (default: rule-based)")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="Disable LLM, use rule-based logic")
    parser.add_argument("--model", default=None, help="Override the LLM model id.")

    args = parser.parse_args()

    if args.clear:
        print("🗑️ Clearing all existing data...\n")
        db.clear_all()

    print(f"\n🚀 Crypto Agent Backtest Framework")
    print(f"{'='*70}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Interval: {args.interval}")
    print(f"Pairs: {len(DEFAULT_PAIRS)} ({', '.join(DEFAULT_PAIRS)})")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} USDT")
    print(f"{'='*70}\n")

    backtester = CryptoBacktester(
        args.start,
        args.end,
        interval=args.interval,
        session_id=args.session_id,
        use_llm=args.use_llm,
        model=args.model,
    )

    print("1️⃣ Loading historical data from Binance...")
    backtester.load_data()

    print("\n2️⃣ Calculating technical indicators...")
    backtester.calculate_indicators()

    print(f"\n📊 Loaded {len(backtester.all_data)} pairs\n")

    print("3️⃣ Running backtests...\n")
    agent_id, _ = backtester.run_agent_backtest()
    bh_id, _ = backtester.run_buyhold_baseline()

    print(f"{'='*70}")
    print(f"✅ All crypto backtests complete!")
    print(f"{'='*70}")
    print(f"\nRun IDs:")
    print(f"  • Crypto Agent: {agent_id}")
    print(f"  • Crypto Buy & Hold: {bh_id}")
    print(f"\n📊 Dashboard: uvicorn dashboard.backend.app:app → http://localhost:8000")


if __name__ == "__main__":
    try:
        main()
    except MarketDataUnavailableError as exc:
        # Library code raises so server threads can catch it; the CLI boundary is
        # where the process exit belongs.
        print(f"❌ {exc}")
        sys.exit(1)
