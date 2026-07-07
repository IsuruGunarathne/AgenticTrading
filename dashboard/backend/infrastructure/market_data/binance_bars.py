"""Binance historical bar loader.

Mirrors :class:`dashboard.backend.infrastructure.market_data.alpaca_bars.AlpacaDataLoader`
so the crypto backtest path is a drop-in swap for the Alpaca stock path. Key
differences from Alpaca:

- Uses the Binance **public** REST API
  (``https://api.binance.com/api/v3/klines``) — no API key or SDK is required for
  historical data, so ``__init__`` takes no credentials and imports nothing at
  module load time.
- Crypto trades 24/7, so intervals shorter than an hour are supported
  (``1m``/``5m``/``15m``/``1h``/``4h``/``1d``).
- Binance symbols carry no slash (``BTCUSDT``, ``ETHUSDT``).

Like the Alpaca loader, a missing dependency or unreachable API raises
:class:`MarketDataUnavailableError` (imported/re-exported from ``alpaca_bars``)
rather than calling ``sys.exit`` — a plain Exception is catchable at every server
call site, so a daemon loader thread never gets wedged by a ``SystemExit``.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

# Reuse the canonical market-data exception so callers can catch a single type
# regardless of the underlying provider (Alpaca or Binance).
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)

# Binance klines: one request returns at most this many candles.
_MAX_LIMIT = 1000

# Supported kline intervals mapped to their duration in milliseconds. Used to
# paginate through the requested window and to advance the cursor between pages.
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


class BinanceDataLoader:
    """Fetches historical OHLCV bars from the Binance public REST API."""

    def __init__(self, base_url: str = "https://api.binance.com"):
        """Initialize the loader.

        No credentials are required — historical klines are a public endpoint.
        The ``requests`` import stays lazy (inside :meth:`fetch_bars`) so
        importing this module performs no network work and pulls in no deps.
        """
        self.base_url = base_url.rstrip("/")
        print("✅ Binance data loader ready (public API, no credentials needed)")

    @staticmethod
    def _to_millis(value: str) -> int:
        """Convert a ``YYYY-MM-DD`` (or ISO) date string to epoch milliseconds (UTC)."""
        # Accept both plain dates and full ISO timestamps.
        try:
            dt = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _fetch_symbol(
        self,
        requests_mod,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch all klines for a single symbol, paginating in <=1000-candle batches."""
        step = _INTERVAL_MS[interval]
        rows: List[List] = []
        cursor = start_ms

        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _MAX_LIMIT,
            }
            response = requests_mod.get(
                f"{self.base_url}/api/v3/klines", params=params, timeout=15
            )

            # Binance signals rate limiting with 429 (and a hard ban with 418).
            # Back off briefly and retry the same cursor rather than dropping data.
            if response.status_code in (429, 418):
                retry_after = int(response.headers.get("Retry-After", "2"))
                print(f"  ⏳ {symbol}: rate limited (HTTP {response.status_code}), "
                      f"waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                print(f"  ⚠️  {symbol}: HTTP {response.status_code} — {response.text[:120]}")
                break

            batch = response.json()
            if not batch:
                break

            rows.extend(batch)

            # Advance past the last candle's open time. When a full page comes
            # back there is likely more data, so keep paging with a small,
            # rate-limit-friendly delay between batches.
            last_open = batch[-1][0]
            cursor = last_open + step
            if len(batch) < _MAX_LIMIT:
                break
            time.sleep(0.1)

        if not rows:
            return pd.DataFrame()

        # Binance kline row layout:
        # [openTime, open, high, low, close, volume, closeTime, ...ignored...]
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_base", "taker_quote", "ignore",
            ],
        )
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("timestamp", inplace=True)
        # A candle's endTime can round the last row past ``end_ms``; drop those.
        df = df[df.index <= pd.to_datetime(end_ms, unit="ms", utc=True)]
        return df.sort_index()

    def fetch_bars(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1h",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV klines from Binance for one or more symbols.

        Args:
            symbols: Binance trading pairs, e.g. ``["BTCUSDT", "ETHUSDT"]``.
            start: Start date (YYYY-MM-DD or ISO 8601), interpreted as UTC.
            end: End date (YYYY-MM-DD or ISO 8601), interpreted as UTC.
            interval: One of ``1m``, ``5m``, ``15m``, ``1h``, ``4h``, ``1d``.

        Returns:
            ``{symbol: DataFrame}`` indexed by UTC timestamp with columns
            ``open, high, low, close, volume`` — the same shape returned by
            :meth:`AlpacaDataLoader.fetch_bars`.

        Raises:
            MarketDataUnavailableError: if ``requests`` is unavailable or the
                interval is not supported.
        """
        if interval not in _INTERVAL_MS:
            raise MarketDataUnavailableError(
                f"Unsupported Binance interval: {interval!r}. "
                f"Choose one of {sorted(_INTERVAL_MS)}"
            )

        try:
            import requests as requests_mod
        except ImportError as e:  # pragma: no cover - requests is a core dep
            print(f"❌ requests not installed: {e}")
            raise MarketDataUnavailableError(
                "requests is not installed (pip install requests)"
            ) from e

        start_ms = self._to_millis(start)
        end_ms = self._to_millis(end)

        print(f"\n📊 Fetching {len(symbols)} crypto pairs from {start} to {end}...")
        print(f"   Timeframe: {interval} (Binance public API)\n")

        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self._fetch_symbol(requests_mod, symbol, start_ms, end_ms, interval)
            except Exception as e:  # noqa: BLE001 - never let one symbol crash the batch
                print(f"  ❌ {symbol}: error fetching bars: {e}")
                continue

            if df.empty:
                print(f"  ⚠️  {symbol}: No data available")
                continue

            data[symbol] = df
            print(f"  ✅ {symbol}: {len(df)} {interval} bars")

        return data
