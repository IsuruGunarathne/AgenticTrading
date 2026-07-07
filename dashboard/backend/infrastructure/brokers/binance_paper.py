"""Binance testnet paper-trading broker adapter.

Mirrors :class:`dashboard.backend.infrastructure.brokers.alpaca_paper.AlpacaPaperTradingClient`
so live crypto paper trading is a drop-in swap for the Alpaca stock adapter. Same
method surface (``get_account`` / ``get_positions`` / ``get_orders`` /
``get_activities`` / ``get_portfolio_history``) and the same defensive contract:
every call returns ``None``/``[]`` on failure and never raises to the caller.

Binance's model differs from Alpaca's, so fields are mapped:
- **Account** — Binance reports per-asset ``balances``; we compute each asset's
  USDT-equivalent value from live ticker prices and sum them into ``equity`` /
  ``portfolio_value``. The free USDT balance is reported as ``cash``.
- **Positions** — spot Binance has no "positions"; each non-zero non-USDT balance
  is surfaced as a :class:`Position` valued at the current ticker price.
- **Orders** — ``GET /api/v3/allOrders`` (requires a symbol; we sweep the
  configured pairs).
- **Prices** — ``GET /api/v3/ticker/price``.

Signed endpoints (account, orders) use the standard Binance HMAC-SHA256 scheme
over the query string. Public endpoints (ticker) need no signature. Testnet base:
``https://testnet.binance.vision``.
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

from dashboard.backend.paths import CREDENTIALS_DIR

# The pairs whose orders we sweep for get_orders/get_activities. Binance requires
# a symbol per allOrders call, so we iterate this small default set. Callers can
# override per-call via the ``symbols`` argument.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]


@dataclass
class Position:
    """Represents a current position."""
    symbol: str
    qty: float
    avg_fill_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str  # 'long' or 'short'
    market_value: float


@dataclass
class Trade:
    """Represents a single trade."""
    id: str
    symbol: str
    qty: float
    side: str
    filled_avg_price: float
    filled_at: str
    order_status: str


class BinancePaperTradingClient:
    """Interface to the Binance **testnet** spot trading API (paper trading)."""

    def __init__(self, api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 symbols: Optional[List[str]] = None):
        """Initialize with Binance testnet credentials."""
        # Always initialize these first
        self.api_key = None
        self.secret_key = None

        if api_key is None:
            self._load_from_credentials()
        else:
            self.api_key = api_key
            self.secret_key = secret_key

        self.base_url = "https://testnet.binance.vision"
        self.headers = {"X-MBX-APIKEY": self.api_key}
        self.symbols = symbols or DEFAULT_SYMBOLS

    def _load_from_credentials(self):
        """Load credentials from environment variables or credentials file."""
        # Try environment variables first (for Render, Docker, etc.)
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')

        if self.api_key and self.secret_key:
            return

        # Fall back to credentials file (for local development)
        creds_path = CREDENTIALS_DIR / "binance.json"

        if creds_path.exists():
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                self.api_key = creds.get('api_key') or creds.get('apiKey')
                self.secret_key = creds.get('secret_key') or creds.get('secretKey')
        else:
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")

    def _signed_params(self, params: Optional[Dict] = None) -> Dict:
        """Append timestamp + HMAC-SHA256 signature to a signed-endpoint request."""
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            (self.secret_key or "").encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current ticker prices for the given symbols (public endpoint).

        Args:
            symbols: Binance pairs, e.g. ``["BTCUSDT", "ETHUSDT"]``.

        Returns:
            ``{symbol: price}``; symbols that fail to resolve are omitted.
        """
        prices: Dict[str, float] = {}
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Error fetching prices: {response.status_code}")
                return prices

            wanted = set(symbols)
            for entry in response.json():
                sym = entry.get("symbol")
                if sym in wanted:
                    prices[sym] = float(entry.get("price", 0))
            return prices
        except Exception as e:
            print(f"Exception in get_current_prices: {e}")
            return prices

    def get_account(self) -> Optional[Dict]:
        """
        Get current account info.

        Binance reports per-asset balances; we value every non-zero asset at its
        USDT ticker price and sum to derive equity/portfolio value, matching the
        Alpaca adapter's return shape.

        Returns:
            Dict with: cash, equity, buying_power, portfolio_value, etc.
        """
        try:
            url = f"{self.base_url}/api/v3/account"
            params = self._signed_params()
            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code != 200:
                print(f"Error fetching account: {response.status_code}")
                return None

            account = response.json()
            balances = account.get("balances", [])

            # Collect non-zero balances and price the non-USDT ones.
            holdings = {}
            for bal in balances:
                asset = bal.get("asset")
                total = float(bal.get("free", 0)) + float(bal.get("locked", 0))
                if total > 0:
                    holdings[asset] = total

            usdt_cash = holdings.get("USDT", 0.0)

            # Price each non-USDT asset via its <ASSET>USDT ticker.
            price_symbols = [f"{a}USDT" for a in holdings if a != "USDT"]
            prices = self.get_current_prices(price_symbols) if price_symbols else {}

            equity = usdt_cash
            for asset, qty in holdings.items():
                if asset == "USDT":
                    continue
                price = prices.get(f"{asset}USDT", 0.0)
                equity += qty * price

            return {
                "cash": usdt_cash,
                "equity": equity,
                "buying_power": usdt_cash,
                "portfolio_value": equity,
                "multiplier": 1,
                "account_number": str(account.get("uid", "")),
                "account_status": "ACTIVE" if account.get("canTrade") else "RESTRICTED",
                "created_at": account.get("updateTime"),
            }
        except Exception as e:
            print(f"Exception in get_account: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Spot Binance has no positions, so each non-zero non-USDT balance is
        surfaced as a long :class:`Position` valued at the current ticker price.
        Unrealized P/L is unavailable from spot balances alone and reported as 0.

        Returns:
            List of Position objects
        """
        try:
            url = f"{self.base_url}/api/v3/account"
            params = self._signed_params()
            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code != 200:
                print(f"Error fetching positions: {response.status_code}")
                return []

            balances = response.json().get("balances", [])
            holdings = {}
            for bal in balances:
                asset = bal.get("asset")
                total = float(bal.get("free", 0)) + float(bal.get("locked", 0))
                if total > 0 and asset != "USDT":
                    holdings[asset] = total

            if not holdings:
                return []

            price_symbols = [f"{a}USDT" for a in holdings]
            prices = self.get_current_prices(price_symbols)

            positions = []
            for asset, qty in holdings.items():
                symbol = f"{asset}USDT"
                current_price = prices.get(symbol, 0.0)
                market_value = qty * current_price
                positions.append(Position(
                    symbol=symbol,
                    qty=qty,
                    avg_fill_price=0.0,   # not available from spot balances
                    current_price=current_price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    side="long",
                    market_value=market_value,
                ))
            return positions
        except Exception as e:
            print(f"Exception in get_positions: {e}")
            return []

    def get_orders(self, limit: int = 50, status: str = "all",
                   symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Get order history.

        Binance's ``allOrders`` endpoint is per-symbol, so we sweep the configured
        pairs and merge the results, newest first.

        Args:
            limit: Max orders to return (across all swept symbols).
            status: 'all', 'open', 'closed', 'cancelled' (Binance-side filtering
                is limited; 'open' maps to still-active orders).
            symbols: Optional override of the pairs to sweep.

        Returns:
            List of order dicts
        """
        try:
            all_orders: List[Dict] = []
            for symbol in (symbols or self.symbols):
                url = f"{self.base_url}/api/v3/allOrders"
                params = self._signed_params({"symbol": symbol, "limit": limit})
                response = requests.get(url, headers=self.headers, params=params, timeout=10)

                if response.status_code != 200:
                    print(f"Error fetching orders for {symbol}: {response.status_code}")
                    continue
                all_orders.extend(response.json())

            # 'open' → still-active orders; otherwise return everything.
            if status == "open":
                active = {"NEW", "PARTIALLY_FILLED"}
                all_orders = [o for o in all_orders if o.get("status") in active]

            all_orders.sort(key=lambda o: o.get("time", 0), reverse=True)
            return all_orders[:limit]
        except Exception as e:
            print(f"Exception in get_orders: {e}")
            return []

    def get_activities(self, activity_type: str = "FILL", limit: int = 100,
                       symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Get trade activity (fills).

        Binance has no combined activities endpoint, so filled orders swept from
        ``allOrders`` stand in for fills.

        Args:
            activity_type: 'FILL' for trades (kept for signature parity).
            limit: Max activities to return.
            symbols: Optional override of the pairs to sweep.

        Returns:
            List of activity dicts (filled orders)
        """
        try:
            orders = self.get_orders(limit=limit, status="all", symbols=symbols)
            fills = [o for o in orders if o.get("status") == "FILLED"]
            return fills[:limit]
        except Exception as e:
            print(f"Exception in get_activities: {e}")
            return []

    def get_portfolio_history(self, timeframe: str = "1D",
                              symbols: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get portfolio performance history.

        Binance testnet exposes no equity-curve endpoint, so we synthesize a
        single current snapshot from account equity. The shape mirrors Alpaca's
        ``portfolio/history`` response (``timestamp``/``equity`` arrays) so the
        dashboard can consume it uniformly.

        Args:
            timeframe: kept for signature parity (unused).

        Returns:
            Dict with a single-point equity curve, or None on failure.
        """
        try:
            account = self.get_account()
            if account is None:
                return None
            now = int(time.time())
            equity = account["equity"]
            return {
                "timestamp": [now],
                "equity": [equity],
                "profit_loss": [0.0],
                "profit_loss_pct": [0.0],
                "base_value": equity,
                "timeframe": timeframe,
            }
        except Exception as e:
            print(f"Exception in get_portfolio_history: {e}")
            return None
