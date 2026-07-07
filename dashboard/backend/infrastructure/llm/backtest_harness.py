"""LLM backtest harness: external model-interaction infrastructure.

Extracted (Phase 2C2) from ``PortfolioManager.make_trading_decision_with_llm`` in
``dashboard/scripts/backtest_hourly_agent.py``. This module owns the *external
LLM infrastructure* concerns only:

* the optional Anthropic SDK import (``Anthropic`` / ``HAS_ANTHROPIC``) and the
  optional OpenAI-compatible SDK import (``OpenAI`` / ``HAS_OPENAI``);
* the default model name (``LLM_MODEL_NAME``);
* the system prompts (stock + crypto) and request parameters;
* Anthropic and OpenAI-compatible client invocation / prompt submission;
* response-text extraction (both provider shapes);
* token-usage extraction (both provider shapes);
* response parsing with the existing JSON-repair fallbacks.

An OpenAI-compatible path (OpenCode Zen / DeepSeek) sits alongside the original
Anthropic path: ``make_openai_client`` builds the client, and
``request_trading_decision`` / ``extract_response_text`` / ``extract_token_usage``
detect the client/response shape and dispatch accordingly. The Anthropic path is
unchanged.

Business rules (market-snapshot construction, action conversion, position
sizing, rule-based fallback) and portfolio/manager-state mutation deliberately
remain in the legacy ``PortfolioManager`` wrapper. Behavior here is byte-for-byte
identical to the original inline logic.

This module reuses the already-extracted ``fix_json_formatting`` from
``dashboard.backend.infrastructure.llm.decision_parsing`` (no duplication) and is
domain/infrastructure-only: it must NOT import FastAPI, API routers, the database
singleton, Alpaca clients, dashboard scripts, ``PortfolioManager``, or
``HourlyBacktester``. Importing it is safe without an Anthropic API key (the SDK
import is optional and there is no import-time network or credential access).
"""

import json
import os
from typing import Dict, Optional

from dashboard.backend.infrastructure.llm.decision_parsing import fix_json_formatting

# Optional: LLM integration. Mirrors the original optional-dependency behavior
# (no API key required at import time; only the SDK presence is detected).
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    # Bind ``Anthropic`` to None (instead of leaving it undefined) so the legacy
    # script can unconditionally re-export it; consumers always guard on
    # ``HAS_ANTHROPIC`` before using the client.
    Anthropic = None
    HAS_ANTHROPIC = False
    print("⚠️  Anthropic SDK not installed. Fallback to rule-based trading.")
    print("   To enable LLM: pip install anthropic")

# Optional: OpenAI-compatible provider support (e.g. OpenCode Zen / DeepSeek).
# Same optional-dependency contract as the Anthropic path: only the SDK presence
# is detected at import time (no key required, no network/credential access).
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    # Bind ``OpenAI`` to None (instead of leaving it undefined) so type checks
    # (``isinstance(client, OpenAI)``) can guard on ``HAS_OPENAI`` first.
    OpenAI = None
    HAS_OPENAI = False

# Default model name (model selection). Re-exported by the legacy script.
LLM_MODEL_NAME = "claude-haiku-4-5-20251001"  # Change this to switch models

# Same default model, but as the CommonStack gateway slug. CommonStack expects
# ``provider/model`` ids, so when routing through CommonStack we must send the
# gateway slug rather than the native Anthropic id (see the integration report's
# "slug/gateway coupling" note). Pricing in token_cost.py matches "claude-haiku-4".
COMMONSTACK_MODEL_NAME = "anthropic/claude-haiku-4-5"

# CommonStack is a unified gateway exposing OpenAI/Google/xAI/DeepSeek/Qwen/
# Anthropic models behind one key. Its Anthropic-compatible endpoint returns
# Anthropic-shaped responses (``content[0].text`` + ``usage.{input,output}_tokens``),
# so the existing request/parse/usage code works unchanged — we only swap the
# client's base_url and pass a ``provider/model`` slug as the model id.
COMMONSTACK_BASE_URL = os.getenv("COMMONSTACK_BASE_URL", "https://api.commonstack.ai")

# OpenAI-compatible provider support. Points at OpenCode Zen by default (its
# ``deepseek-v4-pro`` model), but any OpenAI-compatible endpoint works by
# overriding ``OPENAI_BASE_URL``/``OPENCODE_ZEN_BASE_URL`` and ``LLM_OPENAI_MODEL``.
# Unlike CommonStack (which is Anthropic-shaped), this path speaks the native
# OpenAI chat-completions format (``choices[0].message.content`` +
# ``usage.{prompt,completion}_tokens``), handled by the extractors below.
OPENAI_COMPATIBLE_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/go/v1"))
OPENAI_COMPATIBLE_MODEL_NAME = os.getenv("LLM_OPENAI_MODEL", "deepseek-v4-pro")


def make_openai_client():
    """Create an OpenAI-compatible client (OpenCode Zen / DeepSeek) or ``None``.

    Reads ``OPENAI_API_KEY`` or ``OPENCODE_ZEN_API_KEY`` (in that order) and points
    the client at ``OPENAI_COMPATIBLE_BASE_URL``. Returns ``None`` when the SDK is
    missing or no key is set, so callers fall back to rule-based trading exactly as
    the Anthropic path does. Mirrors the Anthropic path's status printing.
    """
    if not HAS_OPENAI or OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENCODE_ZEN_API_KEY")
    if not key:
        return None
    try:
        client = OpenAI(api_key=key, base_url=OPENAI_COMPATIBLE_BASE_URL)
        print(f"✅ OpenAI-compatible client initialized (base_url={OPENAI_COMPATIBLE_BASE_URL})")
        return client
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to init OpenAI-compatible client: {exc}")
        return None


def make_llm_client():
    """Create a client for trading decisions, preferring Anthropic-compatible.

    Prefers CommonStack when ``COMMONSTACK_API_KEY`` is set (one key reaches all
    leaderboard models); otherwise falls back to native Anthropic via
    ``ANTHROPIC_API_KEY``. When neither key is set (or the Anthropic SDK is
    missing), falls back to the OpenAI-compatible provider (OpenCode Zen /
    DeepSeek) via :func:`make_openai_client`. Returns ``None`` when no SDK or key
    is available, so callers fall back to rule-based trading exactly as before.
    """
    if HAS_ANTHROPIC and Anthropic is not None:
        commonstack_key = os.getenv("COMMONSTACK_API_KEY")
        if commonstack_key:
            try:
                return Anthropic(api_key=commonstack_key, base_url=COMMONSTACK_BASE_URL)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"⚠️  Failed to init CommonStack client: {exc}")
                return None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                return Anthropic(api_key=anthropic_key)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"⚠️  Failed to init Anthropic client: {exc}")
                return None
    # No CommonStack/Anthropic key (or the Anthropic SDK is unavailable): try the
    # OpenAI-compatible provider. Returns None if that too is unavailable.
    return make_openai_client()


def default_model_name() -> str:
    """Return the default model id matching the client ``make_llm_client`` builds.

    When ``COMMONSTACK_API_KEY`` is set we route through CommonStack and must use
    its gateway slug (``anthropic/claude-haiku-4-5``); with an Anthropic key we use
    the native Anthropic id (``claude-haiku-4-5-20251001``); and when only an
    OpenAI-compatible key is present we use ``OPENAI_COMPATIBLE_MODEL_NAME``
    (``deepseek-v4-pro``) so the model matches the client :func:`make_llm_client`
    builds. Callers can still override this with an explicit model id.
    """
    if os.getenv("COMMONSTACK_API_KEY"):
        return COMMONSTACK_MODEL_NAME
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLM_MODEL_NAME
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENCODE_ZEN_API_KEY"):
        return OPENAI_COMPATIBLE_MODEL_NAME
    return LLM_MODEL_NAME

# System prompt sent on every request. Preserved exactly from the original
# inline string (do not "improve" it).
SYSTEM_PROMPT = """You are an expert quantitative trading advisor analyzing DJIA stocks.

You have deep knowledge of:
- Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Indicator interpretation and confluence
- Risk management and position sizing
- Trading psychology and market microstructure

IMPORTANT INSTRUCTIONS:
1. Analyze EACH stock signal provided (don't skip any)
2. For each stock, decide: BUY, SELL, or HOLD
3. Always include a confidence score (0.0-1.0)
4. Return a JSON object with an "actions" array containing one entry per stock
5. Even if you decide HOLD, include it in the actions array
6. Respond with ONLY valid JSON - no explanations outside JSON

Make precise, actionable trading decisions based on the technical indicators provided."""


# Crypto-specific system prompt. Same JSON contract as the stock prompt above,
# but tuned for 24/7 crypto markets and higher volatility. Selected via
# ``get_system_prompt("crypto")`` — the stock ``SYSTEM_PROMPT`` stays the default.
CRYPTO_SYSTEM_PROMPT = """You are an expert cryptocurrency trading advisor.

You have deep knowledge of:
- Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Crypto market dynamics and 24/7 trading patterns
- Risk management and position sizing for volatile assets
- On-chain metrics and market sentiment

IMPORTANT INSTRUCTIONS:
1. Analyze EACH cryptocurrency signal provided (don't skip any)
2. For each crypto, decide: BUY, SELL, or HOLD
3. Always include a confidence score (0.0-1.0)
4. Return a JSON object with an "actions" array containing one entry per crypto
5. Even if you decide HOLD, include it in the actions array
6. Respond with ONLY valid JSON - no explanations outside JSON
7. Consider the higher volatility of crypto — size positions conservatively

Make precise, actionable trading decisions based on the technical indicators provided."""


def get_system_prompt(asset_type: str = "stocks") -> str:
    """Return the system prompt for the given ``asset_type``.

    ``"crypto"`` returns :data:`CRYPTO_SYSTEM_PROMPT`; anything else (default
    ``"stocks"``) returns the original DJIA :data:`SYSTEM_PROMPT`, so existing
    stock callers are unaffected.
    """
    if asset_type == "crypto":
        return CRYPTO_SYSTEM_PROMPT
    return SYSTEM_PROMPT


# Per-request output-token ceiling. Defaults to 2000 (unchanged) but can be
# lowered via env (e.g. LLM_MAX_OUTPUT_TOKENS=600) to cap spend in small demos.
def _parse_max_output_tokens(raw: Optional[str]) -> int:
    """Parse ``LLM_MAX_OUTPUT_TOKENS`` defensively.

    A malformed or non-positive value must not crash the module at import time
    (this constant is evaluated on ``import``, so a bad env var would take down
    every backtest CLI and the backend). Falls back to 2000 with a warning.
    """
    if raw is None:
        return 2000
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        print(
            f"⚠️ Ignoring invalid LLM_MAX_OUTPUT_TOKENS={raw!r} "
            f"(expected a positive integer); using default 2000"
        )
        return 2000
    return value


DEFAULT_MAX_OUTPUT_TOKENS = _parse_max_output_tokens(os.getenv("LLM_MAX_OUTPUT_TOKENS"))


def _is_openai_client(client) -> bool:
    """True when ``client`` is an OpenAI-compatible client (not Anthropic).

    Prefers an ``isinstance`` check against the real SDK class; falls back to a
    duck-typed ``chat.completions`` probe when the SDK type is unavailable. The
    Anthropic/CommonStack client exposes ``messages`` (not ``chat``), so it is
    correctly routed to the Anthropic path.
    """
    if HAS_OPENAI and OpenAI is not None and isinstance(client, OpenAI):
        return True
    chat = getattr(client, "chat", None)
    return chat is not None and hasattr(chat, "completions")


def request_trading_decision(client, *, prompt: str, model: Optional[str] = None,
                             max_tokens: Optional[int] = None, asset_type: str = "stocks"):
    """Submit the prompt to the LLM client and return the raw response.

    Detects the client type and dispatches accordingly:

    * OpenAI-compatible clients use ``client.chat.completions.create(...)`` with a
      ``system`` + ``user`` message pair and ``model or OPENAI_COMPATIBLE_MODEL_NAME``.
    * Anthropic/CommonStack clients keep the original ``client.messages.create(...)``
      call exactly: same model resolution (``model or LLM_MODEL_NAME``), system
      prompt, and single user message.

    ``asset_type`` selects the system prompt via :func:`get_system_prompt`
    (defaults to the original stock prompt). Exceptions are intentionally NOT
    caught here so the legacy wrapper's outer handler can fall back to rule-based
    logic, exactly as before.
    """
    system_prompt = get_system_prompt(asset_type)
    if _is_openai_client(client):
        return client.chat.completions.create(
            model=model or OPENAI_COMPATIBLE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        )
    return client.messages.create(
        model=model or LLM_MODEL_NAME,
        max_tokens=max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,  # Reduced from 3000 (saves tokens)
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )


def extract_response_text(response) -> str:
    """Extract the response text from an Anthropic or OpenAI-compatible response.

    OpenAI-compatible responses carry a ``choices`` list
    (``choices[0].message.content``); Anthropic responses use ``content[0].text``
    exactly as the original.
    """
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return response.content[0].text


def extract_token_usage(response):
    """Return ``(input_tokens, output_tokens)`` deltas from a provider response.

    Handles both usage shapes: OpenAI-compatible (``usage.prompt_tokens`` /
    ``usage.completion_tokens``) and Anthropic (``usage.input_tokens`` /
    ``usage.output_tokens``). Returns ``(0, 0)`` when no usage object is present.
    The caller applies these to manager state and increments the call counter.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if hasattr(usage, "prompt_tokens"):
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def parse_llm_response(llm_response: str) -> Optional[Dict]:
    """Parse the LLM response into a decision dict, or ``None`` on failure.

    Replicates the original STEP-3 parsing exactly: strip markdown fences, slice
    from the first ``{`` to the last ``}``, ``json.loads`` with two escalating
    ``fix_json_formatting`` / bracket-balancing repair attempts, and the same
    printed diagnostics. Returns the parsed ``decision`` dict on success.

    Returns ``None`` in every case where the original method returned
    ``{"actions": []}`` (no JSON found, unrecoverable parse error, or any other
    exception in the parse block), so the caller can return ``{"actions": []}``
    and preserve behavior.
    """
    print(f"\n📫 Parsing LLM response...")
    print(f"   Raw response (first 300 chars): {llm_response[:300]}")

    try:
        # Extract JSON from response
        # First, strip markdown code fences if present
        response_cleaned = llm_response
        if '```json' in response_cleaned:
            response_cleaned = response_cleaned.replace('```json', '').replace('```', '')
        elif '```' in response_cleaned:
            response_cleaned = response_cleaned.replace('```', '')

        start = response_cleaned.find('{')
        end = response_cleaned.rfind('}') + 1
        if start < 0 or end <= 0:
            print(f"   ❌ No JSON found in response")
            print(f"   Full response: {response_cleaned[:500]}")
            return None

        json_str = response_cleaned[start:end]

        # Try to parse
        try:
            decision = json.loads(json_str)
            print(f"   ✅ JSON parsed successfully")
        except json.JSONDecodeError as e:
            # Try to fix common formatting issues
            print(f"   ⚠️  Initial parse failed: {e}")
            print(f"   Attempting to fix JSON formatting...")

            json_str_fixed = fix_json_formatting(json_str)
            try:
                decision = json.loads(json_str_fixed)
                print(f"   ✅ JSON fixed and parsed successfully!")
            except json.JSONDecodeError as e2:
                print(f"   ❌ Still failed after fix: {e2}")
                print(f"   Error at line {e2.lineno}, column {e2.colno}")

                # Show detailed context around error
                lines = json_str_fixed.split('\n')
                if e2.lineno <= len(lines):
                    start = max(0, e2.lineno - 3)
                    end = min(len(lines), e2.lineno + 2)
                    print(f"\n   Context around error (lines {start+1}-{end}):")
                    for i in range(start, end):
                        marker = ">> " if i == e2.lineno - 1 else "   "
                        print(f"   {marker}{i+1:3d}: {lines[i][:70]}")

                # Try one more aggressive fix
                print(f"\n   Attempting second fix attempt (validate structure)...")
                try:
                    # Count opening vs closing brackets
                    open_count = json_str_fixed.count('{')
                    close_count = json_str_fixed.count('}')
                    if open_count != close_count:
                        print(f"   Bracket mismatch: {open_count} open, {close_count} close")
                        # Remove extra closing brackets from the end
                        while json_str_fixed.count('}') > json_str_fixed.count('{'):
                            json_str_fixed = json_str_fixed.rsplit('}', 1)[0] + '}'
                        print(f"   Removed extra closing brackets")

                    decision = json.loads(json_str_fixed)
                    print(f"   ✅ JSON fixed after structure cleanup!")
                except json.JSONDecodeError as e3:
                    print(f"   ❌ Cannot fix: {e3}")
                    return None

        print(f"   Actions from LLM: {len(decision.get('actions', []))}")
        return decision

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"   ❌ Failed to parse JSON: {e}")
        print(f"   LLM response: {llm_response[:500]}...")
        return None
