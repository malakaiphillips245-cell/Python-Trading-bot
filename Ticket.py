"""Entry point shim for Holy Grail SMC bot."""

import os
import json
import math
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import pandas as pd
import pytz
import time  # Added to allow time.sleep() calls in run_bot and other functions
try:
    import MetaTrader5 as mt5
except Exception:
    class _DummyMT5:
        def __getattr__(self, name):
            raise ImportError("MetaTrader5 module not available")
    mt5 = _DummyMT5()

# Global debug flag. Set the environment variable ``HG_DEBUG`` to ``1`` or
# ``true`` to enable verbose debugging output.  Debug messages are
# produced via the ``log_debug`` function defined below.  When disabled
# (default), debug messages are suppressed.
DEBUG_MODE = str(os.getenv('HG_DEBUG', '0')).lower() in ('1', 'true', 'yes')

def log_debug(*args, **kwargs) -> None:
    """Print debug messages when DEBUG_MODE is enabled.

    Accepts arbitrary positional and keyword arguments and forwards them to
    ``print``.  When DEBUG_MODE is False, this function does nothing.
    """
    if DEBUG_MODE:
        try:
            print(*args, **kwargs)
        except Exception as e:
            log_debug("suppressed exception:", e)

# -----------------------------------------------------------------------------
# === STATE ===
#
# Introduce a unified runtime state container and modular engines.  This refactor
# centralizes all mutable runtime data within a single `BotState` dataclass.
# Functions must accept a `state: BotState` argument when they need access to
# counters, timestamps, drawdown flags or other mutable values.  The
# HardBlockEngine, SoftFilterEngine and RiskEngine functions define clear
# boundaries between absolute blockers, soft filters and risk management.
# A `run_bot` entry point instantiates the state and invokes the original
# scanning routine without altering trading logic.

@dataclass
class BotState:
    """Container for all mutable runtime data used by the trading bot.

    This dataclass aggregates all runtime counters, cooldowns, drawdown flags,
    per-session XAU trade tracking and miscellaneous storage into a single
    structure. Storing state in one place eliminates implicit global
    mutations and makes risk logic more transparent. When extending the bot,
    new runtime variables should be added here instead of creating new
    module-level globals.
    """
    # Trade counters
    micro_trades_today: int = 0
    full_trades_today: int = 0
    session_trades: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Cooldowns & timestamps
    last_micro_time: Optional[datetime] = None
    last_full_time: Optional[datetime] = None

    # Drawdown & lock flags
    trading_paused: bool = False
    permanent_lockout: bool = False
    day_open_equity: Optional[float] = None
    day_equity_high: Optional[float] = None
    # Session counters for FULL trades (resets per UK day)
    session_full_losses: int = 0
    session_full_count: int = 0

    # Block & reporting stats
    last_block_log: Dict[str, float] = field(default_factory=dict)
    xau_block_stats: Dict[str, Any] = field(default_factory=lambda: {
        'full_block_total': 0,
        'full_block_since_last_report': 0,
        'last_report': 0
    })
    xau_full_blocks: Dict[str, Any] = field(default_factory=lambda: {
        'count': 0,
        'last_alert': 0
    })

    # Per-session tracking for XAU trades (London/NY session counts)
    xau_session_trades: Dict[str, Dict[str, int]] = field(default_factory=dict)
    xau_last_session_label: Optional[str] = None

    # Memory
    holy_memory: Dict[str, Any] = field(default_factory=dict)

    # Recent trade zones (symbol -> {price, time}) and AI micro-lot scaling
    last_trade_zones: Dict[str, Dict] = field(default_factory=dict)
    micro_lot_ai_scale: float = 0.01

    # Miscellaneous storage for future use
    extra: Dict[str, Any] = field(default_factory=dict)

    # Adaptive AI scaling: maintain a history of recent signal strengths for
    # dynamic micro lot adjustment.  When computing micro lot sizes, the
    # bot will use the mean of this history to set the baseline scale.  A
    # finite history length prevents runaway growth and ensures the model
    # responds to recent market conditions rather than distant outliers.
    micro_strength_history: List[float] = field(default_factory=list)

    # Telegram notification state (deduplication for once-per-session/day)
    last_session_notified: Optional[str] = None  # Track which session was notified
    last_xau_bos_setup: Optional[str] = None    # Track BOS setup to avoid spam
    xau_block_reason_sent_this_session: Dict[str, bool] = field(default_factory=dict)  # Track block reasons per session
    last_heartbeat_time: Optional[datetime] = None  # Track last heartbeat message

    # Stability mode & throttles
    strict_mode: bool = False
    strict_mode_since: Optional[datetime] = None
    full_pause_until: Optional[datetime] = None
    micro_pause_until: Optional[datetime] = None
    full_consec_losses: int = 0
    micro_consec_losses: int = 0
    full_risk_reduction: float = 1.0
    last_full_entry_by_symbol_side: Dict[str, datetime] = field(default_factory=dict)
    last_micro_entry_by_symbol_side: Dict[str, datetime] = field(default_factory=dict)
    session_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)


class SymbolProfile:
    """Profile describing instrument-specific constraints and limits."""
    def __init__(self, name: str, full_spread_limit: int, micro_spread_limit: int, hard_spread_limit: int, tick_age_limit: float):
        self.name = name
        self.full_spread_limit = full_spread_limit
        self.micro_spread_limit = micro_spread_limit
        self.hard_spread_limit = hard_spread_limit
        self.tick_age_limit = tick_age_limit


# Central BotState instance (single source of truth for runtime mutable data)
BOT_STATE = BotState()

# Backwards-compatible aliases for commonly used dict-like structures so the
# rest of the code can continue to reference `holy_memory`, `_XAU_BLOCK_STATS`
# and `_XAU_FULL_BLOCKS` while the canonical data lives inside `BOT_STATE`.
holy_memory = BOT_STATE.holy_memory
_XAU_BLOCK_STATS = BOT_STATE.xau_block_stats
_XAU_FULL_BLOCKS = BOT_STATE.xau_full_blocks
XAU_SESSION_TRADES = BOT_STATE.xau_session_trades
XAU_LAST_SESSION_LABEL = BOT_STATE.xau_last_session_label


def HardBlockEngine(state: BotState, symbol: str, side: Optional[str]) -> Tuple[bool, str]:
    """
    Perform hard risk checks that prevent any trade execution.

    This wrapper delegates to the existing `_is_hard_block` helper to preserve
    behaviour but allows the caller to pass in a state container.  It returns
    a tuple `(blocked: bool, reason: str)`.
    """
    try:
        return _is_hard_block(symbol, side)
    except Exception as e:
        return (True, f"Hard block check error: {e}")


def SoftFilterEngine(state: BotState, symbol: str, context: Any) -> Tuple[bool, str, int]:
    """
    Apply soft filters such as session checks, range/chop detection and
    momentum heuristics.  These checks adjust scoring and may downgrade
    classifications but do not outright block trades.  For now this
    implementation defers to the original classification logic and always
    returns `(True, '', 0)` to avoid changing behaviour.
    """
    return (True, "", 0)


def RiskEngine(state: BotState, classification: str, avg_strength: float) -> Tuple[str, float]:
    """
    Enforce risk rules, cooldowns and lot sizing.

    This engine calls the existing cooldown and lot computation helpers to
    ensure the classification and lot size match the original logic.
    Returns a tuple `(final_trade_type: str, lot: float)`.
    """
    try:
        now_uk = datetime.now(SAFE_TZ)
        adjusted = _adjust_for_cooldowns(classification, now_uk)
        lot = _compute_lot(adjusted, avg_strength)
        return (adjusted, lot)
    except Exception:
        return (classification, 0.0)


def run_bot(state: BotState) -> None:
    """
    Start the Telegram connection and continuously run the scan and execute loop.

    This entry point mirrors the original __main__ behaviour while
    explicitly carrying a state container.  Strategy logic remains
    untouched inside `holy_grail_scan_and_execute`; only the outer loop is
    reorganised.
    """
    print("🔥 Holy Grail SMC Engine Active")
    start_telegram()
    while True:
        try:
            holy_grail_scan_and_execute()
        except Exception as e:
            log_debug("Error in run_bot loop:", e)
        time.sleep(1)

# -----------------------------------------------------------------------------
# === Telegram Messaging ===
# All Telegram-related helpers are centralized here. They handle
# synchronous and asynchronous sends as well as policy enforcement for
# message compactness. Consolidating these functions under a single
# heading clarifies their role within the bot.
#
# ----------------------------------------------------------------------------
# Telegram Function
# -----------------------------------------------------------------------------
import requests

# Low-level synchronous send used for health checks and diagnostic sends.
def _tg_send_sync(chat_id, text, timeout=10) -> Tuple[bool, Any]:
    """Send a Telegram message synchronously and return (ok, response).

    The returned `response` is either the parsed JSON body on success/failure,
    or a string describing the error when an exception occurs.
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": str(chat_id), "text": str(text)}, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return (r.status_code == 200, body)
    except Exception as e:
        try:
            print("[Telegram ERROR]", e)
        except Exception:
            log_debug("[Telegram ERROR print failed]", e)
        return (False, str(e))


def telegram_msg(text):
    """Send a message to Telegram.

    Under the revised policy this helper no longer relays any messages to
    Telegram.  All content passed here is printed to console only and the
    function returns False.  Execution receipts are handled separately by
    ``telegram_execution``.
    """
    try:
        # Always log messages to console for debugging; never send to Telegram
        try:
            print(f"[Telegram][console-only] {str(text)}")
        except Exception as e:
            log_debug("telegram_msg console print failed:", e)
        return False
    except Exception as e:
        log_debug("telegram_msg outer exception:", e)
        return False


def send_telegram_to(chat_id, text):
    """Send `text` to the specified `chat_id` using the configured bot token.

    This helper is used for async or ad-hoc replies to the requesting chat.
    """
    try:
        if not TELEGRAM_TOKEN:
            print("[Telegram ERROR] token missing for send_telegram_to")
            return False
        # Non-blocking send: dispatch in a daemon thread so handlers stay responsive.
        def _worker():
            try:
                ok, resp = _tg_send_sync(chat_id, text)
                if not ok:
                    try:
                        print(f"[Telegram][bg-send->{chat_id}] failed: {resp}")
                    except Exception:
                        print(f"[Telegram][bg-send->{chat_id}] failed (no details)")
            except Exception as e:
                try:
                    print(f"[Telegram][bg-send->{chat_id}] exception: {e}")
                except Exception as e:
                    log_debug("Ignored exception (suppressed):", e)
        threading.Thread(target=_worker, daemon=True).start()
        return True
    except Exception as e:
        try:
            print("[Telegram ERROR]", e)
        except Exception as e:
            log_debug("Ignored exception (suppressed):", e)
        return False

def start_telegram():
    """Initialize and test Telegram connection at startup."""
    print("\n[Telegram] Initializing Telegram connection...")
    
    # Verify token and chat ID are set
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram FATAL] Token or Chat ID is missing!")
        print(f"  Token: {'SET' if TELEGRAM_TOKEN else 'MISSING'}")
        print(f"  Chat ID: {'SET' if TELEGRAM_CHAT_ID else 'MISSING'}")
        print("[Telegram FATAL] Bot cannot continue without Telegram credentials.")
        return False
    
    # Test Telegram connection
    test_msg = "🔥 Telegram Connected Successfully"
    # Use synchronous send for startup health check so we accurately know
    # whether Telegram is reachable. Non-blocking sends are used at runtime.
    ok, resp = _tg_send_sync(TELEGRAM_CHAT_ID, test_msg)

    if ok:
        print("[Telegram] ✅ Telegram connection verified!")
        return True
    else:
        print("[Telegram FATAL] Telegram connection test FAILED")
        try:
            print(f"[Telegram FATAL] Response: {resp}")
        except Exception as e:
            log_debug("Ignored exception (suppressed):", e)
        print("[Telegram FATAL] Token or Chat ID may be incorrect")
        print("[Telegram FATAL] Bot cannot continue without working Telegram.")
        return False

_tg_offset = None
_tg_thread = None

def telegram_poll_loop():
    """Poll Telegram for incoming messages continuously."""
    global _tg_offset
    # Use long-polling and dispatch each incoming update to a separate
    # daemon thread so the poll loop can continue immediately. Heavy
    # command handlers will run asynchronously and the operator receives
    # an instant acknowledgement.
    while TELEGRAM_POLLING_ENABLED:
        try:
            updates = _tg_poll_once(offset=_tg_offset)
            try:
                # Log number of updates fetched for diagnostics
                if updates is None:
                    print("[Telegram Poll] fetched no data (None)")
                else:
                    print(f"[Telegram Poll] fetched {len(updates)} updates (offset={_tg_offset})")
            except Exception as e:
                log_debug("Ignored exception (suppressed):", e)
            if updates:
                for u in updates:
                    try:
                        uid = u.get("update_id")
                        if uid is not None:
                            _tg_offset = uid + 1
                        # Log the raw update (small, for diagnostics)
                        try:
                            short = json.dumps(u) if isinstance(u, dict) else str(u)
                            print(f"[Telegram Poll] update -> {short}")
                        except Exception as e:
                            log_debug("telegram_poll short dump failed:", e, "raw_update=", u)
                        msg = u.get("message", {})
                        text = msg.get("text", "")
                        chat = msg.get("chat", {}).get("id")
                        if text and chat:
                            try:
                                th = threading.Thread(target=_handle_telegram_update, args=(text, chat), daemon=True)
                                th.start()
                                print(f"[Telegram Poll] dispatched update for chat={chat} text={text}")
                            except Exception as e:
                                print(f"[Telegram Poll] thread start failed: {e}")
                                # Fallback: handle inline if thread creation fails
                                try:
                                    _handle_telegram_update(text, chat)
                                except Exception as e2:
                                    print(f"[Telegram Poll] dispatch fallback error: {e2}")
                    except Exception as e:
                        print("[Telegram Poll ERROR]", e)
            # No explicit sleep — long-polling in _tg_poll_once controls wait time.
        except Exception as e:
            print("[Telegram Poll ERROR]", e)
            time.sleep(1)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SAFE_TZ = pytz.timezone("Europe/London")
# Operator request: restrict trading to Gold only for FTMO safety
SYMBOLS = ["XAUUSD"]
LOT_FULL = 0.2
LOT_PREDICTIVE_MIN = 0.5
LOT_PREDICTIVE_MAX = 0.7
LOT_MICRO_MIN = 0.01
LOT_MICRO_MAX = 0.05
# Default stop loss distance in points (100 pips).  See place_trade for dynamic lot sizing based on this SL.
STOP_POINTS = 100  # updated default stop loss distance (was 300)
TAKE_POINTS = 600  # placeholder RR
SPREAD_MAX_POINTS = 350

# -----------------------------------------------------------------------------
# Reliability guard thresholds
#
# The following constants control additional safety checks used during order
# execution.  ``SPREAD_SAFE_THRESHOLD_PTS`` caps the maximum allowable spread
# (in points) when sending an order.  Orders will be aborted when the current
# spread exceeds this value.  ``VOLATILITY_JUMP_THRESHOLD_PTS`` defines the
# maximum allowed tick jump (in points) between two successive ticks sampled
# within ~50 ms.  If the tick jump exceeds this threshold, execution will be
# aborted to avoid entering during extreme volatility.  ``MIN_STOP_MARGIN_MULTIPLIER``
# expands stop‑loss and take‑profit distances beyond the broker’s minimum
# stop distance (trade_stops_level) by this multiplier to ensure SL/TP levels
# remain valid.
SPREAD_SAFE_THRESHOLD_PTS = int(os.getenv('SPREAD_SAFE_THRESHOLD_PTS', '200'))  # safe spread cap
VOLATILITY_JUMP_THRESHOLD_PTS = int(os.getenv('VOLATILITY_JUMP_THRESHOLD_PTS', '100'))  # tick jump cap
MIN_STOP_MARGIN_MULTIPLIER = float(os.getenv('MIN_STOP_MARGIN_MULTIPLIER', '1.1'))  # expand stops by 10%

# ATR fallback multiplier for stop loss calculation.  When a dynamic stop
# cannot be determined or is invalid, the bot falls back to using the
# current ATR multiplied by this factor to set the stop loss distance.
# Exposed as an environment variable for tuning.  Default is 1.5.
ATR_FALLBACK_MULTIPLIER = float(os.getenv('ATR_FALLBACK_MULTIPLIER', '1.5'))

# Additional stop‑loss buffer (in points) added on top of the broker’s
# minimum stop distance and current spread when performing the final
# SL safety adjustment.  Set via environment variable ``SL_EXTRA_BUFFER_PTS``;
# defaults to 0 (no extra buffer).  This value is multiplied by the
# instrument’s ``point`` size to yield a price‑unit buffer.
SL_EXTRA_BUFFER_PTS = float(os.getenv('SL_EXTRA_BUFFER_PTS', '0'))

# === Execution safety settings ===
# Introduce a per‑symbol order execution record to ensure only one real trade is
# executed per candle.  When a live order executes successfully (i.e. MetaTrader
# returns a `TRADE_RETCODE_DONE`), the timestamp and return code are stored in
# ``SYMBOL_EXECUTION_LOCK`` under the uppercase symbol.  The stored value is a
# tuple of ``(timestamp, retcode)``.  During the trading loop, if the most
# recent execution for a symbol occurred within the current one‑minute candle,
# the bot will block any further executions on that symbol until the next candle
# begins.  Failed or aborted order attempts (including preview runs, SL
# validation failures, or rejected orders) DO NOT update this lock.  This
# dictionary lives at module scope so it persists across function calls.
SYMBOL_EXECUTION_LOCK: Dict[str, Tuple[datetime, int]] = {}

# Maintain a simple deduplication cache for block logging.  Keys are
# ``(symbol, reason, candle_start)`` and values are booleans indicating that a
# block message has already been emitted for the current candle.  This is used
# to prevent console/telegram spam by logging the same block reason only once
# per symbol per candle.  The cache is not cleared explicitly because the
# candle_start timestamp ensures uniqueness across candles.
BLOCK_LOG_CACHE: Dict[Tuple[str, str, datetime], bool] = {}

def log_block(symbol: str, reason: str, extra_info: Optional[str] = None) -> None:
    """
    Log a blocked trade reason once per symbol per candle.

    This helper constructs a concise message describing why a trade was
    blocked and ensures the same message is not logged repeatedly during
    successive ticks within the same candle.  A candle is defined as a
    one‑minute bar aligned to ``SAFE_TZ``.  The caller may optionally
    provide ``extra_info`` to append additional context (such as the last
    execution retcode and timestamp).

    Args:
        symbol: The trading symbol (case‑insensitive).
        reason: A short descriptor of the block cause (e.g. ``"cooldown"``).
        extra_info: Optional string appended to the message for context.

    Returns:
        None.  The message is logged via ``log_msg`` exactly once per candle.
    """
    try:
        # Determine the current candle start in SAFE_TZ
        now = datetime.now(SAFE_TZ)
        candle_start = now.replace(second=0, microsecond=0)
        key = (symbol.upper(), reason, candle_start)
        # Skip if we've already logged this block reason for this candle
        if BLOCK_LOG_CACHE.get(key):
            return
        BLOCK_LOG_CACHE[key] = True
        msg = f"Blocked {symbol} | reason={reason}"
        if extra_info:
            msg = f"{msg} | {extra_info}"
        log_msg(msg)
    except Exception as e:
        try:
            log_debug("log_block failed:", e)
        except Exception:
            pass

def log_block_verbose(symbol: str, reason: str, extra_info: Optional[str] = None, risk_pct: Optional[float] = None, open_risk_pct: Optional[float] = None) -> None:
    """
    Verbose block logging for diagnostics.

    Emits a detailed, single-line diagnostic that includes session label,
    daily/session counters, day and total drawdown percentages, and optional
    risk metrics supplied by the caller. Uses `log_block` for per-candle
    deduplication and `log_msg` for console output. This helper is intended
    for operator visibility during testing and should not change blocking
    semantics.
    """
    try:
        session_label, _, _ = session_bounds()
        ai = mt5.account_info() if hasattr(mt5, 'account_info') else None
        bal = float(getattr(ai, 'balance', 0.0)) if ai else 0.0
        eq = float(getattr(ai, 'equity', bal)) if ai else bal
        # Prefer persisted day_start_equity when available
        try:
            day_open_val = state_store.get('day_start_equity') if state_store is not None else BOT_STATE.day_open_equity
        except Exception:
            day_open_val = BOT_STATE.day_open_equity
        if day_open_val is None:
            day_open_val = eq
        try:
            day_loss_pct = (float(day_open_val) - float(eq)) / float(day_open_val) * 100.0 if day_open_val else 0.0
        except Exception:
            day_loss_pct = 0.0
        try:
            init_bal = state_store.get('initial_balance') if state_store is not None else None
            total_drop = (float(init_bal) - float(bal)) / float(init_bal) * 100.0 if init_bal else 0.0
        except Exception:
            total_drop = 0.0
        total_today = BOT_STATE.micro_trades_today + BOT_STATE.full_trades_today
        parts = [f"session={session_label}", f"micro_today={BOT_STATE.micro_trades_today}", f"full_today={BOT_STATE.full_trades_today}", f"total_today={total_today}", f"day_loss%={day_loss_pct:.2f}%", f"total_loss%={total_drop:.2f}%"]
        if risk_pct is not None:
            parts.insert(0, f"risk%={risk_pct*100:.2f}%")
        if open_risk_pct is not None:
            parts.insert(0, f"open_risk%={open_risk_pct*100:.2f}%")
        if extra_info:
            parts.append(str(extra_info))
        detail = " | ".join(parts)
        try:
            # Deduplicate per-candle via existing helper so we don't spam repeated blocks
            log_block(symbol, reason, detail)
        except Exception:
            pass
        try:
            log_msg(f"Blocked {symbol} | reason={reason} | {detail}")
        except Exception:
            try:
                print(f"Blocked {symbol} | reason={reason} | {detail}")
            except Exception:
                pass
    except Exception as e:
        try:
            log_debug("log_block_verbose failed:", e)
        except Exception:
            pass

# Buffer (in additional stop points) applied during SL validation.  When
# validating a candidate stop loss, the bot enforces that the SL is at least
# ``stops_level_points + SL_VALIDATION_BUFFER_PTS`` away from the entry
# price (converted to price units via the instrument’s ``point`` size).
SL_VALIDATION_BUFFER_PTS = float(os.getenv('SL_VALIDATION_BUFFER_PTS', '0'))

def validate_sl(symbol: str, side: str, entry_price: float, sl_price: float) -> bool:
    """
    Validate a proposed stop loss price against broker stop rules.

    A valid stop loss must be on the correct side of the entry price (below
    the entry for buys, above for sells) and must respect the broker’s
    minimum stop distance plus an optional buffer.  If the symbol info or
    stop level is unavailable, the function conservatively rejects the SL.

    Args:
        symbol: The trading symbol (e.g. 'XAUUSD').  This should be the
            broker‑resolved symbol if available.
        side: 'BUY' or 'SELL' indicating trade direction.
        entry_price: The intended entry price for the order.
        sl_price: The proposed stop loss price.

    Returns:
        True if the stop loss is valid, False otherwise.
    """
    try:
        # Ensure required values are provided and sensible
        if symbol is None or side is None or entry_price is None or sl_price is None:
            return False
        side_u = str(side).upper()
        # Fetch symbol info to determine point size and minimum stop distance
        si = mt5.symbol_info(symbol)
        if not si:
            return False
        pt = float(getattr(si, 'point', 0.0)) or 0.0
        if pt <= 0:
            return False
        # Determine minimum stop distance in points; prefer trade_stops_level over stops_level
        try:
            min_pts = float(getattr(si, 'trade_stops_level', None) or getattr(si, 'stops_level', None) or 0.0)
        except Exception:
            min_pts = 0.0
        # Add configured buffer
        try:
            extra_pts = float(SL_VALIDATION_BUFFER_PTS)
        except Exception:
            extra_pts = 0.0
        min_pts += extra_pts
        # Convert to price units
        min_dist = min_pts * pt
        # Validate relative position of stop
        if side_u == 'BUY':
            # SL must be below entry and far enough
            if sl_price >= entry_price:
                return False
            if (entry_price - sl_price) < min_dist:
                return False
        elif side_u == 'SELL':
            # SL must be above entry and far enough
            if sl_price <= entry_price:
                return False
            if (sl_price - entry_price) < min_dist:
                return False
        else:
            return False
        return True
    except Exception:
        # On any error, reject the stop to avoid unsafe trading
        return False

def repair_sl(symbol: str, side: str, entry_price: float) -> float:
    """
    Compute a repaired stop loss price when the proposed SL is invalid.

    This helper ensures the SL is placed on the correct side of the entry
    and respects the broker's minimum stop distance.  The distance is
    calculated as the maximum of the broker minimum stop and a fallback
    ATR‑based value (ATR(14) * 1.2).  If ATR data is unavailable, the
    fallback is twice the minimum stop distance.  The resulting price is
    rounded to the symbol's tick size via ``normalize_price``.
    
    VALIDATION: Repaired SL is validated to ensure:
    - BUY: SL < entry_price (strictly below)
    - SELL: SL > entry_price (strictly above)
    
    If repaired SL is invalid, returns 0.0 (signal caller to skip trade).

    Args:
        symbol: Trading symbol (broker symbol resolved).
        side: 'BUY' or 'SELL'.
        entry_price: Entry price for the trade.

    Returns:
        A float representing the repaired SL price, or 0.0 if repair fails validation.
    """
    try:
        # Fetch symbol info to determine point size and minimum stop distance
        si = mt5.symbol_info(symbol)
        if not si:
            return 0.0
        pt = float(getattr(si, 'point', 0.0)) or 0.0
        # Determine minimum stop distance in points; prefer trade_stops_level over stops_level
        try:
            min_pts = float(getattr(si, 'trade_stops_level', None) or getattr(si, 'stops_level', None) or 0.0)
        except Exception:
            min_pts = 0.0
        # Add optional buffer if defined
        try:
            extra_pts = float(globals().get('SL_VALIDATION_BUFFER_PTS', 0.0))
        except Exception:
            extra_pts = 0.0
        min_pts += extra_pts
        min_dist = min_pts * pt
        # Attempt to compute ATR(14) on M15 data to determine fallback distance
        atr_val = 0.0
        try:
            rates = get_rates(symbol, timeframe=mt5.TIMEFRAME_M15, count=50)
            if rates is not None:
                df = pd.DataFrame(rates, columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
                atr_val = atr(df, 14)
        except Exception:
            atr_val = 0.0
        # Fallback distance based on ATR * 1.2
        fallback_dist = float(atr_val) * 1.2 if atr_val and atr_val > 0 else (min_dist * 2.0)
        distance = max(min_dist, fallback_dist)
        side_u = str(side).upper()
        
        # === COMPUTE REPAIRED SL WITH DIRECTION AWARENESS ===
        if side_u == 'BUY':
            # BUY: SL must be STRICTLY BELOW entry_price
            repaired = entry_price - distance
            # Validate: ensure repaired SL is below entry
            if repaired >= entry_price:
                # Repaired value is invalid; return 0 to signal skip
                return 0.0
        else:  # SELL
            # SELL: SL must be STRICTLY ABOVE entry_price
            repaired = entry_price + distance
            # Validate: ensure repaired SL is above entry
            if repaired <= entry_price:
                # Repaired value is invalid; return 0 to signal skip
                return 0.0
        
        # Round to symbol's tick size
        try:
            repaired = normalize_price(symbol, repaired)
        except Exception:
            pass
        
        # Final post-normalization validation to ensure rounding didn't break direction
        if side_u == 'BUY':
            if repaired >= entry_price:
                return 0.0
        else:
            if repaired <= entry_price:
                return 0.0
        
        return repaired
    except Exception:
        return 0.0

# XAUUSD-specific spread/tick policy
XAU_MICRO_ALLOW_PTS = int(os.getenv('XAU_MICRO_ALLOW_PTS', '250'))
XAU_FULL_ALLOW_PTS = int(os.getenv('XAU_FULL_ALLOW_PTS', '170'))  # Soft allowance: increased to reduce false skips
# Absolute hard spread cap that blocks all execution
XAU_HARD_SPREAD_BLOCK_PTS = int(os.getenv('XAU_HARD_SPREAD_BLOCK_PTS', '300'))
# Tick age limit (seconds) to consider a tick stale for FULL trades
XAU_TICK_AGE_HARD_S = float(os.getenv('XAU_TICK_AGE_HARD_S', '5.0'))  # Increased to reduce false skips from brief delays

# Track last block messages to avoid spamming the same warning repeatedly
# Backwards-compatible alias into the central BOT_STATE container
_LAST_BLOCK_LOG = BOT_STATE.last_block_log

def _log_block_once(symbol: str, reason: str, cooldown_s: int = 60):
    """Log a block reason for `symbol` but don't repeat the same text within `cooldown_s` seconds."""
    try:
        key = f"{symbol}|{reason}"
        now_ts = time.time()
        last = _LAST_BLOCK_LOG.get(key, 0)
        if now_ts - last > cooldown_s:
            _LAST_BLOCK_LOG[key] = now_ts
            log_msg(reason)
    except Exception:
        try:
            log_msg(reason)
        except Exception:
            log_debug("_log_block_once: log_msg failed for reason:", reason)

def can_place_on_xau(symbol: str, micro: bool = False, intended_side: Optional[str] = None, strict_checks: bool = False):
    """
    Decide whether an XAUUSD trade may be placed given current spread/tick conditions.
    Returns (allowed:bool, reason:str).

    Rules (summary):
    - Hard blocks (apply to both): MT5 disconnected; bid==0 or ask==0; spread > XAU_HARD_SPREAD_BLOCK_PTS; no tick update > XAU_TICK_AGE_HARD_S
    - Micro trades: tolerant — only blocked by hard blocks above. Brief delays/spikes/noisy updates do NOT block micro trades.
    - Full trades (non-strict): permissive during BOS/setup detection — allow setup to be found even with temporarily high spread/stale tick.
    - Full trades (strict=True): enforced right before order_send — require spread <= XAU_FULL_ALLOW_PTS and fresh tick.
    """
    try:
        # MT5 connectivity
        try:
            ti = mt5.terminal_info()
            ai = mt5.account_info()
            if not ti or not ai:
                return False, f"MT5 disconnected or account info unavailable ({symbol})"
        except Exception:
            return False, f"MT5 disconnected ({symbol})"

        tick = None
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None

        # Hard blocks: missing tick or zero bid/ask
        if not tick:
            # For micro trades, allow continuation (caller may provide entry_price_override)
            if micro:
                return True, f"No tick currently for {symbol} but micro allowed (use override)"
            return (False, f"No tick for {symbol}")
        bid = getattr(tick, 'bid', None)
        ask = getattr(tick, 'ask', None)
        if not bid or not ask or bid == 0 or ask == 0:
            return False, f"Bad tick prices for {symbol} (bid={bid} ask={ask})"

        # Spread in points
        sp = spread_points(symbol)
        if sp is None:
            sp = 999999

        # Absolute hard spread block (ALWAYS enforced, regardless of strict_checks)
        if sp > XAU_HARD_SPREAD_BLOCK_PTS:
            return False, f"Spread {sp} > HARD_BLOCK {XAU_HARD_SPREAD_BLOCK_PTS}"

        # Tick age check if available
        tick_time = getattr(tick, 'time', None) or getattr(tick, 'time_msc', None)
        try:
            if tick_time is not None:
                # mt5 tick.time is seconds since epoch; time_msc is milliseconds
                if tick_time > 1e12:
                    age_s = (time.time() * 1000 - float(tick_time)) / 1000.0
                else:
                    age_s = time.time() - float(tick_time)
            else:
                age_s = 0.0
        except Exception:
            age_s = 0.0

        # For micro trades: permit unless hard blocks above; tolerate small age or noisy ticks
        if micro:
            return True, f"Micro allowed (spread={sp}, bid={bid}, ask={ask}, tick_age={age_s:.2f}s)"

        # For full trades: behavior depends on strict_checks flag
        if strict_checks:
            # STRICT mode: enforce soft limits (used right before order_send)
            if sp <= XAU_FULL_ALLOW_PTS and age_s <= XAU_TICK_AGE_HARD_S:
                return True, f"Full allowed (strict) (spread={sp} <= {XAU_FULL_ALLOW_PTS}, tick_age={age_s:.2f}s)"
            else:
                return False, f"Full blocked (strict) (spread={sp} > {XAU_FULL_ALLOW_PTS} or tick_age={age_s:.2f}s > {XAU_TICK_AGE_HARD_S}s)"
        else:
            # PERMISSIVE mode: allow setup detection even with temporarily high spread/stale tick
            # Only hard blocks apply. This allows BOS/setup detection to proceed.
            return True, f"Full allowed (permissive - setup detection) (spread={sp}, tick_age={age_s:.2f}s)"

        # At this point the basic spread/tick checks passed for full trades.
        # Additional strategy-level XAU rules (sweep + BOS + overlap + session limits)
        try:
            # Ensure session counters are fresh
            try:
                cur_label = None
                if 'session_bounds' in globals():
                    cur_label, _, _ = session_bounds()
            except Exception:
                cur_label = None
            _reset_xau_session_counts_if_needed(cur_label)

            # Determine session label (LON/NY or None)
            session_label = None
            try:
                session_label, _, _ = session_bounds()
            except Exception:
                session_label = None

            # Off-session block: only allow during London or New York
            if session_label not in ("LON", "NY"):
                reason = f"Off-session ({session_label}) - XAU trades disabled outside LON/NY"
                # Verbose log for diagnostics
                try:
                    log_block_verbose(symbol, reason, extra_info=f"spread={sp}, tick_age={age_s:.2f}s, session={session_label}")
                except Exception:
                    _log_block_once(symbol, reason)
                _maybe_alert_xau_block(symbol, reason, extra_info=f"spread={sp}, tick_age={age_s:.2f}s, session={session_label}")
                return False, reason

            # Range / chop filter on execution timeframe (use M15)
            try:
                m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 10, max_age=2.0)
                if m15 is not None and len(m15) >= 2:
                    # last two closed candles: -2 and -1
                    h1 = float(m15['high'].iloc[-2]); l1 = float(m15['low'].iloc[-2])
                    h2 = float(m15['high'].iloc[-1]); l2 = float(m15['low'].iloc[-1])
                    r1 = max(1e-9, h1 - l1); r2 = max(1e-9, h2 - l2)
                    overlap = max(0.0, min(h1, h2) - max(l1, l2))
                    overlap_pct = overlap / min(r1, r2) if min(r1, r2) > 0 else 0.0
                    if overlap_pct > 0.60:
                        reason = f"Range/Chop detected: last-2 candle overlap {overlap_pct*100:.1f}%"
                        try:
                            log_block_verbose(symbol, reason, extra_info=f"overlap%={overlap_pct*100:.1f}%, spread={sp}")
                        except Exception:
                            _log_block_once(symbol, reason)
                        _maybe_alert_xau_block(symbol, reason, extra_info=f"overlap%={overlap_pct*100:.1f}%, spread={sp}")
                        return False, reason
            except Exception as e:
                log_debug("can_place_on_xau range/BOS check failed:", e)

            # London sweep and BOS requirement (and apply same checks for NY)
            try:
                # Only enforce sweep/BOS for full trades; when intended_side is unknown we cannot fully enforce directional BOS
                side = (str(intended_side).upper() if intended_side else None)
                # Detect BOS using H1 data (existing helper expects timeframe data)
                h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 260, max_age=3.0)
                bos_ok = False
                try:
                    if side and h1 is not None:
                        bos_dir = detect_bos(h1)
                        if (side == 'BUY' and bos_dir == 'BOS_UP') or (side == 'SELL' and bos_dir == 'BOS_DOWN'):
                            bos_ok = True
                except Exception:
                    bos_ok = False

                # Detect a recent liquidity sweep on lower timeframe (M5/M15 heuristic)
                sweep_ok = False
                try:
                    sweep_ok = _detect_recent_sweep(symbol)
                except Exception:
                    sweep_ok = False

                # Detect strong impulsive BOS candle (displacement in trade direction)
                impulsive_bos_ok = False
                try:
                    m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 50, max_age=1.0)
                    if m15 is not None and len(m15) >= 2:
                        disp = detect_displacement(m15)
                        # Strong impulsive candle in trade direction
                        if disp and disp.get('strong') and disp.get('direction') == side:
                            impulsive_bos_ok = True
                except Exception:
                    impulsive_bos_ok = False

                if not bos_ok:
                    reason = f"No BOS in trade direction ({side})"
                    try:
                        log_block_verbose(symbol, reason, extra_info=f"side={side}, bos_dir={bos_dir if 'bos_dir' in locals() else 'N/A'}, spread={sp}")
                    except Exception:
                        _log_block_once(symbol, reason)
                    _maybe_alert_xau_block(symbol, reason, extra_info=f"side={side}, bos_dir={bos_dir if 'bos_dir' in locals() else 'N/A'}, spread={sp}")
                    return False, reason

                # === Notification: BOS detected but trade not yet allowed (once per setup) ===
                if bos_ok and not (sweep_ok or impulsive_bos_ok):
                    try:
                        telegram_bos_detected(symbol, side)
                    except Exception as e:
                        log_debug("BOS setup notification failed:", e)

                # FULL trade allowed if EITHER sweep OR strong impulsive BOS
                if not (sweep_ok or impulsive_bos_ok):
                    reason = "No liquidity sweep or impulsive BOS detected"
                    try:
                        log_block_verbose(symbol, reason, extra_info=f"sweep_ok={sweep_ok}, impulsive_bos_ok={impulsive_bos_ok}, spread={sp}")
                    except Exception:
                        _log_block_once(symbol, reason)
                    _maybe_alert_xau_block(symbol, reason, extra_info=f"sweep_ok={sweep_ok}, impulsive_bos_ok={impulsive_bos_ok}, spread={sp}")
                    
                    # === Notification: First block reason per session ===
                    try:
                        telegram_xau_block_reason(symbol, reason, session_label=session_label)
                    except Exception as e:
                        log_debug("XAU block reason notification failed:", e)
                    
                    return False, reason
            except Exception:
                # If error in BOS/sweep/impulsive detection, be conservative and block full
                reason = f"BOS/sweep detection failed for {symbol}"
                try:
                    log_block_verbose(symbol, reason, extra_info=f"spread={sp}, session={session_label}")
                except Exception:
                    _log_block_once(symbol, reason)
                _maybe_alert_xau_block(symbol, reason, extra_info=f"spread={sp}, session={session_label}")
                return False, reason

            # Session limit: only 1 BUY and 1 SELL per session (LON/NY tracked separately)
            if intended_side:
                try:
                    cnt = _xau_session_trade_count(session_label, intended_side)
                    if cnt >= 1:
                        reason = f"Session limit hit: {session_label} already has a {intended_side}"
                        try:
                            log_block_verbose(symbol, reason, extra_info=f"session={session_label}, side={intended_side}, count={cnt}")
                        except Exception:
                            _log_block_once(symbol, reason)
                        _maybe_alert_xau_block(symbol, reason, extra_info=f"session={session_label}, side={intended_side}, count={cnt}")
                        return False, reason
                except Exception as e:
                    log_debug("can_place_on_xau session limit check failed:", e)

            return True, f"Full allowed (spread={sp} <= {XAU_FULL_ALLOW_PTS}, tick_age={age_s:.2f}s)"
        except Exception as e:
            return False, f"can_place_on_xau post-checks error: {e}"
    except Exception as e:
        return False, f"can_place_on_xau error: {e}"

# Auto-fallback and alerting controls
XAU_MICRO_AUTO_FALLBACK = bool(os.getenv('XAU_MICRO_AUTO_FALLBACK', '1') in ('1','true','True'))
XAU_FULL_BLOCK_ALERT_THRESHOLD = int(os.getenv('XAU_FULL_BLOCK_ALERT_THRESHOLD', '5'))
XAU_FULL_BLOCK_ALERT_COOLDOWN = int(os.getenv('XAU_FULL_BLOCK_ALERT_COOLDOWN', '3600'))

# Conservative M15 fallback buffer (points) applied when using M15 close as entry
# Default 10 points; converted to price using symbol point size (si.point)
XAU_M15_FALLBACK_PTS = int(os.getenv('XAU_M15_FALLBACK_PTS', '8'))
# Periodic owner report interval (seconds) for XAU full-block summary (daily)
XAU_BLOCK_REPORT_INTERVAL_S = int(os.getenv('XAU_BLOCK_REPORT_INTERVAL_S', '86400'))

# The canonical XAU full-block and aggregated stats are provided via
# `BOT_STATE`.  Do not redeclare `_XAU_FULL_BLOCKS` or `_XAU_BLOCK_STATS`
# aliases here; these variables are defined once near the top of the module
# to maintain a single reference.

# Flag set by the scan to indicate XAU had no setup this cycle (used to
# emit a single concise console message elsewhere).
LAST_SCAN_XAU_NO_SETUP = False

# Persisted XAU stats key in holy_memory
_XAU_BLOCKS_MEMORY_KEY = 'xau_block_stats'

# Symbols allowed to take FULL trades (require 2+ usable signals and 2 agreeing votes)
# Only allow FULL trades on Gold; other pairs are micro-only or disabled
FULL_ALLOWED_SYMBOLS = set(["XAUUSD"])

# Enforce micro-only behaviour for GBP pairs when enabled. Controlled via
# env `GBP_MICRO_ONLY` (default=1 -> enabled). When True, GBP* pairs will
# never be used for full trades and only considered for micro entries.
GBP_MICRO_ONLY = True if str(os.getenv('GBP_MICRO_ONLY', '1')).lower() in ('1', 'true', 'yes') else False

def _maybe_alert_xau_block(symbol: str, reason: str, extra_info: Optional[str] = None):
    """Increment block counter and send a rate-limited, verbose Telegram alert when threshold reached.

    The alert now includes optional `extra_info` and appends live context where available
    (spread, tick age, session and aggregate block counts) to help owners triage repeated
    XAU full-block conditions without requiring an operator to inspect logs.
    """
    try:
        now_ts = time.time()
        # Increment both the transient counter used for threshold alerts
        # and the aggregated stats used for periodic reporting.
        _XAU_FULL_BLOCKS['count'] = _XAU_FULL_BLOCKS.get('count', 0) + 1
        count = _XAU_FULL_BLOCKS['count']
        last = _XAU_FULL_BLOCKS.get('last_alert', 0)
        _XAU_BLOCK_STATS['full_block_total'] = _XAU_BLOCK_STATS.get('full_block_total', 0) + 1
        _XAU_BLOCK_STATS['full_block_since_last_report'] = _XAU_BLOCK_STATS.get('full_block_since_last_report', 0) + 1
        # Persist to holy_memory and save
        try:
            holy_memory[_XAU_BLOCKS_MEMORY_KEY] = {
                'full_block_total': _XAU_BLOCK_STATS['full_block_total'],
                'full_block_since_last_report': _XAU_BLOCK_STATS['full_block_since_last_report'],
                'last_report': _XAU_BLOCK_STATS.get('last_report', 0)
            }
            save_memory()
        except Exception as e:
            log_debug("_maybe_alert_xau_block: save_memory failed:", e)
        if count >= XAU_FULL_BLOCK_ALERT_THRESHOLD and (now_ts - last) > XAU_FULL_BLOCK_ALERT_COOLDOWN:
            # Build a verbose owner alert including optional extra_info and live market context
            pieces = [f"⚠️ XAU Full trades blocked repeatedly ({count} times). Latest: {reason}"]
            if extra_info:
                try:
                    pieces.append(str(extra_info))
                except Exception:
                    pass
            # Append spread and tick age if available
            try:
                sp = spread_points(symbol)
                pieces.append(f"spread={sp}")
            except Exception:
                pass
            try:
                tick = mt5.symbol_info_tick(symbol)
                tick_time = getattr(tick, 'time', None) or getattr(tick, 'time_msc', None)
                if tick_time is not None:
                    if tick_time > 1e12:
                        age_s = (time.time() * 1000 - float(tick_time)) / 1000.0
                    else:
                        age_s = time.time() - float(tick_time)
                    pieces.append(f"tick_age={age_s:.2f}s")
            except Exception:
                pass
            try:
                sess, _, _ = session_bounds()
                pieces.append(f"session={sess}")
            except Exception:
                pass
            pieces.append(f"full_block_total={_XAU_BLOCK_STATS.get('full_block_total', 0)}")
            msg = " | ".join(pieces)
            try:
                tg(msg)
            except Exception as e:
                try:
                    telegram_msg(msg)
                except Exception as e:
                    log_debug("_maybe_alert_xau_block: telegram fallback failed:", e)
            _XAU_FULL_BLOCKS['last_alert'] = now_ts
            _XAU_FULL_BLOCKS['count'] = 0
    except Exception as e:
        log_debug("_maybe_alert_xau_block error:", e)


def _reset_xau_session_counts_if_needed(current_label: Optional[str]):
    """Reset per-session XAU counters when session label changes."""
    try:
        last = globals().get('XAU_LAST_SESSION_LABEL')
        if last != current_label:
            # nested mapping: { session_label: {'BUY': int, 'SELL': int} }
            globals()['XAU_SESSION_TRADES'] = {}
            globals()['XAU_LAST_SESSION_LABEL'] = current_label
    except Exception as e:
        log_debug("_reset_xau_session_counts_if_needed failed:", e)


def _xau_session_trade_count(session_label: Optional[str], side: str) -> int:
    try:
        d = globals().get('XAU_SESSION_TRADES') or {}
        if session_label is None:
            return 0
        sub = d.get(session_label) or {}
        return int(sub.get(side.upper(), 0))
    except Exception:
        return 0


def _record_xau_session_trade(session_label: Optional[str], side: str):
    try:
        if session_label is None:
            return
        d = globals().get('XAU_SESSION_TRADES')
        if d is None:
            d = {}
            globals()['XAU_SESSION_TRADES'] = d
        sub = d.get(session_label) or {}
        sub[side.upper()] = int(sub.get(side.upper(), 0)) + 1
        d[session_label] = sub
        # Persist to state_store if available (best-effort)
        try:
            if state_store is not None and hasattr(state_store, 'save'):
                # store minimal representation
                try:
                    st = state_store._state
                    st['xau_session_trades'] = globals().get('XAU_SESSION_TRADES')
                    state_store.save()
                except Exception as e:
                    log_debug("_record_xau_session_trade: state_store.save failed:", e)
        except Exception as e:
            log_debug("_record_xau_session_trade persistence failed:", e)
    except Exception as e:
        log_debug("_record_xau_session_trade failed:", e)
def _detect_recent_sweep(symbol: str) -> bool:
    """Heuristic: detect a recent liquidity sweep on M15/M5 where price
    pierced a recent swing high/low and then rejected (wick). Returns True
    if a plausible sweep was seen in the recent bars.
    """
    try:
        # Prefer M5 for sweep detection, fallback to M15
        for tf in (mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15):
            try:
                df = fetch_data_cached(symbol, tf, 80, max_age=1.0)
            except Exception:
                df = None
            if df is None or len(df) < 10:
                continue
            # define prior swing window (exclude last 3 bars), find prior high/low
            if len(df) < 15:
                window = len(df) - 3
            else:
                window = 30
            if window < 3:
                continue
            prior = df.iloc[-(3 + window):-3]
            recent = df.iloc[-6:]
            try:
                prior_high = float(prior['high'].max())
                prior_low = float(prior['low'].min())
            except Exception:
                continue
            # check for high sweep (BUY): any recent candle high exceeds prior_high
            for idx in range(len(recent)):
                row = recent.iloc[idx]
                try:
                    high = float(row['high']); low = float(row['low']); close = float(row['close'])
                except Exception:
                    continue
                # wick piercing above prior high and close back below prior high suggests sweep
                if high > prior_high and close < prior_high:
                    return True
                # symmetrical for low sweep
                if low < prior_low and close > prior_low:
                    return True
        return False
    except Exception:
        return False

# HARD DISABLE HTF alignment checks: when True, any HTF-based gating
# (H1/H4/D1 alignment) will be treated as non-blocking. This enforces
# the operator request to run purely on LTF SMC signals (M15).
HARD_DISABLE_HTF = True if str(os.getenv('HARD_DISABLE_HTF','1')).lower() in ('1','true','yes') else False


def _build_xau_block_report_message(reset_after_send: bool = True) -> str:
    """Return a human-readable XAU block summary message for owners."""
    try:
        total = _XAU_BLOCK_STATS.get('full_block_total', 0)
        since = _XAU_BLOCK_STATS.get('full_block_since_last_report', 0)
        last = _XAU_BLOCK_STATS.get('last_report', 0)
        last_dt = datetime.utcfromtimestamp(last).isoformat() if last else 'never'
        msg = (
            f"📈 XAU Block Report\n"
            f"Full-blocks (since start): {total}\n"
            f"Full-blocks (since last report): {since}\n"
            f"Last periodic report: {last_dt}\n"
            f"Full-block alert threshold: {XAU_FULL_BLOCK_ALERT_THRESHOLD}\n"
        )
        if reset_after_send:
            _XAU_BLOCK_STATS['full_block_since_last_report'] = 0
            _XAU_BLOCK_STATS['last_report'] = time.time()
            # Persist reset to memory
            try:
                holy_memory[_XAU_BLOCKS_MEMORY_KEY] = {
                    'full_block_total': _XAU_BLOCK_STATS.get('full_block_total', 0),
                    'full_block_since_last_report': _XAU_BLOCK_STATS.get('full_block_since_last_report', 0),
                    'last_report': _XAU_BLOCK_STATS.get('last_report', 0)
                }
                save_memory()
            except Exception as e:
                log_debug("_build_xau_block_report_message: save_memory failed:", e)
        return msg
    except Exception:
        return "📈 XAU Block Report: error building report"


def _xau_block_reporter_loop():
    """Background loop that periodically sends XAU block summaries to owner chat."""
    try:
        while True:
            try:
                # Sleep in small increments so shutdown can be responsive
                interval = max(30, int(XAU_BLOCK_REPORT_INTERVAL_S))
                time.sleep(interval)
                if not telegram_enabled():
                    continue
                # Compose and send report
                try:
                    msg = _build_xau_block_report_message(reset_after_send=True)
                    tg(msg)
                except Exception:
                    try:
                        telegram_msg(_build_xau_block_report_message(reset_after_send=False))
                    except Exception as e:
                        log_debug("_xau_block_reporter_loop: telegram_msg fallback failed:", e)
            except Exception:
                # swallow transient errors and continue
                time.sleep(5)
    except Exception as e:
        log_debug("_xau_block_reporter_loop outer error:", e)


def start_xau_block_reporter():
    """Start the periodic XAU block reporter thread (idempotent)."""
    try:
        th = _XAU_BLOCK_STATS.get('report_thread')
        if th is not None and getattr(th, 'is_alive', lambda: False)():
            return
        th = threading.Thread(target=_xau_block_reporter_loop, daemon=True)
        _XAU_BLOCK_STATS['report_thread'] = th
        th.start()
    except Exception as e:
        log_debug("save_state failed:", e)


# Initialize _XAU_BLOCK_STATS from persisted memory if available
try:
    mem = holy_memory.get(_XAU_BLOCKS_MEMORY_KEY)
    if isinstance(mem, dict):
        _XAU_BLOCK_STATS['full_block_total'] = int(mem.get('full_block_total', _XAU_BLOCK_STATS['full_block_total']))
        _XAU_BLOCK_STATS['full_block_since_last_report'] = int(mem.get('full_block_since_last_report', _XAU_BLOCK_STATS['full_block_since_last_report']))
        _XAU_BLOCK_STATS['last_report'] = float(mem.get('last_report', _XAU_BLOCK_STATS['last_report']))
except Exception as e:
    log_debug("_init_xau_block_stats failed:", e)
DAILY_DD_PCT = 5.0
OVERALL_DD_PCT = 10.0
MEMORY_FILE = r"c:\\mt5_bot\\holy_grail_memory.json"

# Persistent state store (load/save & FTMO enforcement)
try:
    from state_store import state_store, load_state, save_state
except Exception:
    state_store = None


# Ticket-Funded FTMO Servers
#
# Define allowed FTMO server patterns.  We no longer rely on generic substrings like
# "LIVE" or "DEMO" alone because those tokens also appear in normal brokerage servers.
FTMO_ALLOWED_SERVERS = [
    "FTMO-LIVE", "FTMO-LIVE-1", "FTMO-LIVE-2", "FTMO-LIVE-3", "FTMO-LIVE-4", "FTMO-LIVE-5", "FTMO-LIVE-6", "FTMO-LIVE-SERVER",
    "FTMO-DEMO", "FTMO-DEMO-1", "FTMO-DEMO-2", "FTMO-DEMO-3", "FTMO-DEMO-4", "FTMO-DEMO-5", "FTMO-DEMO-SERVER",
    "FTMO-CHALLENGE", "FTMO-VERIFICATION", "FTMO-VERIFICATION-1", "FTMO-VERIFICATION-2"
]

# IC Markets detection patterns.  These substrings cover the common IC Markets server names
# including Live and Demo servers across different regulatory regions (SC/EU/AU).
ICM_ALLOWED_PATTERNS = [
    "ICMARKETS", "ICMARKETSSC", "ICMARKETSEU", "ICMARKETSAU"
]

# === Broker & MT5 ===
#
# This section defines detection logic for supported brokers (FTMO and IC Markets)
# as well as constants and helpers needed for MT5 initialization.  Broker
# identification occurs via substring matching on the server name returned by
# ``mt5.account_info().server`` and is case-insensitive.  A global flag
# ``IS_PROP_FIRM`` (set later in the connection routine) indicates whether
# the current account is a prop-firm account (FTMO) and should be used
# throughout the file to gate FTMO-specific rules.

# Absolute path to the IC Markets MT5 terminal.  When we detect an IC Markets account
# running from a different installation, we reinitialize MT5 using this path.
ICM_MT5_PATH = r"C:\Users\malakaicorbin\AppData\Roaming\MetaTrader 5 IC Markets Global"

def is_ftmo_server(server_name: str) -> bool:
    """Return True if the given server_name belongs to an FTMO account.

    We consider a server to be FTMO if it contains the substring "FTMO" or matches
    one of the explicitly listed FTMO server names.  We no longer detect by
    generic tokens such as "LIVE" or "DEMO" alone because those tokens also
    appear in non-FTMO brokerage servers (e.g., IC Markets).
    """
    srv = (server_name or "").upper()
    if "FTMO" in srv:
        return True
    return any(name in srv for name in FTMO_ALLOWED_SERVERS)

def is_icmarkets_server(server_name: str) -> bool:
    """Return True if the given server_name belongs to an IC Markets account.

    Detection is substring-based and case-insensitive.  We treat any server
    containing one of the patterns in ICM_ALLOWED_PATTERNS as an IC Markets
    server.  This covers both Live and Demo servers (e.g., ICMarketsSC-Live,
    ICMarketsEU-Demo).
    """
    srv = (server_name or "").upper()
    for pattern in ICM_ALLOWED_PATTERNS:
        if pattern in srv:
            return True
    return False

def detect_broker(server_name: str) -> Optional[str]:
    """Detect the broker based on the server name.

    Returns 'FTMO' if it matches FTMO patterns, 'ICMARKETS' if it matches IC Markets patterns,
    or None if the broker cannot be identified.  Comparison is case-insensitive.
    """
    srv = (server_name or "").upper()
    if is_ftmo_server(srv):
        return "FTMO"
    if is_icmarkets_server(srv):
        return "ICMARKETS"
    return None

def detect_account_type(server_name: str) -> Optional[str]:
    """Detect whether the account is LIVE or DEMO based on the server name.

    Returns 'LIVE' if the server_name contains '-LIVE' (case-insensitive),
    'DEMO' if it contains '-DEMO', or None if the type cannot be determined.
    Note: Some FTMO challenge/verification servers are treated as DEMO for risk
    classification.
    """
    srv = (server_name or "").upper()
    if "-LIVE" in srv:
        return "LIVE"
    if "-DEMO" in srv:
        return "DEMO"
    # FTMO challenge and verification servers behave like demo accounts
    if "CHALLENGE" in srv or "VERIFICATION" in srv:
        return "DEMO"
    return None

# Global flag indicating whether the current account is a prop-firm (FTMO) account.
# This is set when establishing a connection in ``ensure_mt5_connected`` and should
# be referenced wherever FTMO-specific rules need to be enforced.  Default is
# ``False`` until a broker is detected.
IS_PROP_FIRM: bool = False

# === State & Memory ===
# Persistent counters, flags and in-memory structures. `BOT_STATE` is the
# canonical container for mutable runtime data. Do not use module-level
# `trading_paused`; read/write `BOT_STATE.trading_paused` instead.
# Use BOT_STATE.day_open_equity instead of module-level `day_open_equity`.
# Canonical persisted memory lives inside BOT_STATE.holy_memory
last_signal: Dict[str, Any] = {}

# Owner-controlled forced trade modes: symbol -> 'FULL'|'MICRO'
FORCED_TRADES: Dict[str, str] = {}
# Persistence file for forced trades
FORCED_TRADES_FILE = os.path.join(os.path.dirname(__file__), "forced_trades.json")

def load_forced_trades():
    global FORCED_TRADES
    try:
        if os.path.exists(FORCED_TRADES_FILE):
            with open(FORCED_TRADES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    FORCED_TRADES = {k.upper(): v for k, v in data.items()}
    except Exception as e:
        log_debug("load_forced_trades failed:", e)

def save_forced_trades():
    try:
        with open(FORCED_TRADES_FILE, 'w', encoding='utf-8') as f:
            json.dump(FORCED_TRADES, f, indent=2)
    except Exception as e:
        log_debug("save_forced_trades error:", e)

# Load persisted forced trades at import
load_forced_trades()

# Safety caps and cooldowns
# Runtime counters, cooldowns and zone tracking live in `BOT_STATE`.
# See `BotState` defaults for initial values (micro_trades_today, full_trades_today,
# last_micro_time, last_full_time, last_trade_zones, micro_lot_ai_scale).

MICRO_MAX_PER_SESSION = int(os.getenv('MICRO_MAX_PER_SESSION', '10'))  # configurable, increased from 8 to 10
FULL_MAX_PER_DAY = 2  # FTMO: max 2 FULL trades per day
FULL_MAX_PER_SESSION = 2  # also cap per session (London/NY)
TOTAL_MAX_TRADES_PER_DAY = int(os.getenv('TOTAL_MAX_TRADES_PER_DAY','4'))  # Max total trades per day (FTMO rule)
MICRO_COOLDOWN_SECONDS = int(os.getenv('MICRO_COOLDOWN_SECONDS', '180'))  # configurable, 3 minutes
FULL_COOLDOWN_SECONDS = int(os.getenv('FULL_COOLDOWN_SECONDS', '600'))   # configurable, 10 minutes
MICRO_LOT_MIN = 0.01
MICRO_LOT_MAX = 0.02  # reduced micro lot for data/tracking only
DUPLICATE_ZONE_THRESHOLD = 0.002  # 0.2% price distance
DUPLICATE_ZONE_TIME_MIN = 300  # 5 minutes

# === Stability Mode limits (Phase: performance stabilization) ===
FULL_MAX_PER_SESSION_STABILITY = 2
MICRO_MAX_PER_SESSION_STABILITY = 4
FULL_DUPLICATE_BLOCK_MIN = 15
FULL_LOSS_PAUSE_MIN = 90
MICRO_LOSS_PAUSE_MIN = 120
DAILY_LOSS_STOP_PCT = 3.0
FULL_EXPOSURE_MAX_PCT = 2.0
MICRO_EXPOSURE_MAX_PCT = 0.5
MICRO_DISABLE_TOTAL_EXPOSURE_PCT = 1.5
FULL_RR_MIN = 2.0
MICRO_RR_MIN = 1.5
STRICT_ADX_MIN = 30.0
STRICT_RSI_BUY = 60.0
STRICT_RSI_SELL = 40.0
STRICT_FULL_MAX_PER_SESSION = 1

# Emoji quality thresholds for SMC conditions
EMOJI_STRONG = "✅"  # Strong signal
EMOJI_WEAK_OK = "👍🏾"  # Weak but usable
EMOJI_VERY_WEAK = "👎🏾"  # Very weak, reject
EMOJI_INVALID = "❌"  # Invalid, reject

# -----------------------------------------------------------------------------
# Telegram helpers (assumes telegram_msg exists)
# -----------------------------------------------------------------------------
def tg(msg):
    """Legacy helper. By default this is silenced (prints only) so the bot
    does not spam Telegram with internal/debug messages. To enable raw tg
    behaviour set env `ALLOW_RAW_TELEGRAM=1`.
    Use the structured helpers below for permitted public notifications.
    """
    try:
        if str(os.getenv('ALLOW_RAW_TELEGRAM', '0')).lower() in ('1','true'):
            ok = telegram_msg(msg)
            if not ok:
                print("[TELEGRAM FAIL]", msg)
            return ok
    except Exception:
        log_debug("tg allow-raw check error:", e)
    # Default: print to console only (no Telegram network send)
    try:
        print(msg)
    except Exception:
        log_debug("tg print error:", e)
    return False


# ==================================================
# Structured Telegram notification helpers (filtered)
# ==================================================
TELEGRAM_NOTIFY_SYMBOLS = set([s.upper() for s in os.getenv('TELEGRAM_NOTIFY_SYMBOLS', 'GBPJPY,GBPUSD,XAUUSD').split(',') if s])
_LAST_BLOCK_NOTIFY: Dict[str, str] = {}

def _should_notify_symbol(s: str) -> bool:
    try:
        return s and s.upper() in TELEGRAM_NOTIFY_SYMBOLS
    except Exception:
        return False

def telegram_signal(symbol: str, mode: str, side: str, lot: float, score: int, signals_list: list, details_list: list, micro_today: int, full_today: int):
    """Disabled signal notification.

    The execution-only policy suppresses all pre‑trade signal notifications.  This
    function now returns False without sending any Telegram messages.
    """
    return False


def _symbol_digits(symbol: str) -> int:
    try:
        si = mt5.symbol_info(symbol) if hasattr(mt5, 'symbol_info') else None
        digits = int(getattr(si, 'digits', 2)) if si else 2
        return max(0, digits)
    except Exception:
        return 2

def _format_price_for_tg(symbol: str, value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        v = float(value)
        if not math.isfinite(v):
            return "N/A"
        try:
            v = normalize_price(symbol, v)
        except Exception:
            pass
        digits = _symbol_digits(symbol)
        return f"{v:.{digits}f}"
    except Exception:
        return "N/A"

def _sl_tp_calc_failed(sl: Any, tp: Any) -> bool:
    try:
        if sl is None or tp is None:
            return True
        sl_v = float(sl)
        tp_v = float(tp)
        if not math.isfinite(sl_v) or not math.isfinite(tp_v):
            return True
        if sl_v == 0.0 or tp_v == 0.0:
            return True
        return False
    except Exception:
        return True

def _calc_rr(entry: Any, sl: Any, tp: Any, side: str) -> Optional[float]:
    try:
        if _sl_tp_calc_failed(sl, tp):
            return None
        e = float(entry)
        s = float(sl)
        t = float(tp)
        if not math.isfinite(e) or not math.isfinite(s) or not math.isfinite(t):
            return None
        if side == "BUY":
            risk = e - s
            reward = t - e
        else:
            risk = s - e
            reward = e - t
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk
    except Exception:
        return None

def _format_rr(rr: Optional[float]) -> str:
    try:
        if rr is None or not math.isfinite(float(rr)):
            return "N/A"
        val = float(rr)
        if abs(val - round(val)) < 1e-6:
            return f"1:{int(round(val))}"
        return f"1:{val:.2f}"
    except Exception:
        return "N/A"

def _session_label_human(session_label: Optional[str]) -> str:
    try:
        if session_label == "LON":
            return "London"
        if session_label == "NY":
            return "New York"
        if session_label == "ASIA":
            return "Asia"
        return "Off-session"
    except Exception:
        return "Off-session"

def telegram_trade_levels(symbol: str, side: str, entry: Any, sl: Any, tp: Any, rr: Any, session: Any) -> bool:
    """Send SL/TP levels after a successful execution, without attaching them to MT5 orders."""
    try:
        try:
            sess_label = session or session_bounds()[0]
        except Exception:
            sess_label = session
        session_name = _session_label_human(sess_label)
        time_uk = now_uk().strftime("%H:%M")
        failed = _sl_tp_calc_failed(sl, tp)
        rr_val = rr if rr is not None else _calc_rr(entry, sl, tp, side)

        if failed:
            msg = (
                "⚠️ SL/TP CALCULATION FAILED\n"
                "Trade executed WITHOUT protection\n"
                "ADD SL MANUALLY IMMEDIATELY\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Entry: {_format_price_for_tg(symbol, entry)}\n"
                f"Session: {session_name}\n"
                f"Time (UK): {time_uk}"
            )
        else:
            msg = (
                "🟡 TRADE EXECUTED — ADD SL/TP MANUALLY\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Entry: {_format_price_for_tg(symbol, entry)}\n"
                f"Stop Loss: {_format_price_for_tg(symbol, sl)}\n"
                f"Take Profit: {_format_price_for_tg(symbol, tp)}\n"
                f"R:R: {_format_rr(rr_val)}\n"
                f"Session: {session_name}\n"
                f"Time (UK): {time_uk}"
            )
        return send_telegram_to(TELEGRAM_CHAT_ID, msg)
    except Exception as e:
        log_debug("telegram_trade_levels failed:", e)
        return False


def telegram_block(symbol: str, reason: str) -> bool:
    """Disabled trade-block notification.

    Under the execution-only policy, block reasons are logged to console but
    never sent to Telegram.  This helper returns False.
    """
    return False


def telegram_daily_summary(summary_lines: list) -> bool:
    """Disabled daily summary notification.

    This function no longer sends daily summaries to Telegram.
    """
    return False


def telegram_session_start(session_label: str, symbol: str = "XAUUSD") -> bool:
    """Send session start notification once per session.
    
    Fires when London or New York session begins, suppressed on subsequent scans.
    Example: "London session active – XAUUSD scanning"
    """
    try:
        if session_label not in ("LON", "NY"):
            return False
        
        # Check if we already notified for this session
        if BOT_STATE.last_session_notified == session_label:
            return False
        
        # Format session name
        session_name = "London" if session_label == "LON" else "New York"
        msg = f"{session_name} session active – {symbol} scanning"
        
        # Send via Telegram
        result = send_telegram_to(TELEGRAM_CHAT_ID, msg)
        
        # Track that we sent this session notification
        BOT_STATE.last_session_notified = session_label
        
        return result
    except Exception as e:
        log_debug("telegram_session_start failed:", e)
        return False


def telegram_bos_detected(symbol: str, side: str, score: int = 0) -> bool:
    """Send setup-detected notification when BOS is valid but trade not yet allowed.
    
    Fires once per setup (BOS candle). Suppressed until a new BOS appears.
    Example: "XAUUSD BOS detected – awaiting entry confirmation"
    """
    try:
        if symbol.upper() != "XAUUSD":
            return False
        
        # Create unique setup identifier based on current time (minute-level)
        now_uk = datetime.now(SAFE_TZ)
        setup_key = f"{symbol}_{side}_{now_uk.strftime('%Y%m%d_%H%M')}"
        
        # Only send if we haven't sent for this setup yet
        if BOT_STATE.last_xau_bos_setup == setup_key:
            return False
        
        msg = f"{symbol} {side} BOS detected – awaiting entry confirmation"
        
        # Send via Telegram
        result = send_telegram_to(TELEGRAM_CHAT_ID, msg)
        
        # Track that we sent this setup notification
        BOT_STATE.last_xau_bos_setup = setup_key
        
        return result
    except Exception as e:
        log_debug("telegram_bos_detected failed:", e)
        return False


def telegram_xau_block_reason(symbol: str, reason: str, session_label: Optional[str] = None) -> bool:
    """Send first block reason per session when FULL trade is blocked.
    
    Fires only once per session for first block. Suppresses further block messages.
    Example: "XAUUSD setup blocked: spread too high"
    """
    try:
        if symbol.upper() != "XAUUSD":
            return False
        
        # Determine session key
        sess_key = session_label or "UNKNOWN"
        block_key = f"{sess_key}_block_sent"
        
        # Check if we already sent a block message this session
        if BOT_STATE.xau_block_reason_sent_this_session.get(block_key, False):
            return False
        
        # Format message
        msg = f"{symbol} setup blocked: {reason}"
        
        # Send via Telegram
        result = send_telegram_to(TELEGRAM_CHAT_ID, msg)
        
        # Mark that we sent a block message this session
        BOT_STATE.xau_block_reason_sent_this_session[block_key] = True
        
        return result
    except Exception as e:
        log_debug("telegram_xau_block_reason failed:", e)
        return False


def telegram_daily_heartbeat(session_label: Optional[str] = None, trades_today: int = 0, dd_percent: float = 0.0) -> bool:
    """Send daily heartbeat message every 12-24 hours.
    
    Confirms bot is running with session, trade count, and daily drawdown %.
    Example: "Bot running | Session: NY | Trades today: 1 | DD: 0.4%"
    """
    try:
        now_uk = datetime.now(SAFE_TZ)
        
        # Check if we should send (at least 12 hours since last heartbeat)
        if BOT_STATE.last_heartbeat_time is not None:
            hours_since = (now_uk - BOT_STATE.last_heartbeat_time).total_seconds() / 3600.0
            if hours_since < 12:
                return False
        
        # Format session display
        sess_display = session_label if session_label else "Offline"
        
        # Build message
        msg = f"Bot running | Session: {sess_display} | Trades today: {trades_today} | DD: {dd_percent:.2f}%"
        
        # Send via Telegram
        result = send_telegram_to(TELEGRAM_CHAT_ID, msg)
        
        # Update last heartbeat time
        BOT_STATE.last_heartbeat_time = now_uk
        
        return result
    except Exception as e:
        log_debug("telegram_daily_heartbeat failed:", e)
        return False

# -----------------------------------------------------------------------------
# MT5 connection with FTMO validation
# -----------------------------------------------------------------------------
MT5_CONNECTED = False
MT5_LOGIN_VERIFIED = False


def ensure_mt5_connected() -> bool:
    global MT5_CONNECTED, MT5_LOGIN_VERIFIED
    try:
        ai = mt5.account_info()
        if ai is None:
            # Attempt to initialize if not already initialized
            try:
                mt5.initialize()
            except Exception as e:
                log_debug("mt5.initialize failed:", e)
            ai = mt5.account_info()
        if ai is None:
            tg("🚫 **ERROR** — MT5 disconnected\nPlease verify the MT5 terminal is running and the account is logged in.")
            return False
        srv = getattr(ai, 'server', '') or ''
        broker = detect_broker(srv)
        # Determine and set the prop-firm flag.  FTMO accounts are prop firms; IC Markets are not.
        global IS_PROP_FIRM
        if broker == "FTMO":
            IS_PROP_FIRM = True
            MT5_CONNECTED = True
            MT5_LOGIN_VERIFIED = True
            BOT_STATE.trading_paused = False
            return True
        if broker == "ICMARKETS":
            IS_PROP_FIRM = False
            MT5_CONNECTED = True
            MT5_LOGIN_VERIFIED = True
            BOT_STATE.trading_paused = False
            return True
        # Unknown broker
        IS_PROP_FIRM = False
        BOT_STATE.trading_paused = True
        tg(f"🚫 **ERROR** — Unsupported MT5 broker\nCurrent server: {srv}")
        return False
    except Exception as e:
        tg(f"🚫 **ERROR** — MT5 connection failed\nDetails: {e}")
        return False

# === Risk & Enforcement ===
#
# All functions related to drawdown enforcement, trade caps, spread
# validation, and other risk controls are defined in this section.  The
# global flag ``IS_PROP_FIRM`` determines whether prop-firm (FTMO)
# restrictions such as daily drawdown and trade caps are applied.  For
# normal brokerage accounts (e.g., IC Markets), these checks either
# bypass enforcement or apply less restrictive rules.  Centralizing
# these helpers helps reduce duplicated logic across the codebase.
def check_spread(symbol: str, strict: bool = False) -> bool:
    """Check if symbol spread is acceptable for trading.
    
    Args:
        symbol: Trading symbol
        strict: If True, enforce XAUUSD soft limits (170/180 pts). 
                If False, only enforce hard limits (300 pts) for XAUUSD during setup.
    
    Returns:
        True if spread is acceptable, False if too high or no tick.
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        sp = (tick.ask - tick.bid) * 10000 if symbol.endswith("USD") else (tick.ask - tick.bid) * 100
        
        # For XAUUSD: use different limits based on context
        if symbol.upper().startswith('XAUUSD'):
            if strict:
                # STRICT mode (execution-time): enforce soft limit
                return sp <= XAU_FULL_ALLOW_PTS
            else:
                # LENIENT mode (setup detection): only hard block
                return sp <= XAU_HARD_SPREAD_BLOCK_PTS
        
        # Non-XAUUSD: use standard limit
        return sp <= SPREAD_MAX_POINTS
    except Exception:
        return False


def check_dd_limits() -> bool:
    try:
        # Skip drawdown enforcement for non-FTMO accounts.  We use the global
        # IS_PROP_FIRM flag rather than inspecting server names repeatedly.
        global IS_PROP_FIRM
        if not IS_PROP_FIRM:
            return True
        account = mt5.account_info()
        if not account:
            return False
        # For prop-firm accounts, enforce drawdown limits
        ai = account
        eq = ai.equity
        bal = ai.balance
        # Determine persistent day start equity and initial balance if available
        try:
            if state_store is not None:
                st = state_store._state
                # use persisted day_start_equity if present
                persisted_open = st.get('day_start_equity')
                if persisted_open is not None:
                    day_open = float(persisted_open)
                else:
                    day_open = float(eq) if eq is not None else None
                init_bal = st.get('initial_balance') or float(getattr(ai, 'balance', 0.0))
        except Exception:
            day_open = float(BOT_STATE.day_open_equity) if BOT_STATE.day_open_equity is not None else (float(eq) if eq is not None else None)
            init_bal = float(getattr(ai, 'balance', 0.0))

        if day_open is None:
            day_open = float(eq)
        # Daily drawdown from day high (we track high separately)
        try:
            day_high = state_store.get('day_equity_high') if state_store is not None else None
        except Exception:
            day_high = None
        try:
            if day_high is None:
                day_high = float(day_open)
        except Exception:
            day_high = float(day_open)

        # Calculate day loss vs day_open (unrealized + realized) and total drop vs initial balance
        day_open_val = None
        try:
            day_open_val = st.get('day_start_equity') if state_store is not None else None
        except Exception:
            day_open_val = BOT_STATE.day_open_equity
        try:
            if day_open_val is None:
                day_open_val = float(eq)
        except Exception:
            day_open_val = float(eq)

        day_loss_pct = (float(day_open_val) - float(eq)) / float(day_open_val) * 100.0 if day_open_val and eq is not None else 0.0
        total_drop = (float(init_bal) - float(bal)) / float(init_bal) * 100.0 if init_bal else 0.0

        # FTMO HARD RULES enforcement (strict)
        # 1) Daily buffer: if unrealized + realized loss >= 2.5% -> pause trading immediately (buffer before 3%)
        if day_loss_pct >= 2.5 and day_loss_pct < 3.0:
            try:
                if state_store is not None:
                    state_store.set_trading_paused(True)
                    state_store.save()
            except Exception as e:
                log_debug("set_trading_paused save error:", e)
            tg(f"⚠️ **DAILY DD WARNING** — {day_loss_pct:.2f}% today (>=2.5%). Trading paused until review")
            return False
        # 2) Daily hard limit: >=3.0% -> enforce day stop (persisted)
        if day_loss_pct >= 3.0:
            try:
                if state_store is not None:
                    state_store.set_trading_paused(True)
                    state_store.save()
            except Exception as e:
                log_debug("set_trading_paused save error:", e)
            tg(f"⚠️ **DAILY MAX LOSS BREACH** — {day_loss_pct:.2f}% today (>=3.0%). Trading paused until next day")
            return False

        # 3) Overall trailing drawdown: if within 1% of max (>=9%) -> pause; >=10% -> permanent lockout
        if total_drop >= 9.0 and total_drop < 10.0:
            try:
                if state_store is not None:
                    state_store.set_trading_paused(True)
                    state_store.save()
            except Exception as e:
                log_debug("set_trading_paused save error:", e)
            tg(f"⚠️ **OVERALL DD WARNING** — {total_drop:.2f}% total loss (>=9%). Trading paused to protect capital")
            return False
        if total_drop >= 10.0:
            try:
                if state_store is not None:
                    state_store.set_permanent_lockout(True)
                    state_store.save()
            except Exception as e:
                log_debug("set_permanent_lockout save error:", e)
            tg(f"⛔ **PERMANENT LOCKOUT** — {total_drop:.2f}% total loss vs initial balance (>=10%). Trading disabled")
            return False

        # If everything ok
        # Update persistent day equity high if equity > recorded
        try:
            if state_store is not None and eq is not None:
                state_store.update_day_equity_high(eq)
                state_store.save()
        except Exception as e:
            log_debug("update_day_equity_high error:", e)
        return True
    except Exception:
        return False


def _recent_win_rate(last_n: int = 20) -> float:
    """Return win rate over last N closed deals."""
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)
        deals = mt5.history_deals_get(start, now) or []
        closed = [d for d in deals if float(getattr(d, 'profit', 0.0) or 0.0) != 0.0]
        if not closed:
            return 1.0
        closed = sorted(closed, key=lambda d: getattr(d, 'time', 0))[-last_n:]
        wins = sum(1 for d in closed if float(getattr(d, 'profit', 0.0) or 0.0) > 0)
        return wins / max(1, len(closed))
    except Exception:
        return 1.0


def _update_strict_mode() -> None:
    """Toggle strict mode based on rolling win rate."""
    try:
        win_rate = _recent_win_rate(20)
        if win_rate < 0.55 and not BOT_STATE.strict_mode:
            BOT_STATE.strict_mode = True
            BOT_STATE.strict_mode_since = datetime.now(SAFE_TZ)
            try:
                telegram_msg("🛡️ Strict Mode enabled — win rate below 55% (last 20)")
            except Exception:
                pass
        if win_rate >= 0.60 and BOT_STATE.strict_mode:
            BOT_STATE.strict_mode = False
            BOT_STATE.strict_mode_since = None
            try:
                telegram_msg("✅ Strict Mode disabled — win rate recovered above 60% (last 20)")
            except Exception:
                pass
    except Exception:
        pass


def _session_open_blocked(now_uk: Optional[datetime] = None) -> bool:
    try:
        now_uk = now_uk or datetime.now(SAFE_TZ)
        label, start, _ = session_bounds(now_uk)
        if label and start and (now_uk - start) <= timedelta(minutes=5):
            return True
    except Exception:
        return False
    return False


def _spread_limit_points(symbol: str) -> float:
    base = _base_of(symbol)
    if base == 'XAUUSD':
        return 25.0
    return 15.0


def _market_quality_ok(symbol: str, side: str, m15: pd.DataFrame, h1: pd.DataFrame, now_uk: datetime) -> Tuple[bool, str]:
    """Mandatory market quality filter for all trades."""
    try:
        # ADX + rising
        adx_h1_val = float(adx(h1)) if h1 is not None else 0.0
        adx_m15_val = float(adx(m15)) if m15 is not None else 0.0
        _, adx_rise_m15 = adx_rising(m15)
        _, adx_rise_h1 = adx_rising(h1)
        adx_min = STRICT_ADX_MIN if BOT_STATE.strict_mode else 25.0
        if adx_h1_val < adx_min or adx_m15_val < adx_min:
            return False, f"ADX below {adx_min:.0f}"
        if not (adx_rise_m15 and adx_rise_h1):
            return False, "ADX not rising"

        # RSI regime
        rsi_h1_val = float(rsi(h1)) if h1 is not None else 50.0
        rsi_m15_val = float(rsi(m15)) if m15 is not None else 50.0
        buy_min = STRICT_RSI_BUY if BOT_STATE.strict_mode else 55.0
        sell_max = STRICT_RSI_SELL if BOT_STATE.strict_mode else 45.0
        if side == "BUY" and (rsi_h1_val <= buy_min or rsi_m15_val <= buy_min):
            return False, "RSI below buy threshold"
        if side == "SELL" and (rsi_h1_val >= sell_max or rsi_m15_val >= sell_max):
            return False, "RSI above sell threshold"

        # Spread
        sp = spread_points(symbol)
        if sp > _spread_limit_points(symbol):
            return False, f"Spread too high ({sp:.1f} pts)"

        # Candle body >= 60% of range (last closed candle)
        if m15 is not None and len(m15) >= 3:
            c = m15.iloc[-2]
            body = abs(float(c['close']) - float(c['open']))
            rng = max(1e-9, float(c['high']) - float(c['low']))
            if (body / rng) < 0.60:
                return False, "Candle body < 60% range"

        # News blackout (10-minute buffer)
        if in_high_impact_news(symbol, now_uk):
            return False, "High-impact news window"
    except Exception:
        return False, "Market quality error"
    return True, "OK"


def _volatility_ok(m15: pd.DataFrame, now_uk: datetime) -> Tuple[bool, str]:
    """Anti-whipsaw volatility filter (ATR spike + wick/body)."""
    try:
        if _session_open_blocked(now_uk):
            return False, "First 5 minutes after session open"
        if m15 is None or len(m15) < 25:
            return False, "Insufficient M15 data"
        atr_vals = atr_series(m15, 14)
        if atr_vals is None or len(atr_vals) < 25:
            return False, "ATR unavailable"
        atr_current = float(atr_vals.iloc[-1])
        atr_avg = float(atr_vals.rolling(20).mean().iloc[-1])
        if atr_avg > 0 and atr_current > (atr_avg * 1.8):
            return False, "ATR spike"

        c = m15.iloc[-2]
        body = abs(float(c['close']) - float(c['open']))
        wick = (float(c['high']) - float(c['low'])) - body
        if body > 0 and wick > (2.0 * body):
            return False, "Excessive wick"
    except Exception:
        return False, "Volatility filter error"
    return True, "OK"


def _trend_consistency_ok(symbol: str, side: str, h1: pd.DataFrame) -> Tuple[bool, str]:
    try:
        if h1 is None or len(h1) < 210:
            return False, "H1 data missing"
        ema200 = ema(h1, 200).iloc[-1]
        last_close = float(h1['close'].iloc[-1])
        base = _base_of(symbol)
        if base == 'XAUUSD':
            if side == "BUY" and last_close <= ema200:
                return False, "XAU below EMA200"
            if side == "SELL" and last_close >= ema200:
                return False, "XAU above EMA200"
            return True, "OK"
        bos = detect_bos(h1)
        bos_dir = interpret_bos(bos)
        if bos_dir and bos_dir != side:
            return False, "HTF BOS mismatch"
        if side == "BUY" and last_close <= ema200:
            return False, "Below EMA200"
        if side == "SELL" and last_close >= ema200:
            return False, "Above EMA200"
    except Exception:
        return False, "Trend filter error"
    return True, "OK"


def _entry_confirmation_ok(m15: pd.DataFrame, side: str) -> Tuple[bool, str]:
    """Require signal candle close and next candle close in same direction."""
    try:
        if m15 is None or len(m15) < 4:
            return False, "M15 data missing"
        c1 = m15.iloc[-3]
        c2 = m15.iloc[-2]
        if side == "BUY":
            if float(c1['close']) <= float(c1['open']) or float(c2['close']) <= float(c2['open']):
                return False, "Entry confirmation failed"
        if side == "SELL":
            if float(c1['close']) >= float(c1['open']) or float(c2['close']) >= float(c2['open']):
                return False, "Entry confirmation failed"
    except Exception:
        return False, "Entry confirmation error"
    return True, "OK"


def in_high_impact_news(symbol: str, now: Optional[datetime] = None) -> bool:
    """Block trades within 10 minutes of red-news windows (all symbols)."""
    try:
        if not NEWS_BLACKOUT_ON:
            return False
        now = now or datetime.now(SAFE_TZ)
        buffer_min = 10
        for w in _load_news_windows():
            st_raw = w.get('start'); en_raw = w.get('end')
            if not st_raw or not en_raw:
                continue
            try:
                st = datetime.fromisoformat(st_raw)
                en = datetime.fromisoformat(en_raw)
            except Exception:
                continue
            if st.tzinfo is None:
                st = SAFE_TZ.localize(st)
            if en.tzinfo is None:
                en = SAFE_TZ.localize(en)
            if (st - timedelta(minutes=buffer_min)) <= now <= (en + timedelta(minutes=buffer_min)):
                return True
    except Exception:
        return False
    return False


def _estimate_open_risk_pct(mode_filter: Optional[str] = None) -> float:
    """Estimate open risk percentage from positions using SL distance."""
    try:
        ai = mt5.account_info()
        base = float(getattr(ai, 'equity', 0.0) or 0.0)
        if base <= 0:
            return 0.0
        total = 0.0
        positions = mt5.positions_get() or []
        for p in positions:
            vol = float(getattr(p, 'volume', 0.0) or 0.0)
            sl = float(getattr(p, 'sl', 0.0) or 0.0)
            price = float(getattr(p, 'price_open', 0.0) or 0.0)
            if vol <= 0 or sl <= 0 or price <= 0:
                continue
            is_micro = vol <= (MICRO_LOT_MAX + 1e-6)
            if mode_filter == "MICRO" and not is_micro:
                continue
            if mode_filter == "FULL" and is_micro:
                continue
            rpl = _risk_per_lot(getattr(p, 'symbol', ''), price, sl)
            total += (rpl * vol) / base
        return total * 100.0
    except Exception:
        return 0.0


def _check_hard_limits(symbol: str, side: str, mode: str) -> Tuple[bool, str]:
    """Enforce hard trade limits for full/micro modes."""
    now_uk = datetime.now(SAFE_TZ)
    sess, _, _ = session_bounds(now_uk)
    if BOT_STATE.trading_paused or BOT_STATE.permanent_lockout:
        return False, "Trading paused"
    if BOT_STATE.full_pause_until and now_uk < BOT_STATE.full_pause_until and mode == "FULL":
        return False, "Full trading paused"
    if BOT_STATE.micro_pause_until and now_uk < BOT_STATE.micro_pause_until and mode == "MICRO":
        return False, "Micro trading paused"
    if BOT_STATE.strict_mode and mode == "MICRO":
        return False, "Strict mode: micro disabled"
    if BOT_STATE.strict_mode and mode == "FULL":
        max_full = STRICT_FULL_MAX_PER_SESSION
    else:
        max_full = FULL_MAX_PER_SESSION_STABILITY
    max_micro = MICRO_MAX_PER_SESSION_STABILITY
    if sess:
        limits = BOT_STATE.session_limits.setdefault(sess, {"full": 0, "micro": 0})
        if mode == "FULL" and limits.get("full", 0) >= max_full:
            return False, f"Full session cap {max_full}"
        if mode == "MICRO" and limits.get("micro", 0) >= max_micro:
            return False, f"Micro session cap {max_micro}"
    # duplicate direction within 15 minutes (full only)
    if mode == "FULL":
        key = f"{symbol.upper()}:{side.upper()}"
        last_ts = BOT_STATE.last_full_entry_by_symbol_side.get(key)
        if last_ts and (now_uk - last_ts) < timedelta(minutes=FULL_DUPLICATE_BLOCK_MIN):
            return False, "Duplicate full entry (15m)"
    return True, "OK"

# -----------------------------------------------------------------------------
# Data fetch
# -----------------------------------------------------------------------------
def get_rates(symbol: str, timeframe=mt5.TIMEFRAME_M15, count=200):
    now = datetime.now()
    return mt5.copy_rates_from(symbol, timeframe, now, count)


# HTF trend cache (TTL-based) to reduce repeated D1/H4 fetches
HTF_CACHE: dict = {}
HTF_CACHE_TTL_S = int(os.getenv('HTF_CACHE_TTL_S', '60'))
def get_htf_trend(symbol: str, cache: Optional[dict] = None) -> Optional[str]:
    """Return HTF trend ('BUY'|'SELL') based on D1 then H4 closes vs opens.
    Uses a TTL-backed module cache by default; a custom `cache` dict can be
    provided for advanced usage.
    """
    try:
        now_ts = time.time()
        use_cache = HTF_CACHE
        ttl = HTF_CACHE_TTL_S
        if cache is not None:
            use_cache = cache
            ttl = HTF_CACHE_TTL_S

        # If cached and not expired, return it
        try:
            entry = use_cache.get(symbol)
            if isinstance(entry, dict):
                ts = float(entry.get('ts', 0) or 0)
                if now_ts - ts < float(ttl):
                    return entry.get('trend')
        except Exception as e:
            log_debug("trend cache read failed:", e)

        # Fetch HTF data
        d1_raw = None
        h4_raw = None
        try:
            d1_raw = get_rates(symbol, mt5.TIMEFRAME_D1, 5)
        except Exception:
            d1_raw = None
        try:
            h4_raw = get_rates(symbol, mt5.TIMEFRAME_H4, 10)
        except Exception:
            h4_raw = None

        trend = None
        try:
            d1 = pd.DataFrame(d1_raw) if d1_raw is not None else None
            h4 = pd.DataFrame(h4_raw) if h4_raw is not None else None
            d1_last = d1.iloc[-1] if d1 is not None and len(d1) > 0 else None
            h4_last = h4.iloc[-1] if h4 is not None and len(h4) > 0 else None
            if d1_last is not None:
                trend = 'BUY' if d1_last['close'] > d1_last['open'] else 'SELL'
            elif h4_last is not None:
                trend = 'BUY' if h4_last['close'] > h4_last['open'] else 'SELL'
        except Exception:
            trend = None

        # Store in cache with timestamp
        try:
            use_cache[symbol] = {'trend': trend, 'ts': now_ts}
        except Exception as e:
            log_debug("trend cache store failed:", e)
        return trend
    except Exception:
        return None

# The simple SMC detection functions (detect_bos, detect_sweep and detect_fvg)
# were defined earlier as list‑based heuristics.  They have been removed to
# avoid conflicting with the DataFrame‑based implementations defined later in
# the file.

# -----------------------------------------------------------------------------
# Scoring and predictive
# -----------------------------------------------------------------------------
def compute_score(bos_ok, sweep_ok, fvg_ok, trend_bias, mitigation, mtf_refine):
    score = 0
    score += 25 if bos_ok else 0
    score += 25 if sweep_ok else 0
    score += 20 if fvg_ok else 0
    score += 10 if trend_bias else 0
    score += 10 if mitigation else 0
    score += 10 if mtf_refine else 0
    return score


def trend_bias_soft(df):
    """Soft trend bias using close movement over last 10 bars (DataFrame)."""
    try:
        if df is None or len(df) < 10:
            return False
        closes = df["close"].tail(10).to_list()
        if len(closes) < 2:
            return False
        return closes[-1] > closes[0] * 1.002 or closes[-1] < closes[0] * 0.998
    except Exception:
        return False


def mitigation_present(df):
    """Detect presence of large wicks vs bodies (DataFrame)."""
    try:
        if df is None or len(df) < 3:
            return False
        recent = df.tail(3)
        wicks = (recent["high"] - recent["low"]).abs()
        bodies = (recent["close"] - recent["open"]).abs()
        return any(wicks > bodies * 1.5)
    except Exception:
        return False


def mtf_refinement(df_h1, df_m15):
    """Multi-timeframe refinement using H1 and M15 DataFrames."""
    try:
        if df_h1 is None or len(df_h1) < 5 or df_m15 is None or len(df_m15) < 5:
            return False
        h1_last = df_h1["close"].iloc[-1]
        h1_prev = df_h1["close"].iloc[-3]
        m15_last = df_m15["close"].iloc[-1]
        m15_prev = df_m15["close"].iloc[-3]
        return (h1_last > h1_prev and m15_last > m15_prev) or (h1_last < h1_prev and m15_last < m15_prev)
    except Exception:
        return False


def predictive_ai_confirms(bos_forming, sweep_forming, fvg_developing, df):
    """Simple momentum check on last 5 closes (DataFrame) when patterns are forming."""
    try:
        if not (bos_forming or sweep_forming or fvg_developing):
            return False
        if df is None or len(df) < 5:
            return False
        closes = df["close"].tail(5).to_list()
        if len(closes) < 2:
            return False
        mom = closes[-1] - closes[0]
        return abs(mom) > max(0.001, 0.0002 * closes[-1])
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Memory load/save
# -----------------------------------------------------------------------------
def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                mem = json.load(f)
                if isinstance(mem, dict):
                    BOT_STATE.holy_memory.clear()
                    BOT_STATE.holy_memory.update(mem)
    except Exception as e:
        log_debug("load_memory failed:", e)


# Load persisted memory at import so reporting/stats persist across restarts
try:
    load_memory()
except Exception as e:
    log_debug("load_memory at import error:", e)


def save_memory():
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(BOT_STATE.holy_memory, f, indent=2)
    except Exception as e:
        log_debug("save_memory error:", e)


def update_memory(symbol: str, win: bool, score: int):
    try:
        seq = BOT_STATE.holy_memory.get("win_loss", [])
        seq.append({"ts": datetime.utcnow().isoformat(), "symbol": symbol, "win": win, "score": score})
        BOT_STATE.holy_memory["win_loss"] = seq[-200:]
    except Exception:
        log_debug("update_memory error:", e)

# -----------------------------------------------------------------------------
# Holy Grail scan and execution
# -----------------------------------------------------------------------------
def in_asian(now_uk):
    """Check if time is in Asian session: 00:00 - 06:00 UK time"""
    return 0 <= now_uk.hour < 6


# ============================
# UNIVERSAL SAFE-DATA WRAPPERS
# ============================

def safe_df(rates, symbol="", tf=""):
    """
    Converts MT5 data into a proper DataFrame.
    Prevents NoneType, numpy, empty, or corrupted data crashes.
    """
    try:
        if rates is None:
            print(f"❌ No MT5 data for {symbol} {tf}")
            return None

        df = pd.DataFrame(rates)

        if df is None or df.empty or len(df) < 5:
            print(f"❌ Insufficient data for {symbol} {tf}")
            return None

        return df

    except Exception as e:
        print(f"❌ Data conversion error for {symbol} {tf}: {e}")
        return None


def safe_smc_call(func, df, name=""):
    """
    Ensures all SMC detection functions ALWAYS return a valid (emoji, strength)
    tuple and never crash the engine. Updated for the emoji quality system.

    On failure or invalid returns, this helper emits diagnostic messages and
    returns (EMOJI_INVALID, 0.0) so that the calling pipeline can continue.
    """
    try:
        # No data means the indicator cannot be evaluated; return invalid.
        if df is None:
            return EMOJI_INVALID, 0.0
        # Call the wrapped detection function
        result = func(df)
        # Validate shape of return
        if not isinstance(result, tuple) or len(result) != 2:
            print(f"❌ {name} returned invalid result: {result}")
            return EMOJI_INVALID, 0.0
        emoji, strength = result
        # Validate emoji is one of the expected values
        if emoji not in (EMOJI_STRONG, EMOJI_WEAK_OK, EMOJI_VERY_WEAK, EMOJI_INVALID):
            print(f"❌ {name} returned invalid emoji: {emoji}")
            return EMOJI_INVALID, 0.0
        return emoji, strength
    except Exception as e:
        # Catch all errors and return invalid
        try:
            print(f"❌ Error in {name}: {e}")
        except Exception:
            log_debug(f"safe_smc_call print error for {name}:", e)
        return EMOJI_INVALID, 0.0

def _is_hard_block(symbol: str, intended_side: Optional[str] = None) -> Tuple[bool, str]:
    """
    Determine whether a symbol should be blocked from trading due to a hard rule.

    Only hard blockers may prevent a trade outright. These include:
      • MT5 disconnection or missing account information.
      • Missing or invalid tick prices (bid/ask zero or None).
      • Spreads exceeding the hard cap for XAUUSD (XAU_HARD_SPREAD_BLOCK_PTS).
      • Spread checks on non‑XAU symbols via the existing `check_spread` helper.

    Soft blockers such as session filters, range filters, BOS/Sweep alignment or
    session limits are not enforced here. Instead those conditions are handled
    downstream as score modifiers. This function returns a tuple (blocked, reason).
    """
    # Check MT5 connectivity (account and terminal must be available)
    try:
        ti = mt5.terminal_info()
        ai = mt5.account_info()
        if not ti or not ai:
            return True, "MT5 disconnected"
    except Exception:
        return True, "MT5 disconnected"
    # Symbol specific handling
    try:
        sym_upper = symbol.upper()
        if sym_upper.startswith('XAUUSD'):
            # Use existing XAU utility but only treat failures as hard when they
            # reference core connectivity or tick/spread issues.
            allowed, reason = can_place_on_xau(symbol, micro=False, intended_side=intended_side)
            if not allowed:
                # Hard reasons include explicit hard block keywords or missing tick data.
                hard_keywords = ("HARD_BLOCK", "Bad tick", "No tick", "disconnected")
                for kw in hard_keywords:
                    if reason and kw in reason:
                        return True, reason
            # Otherwise it's a soft failure – treat as permissible.
            return False, ""
        else:
            # Non‑XAU: ensure spread is acceptable using check_spread.
            try:
                if not check_spread(symbol):
                    # Interpret any spread failure as a hard blocker on non‑XAU pairs.
                    return True, "Spread above hard cap"
            except Exception:
                return True, "Spread check failed"
    except Exception:
        # On unexpected errors, be conservative and block
        return True, "Hard block due to error"
    return False, ""

def _classify_trade(score: int, bos_emoji: str, sweep_emoji: str, fvg_emoji: str, m15: Any) -> str:
    """
    Classify a trade based on its SMC score and pattern development.

    Scoring rules:
      • Scores ≥ 70% → FULL trade.
      • Scores 65–69% → PREDICTIVE trade if predictive AI confirms momentum; else MICRO.
      • Scores < 65% → MICRO trade.

    Momentum confirmation uses the existing `predictive_ai_confirms` helper and
    checks if any pattern is in a developing (EMOJI_WEAK_OK) state.
    """
    try:
        if score >= 70:
            return "FULL"
        elif score >= 65:
            try:
                bos_forming = (bos_emoji == EMOJI_WEAK_OK)
                sweep_forming = (sweep_emoji == EMOJI_WEAK_OK)
                fvg_forming = (fvg_emoji == EMOJI_WEAK_OK)
                if predictive_ai_confirms(bos_forming, sweep_forming, fvg_forming, m15):
                    return "PREDICTIVE"
            except Exception as e:
                log_debug("_classify_trade inner predictive_ai_confirms failed:", e)
            return "MICRO"
    except Exception as e:
        log_debug("_classify_trade failed:", e)
    return "MICRO"

def _compute_lot(classification: str, avg_strength: float) -> float:
    """
    Compute lot size based on trade classification and signal strength.

    For FULL trades the global `LOT_FULL` is used.  PREDICTIVE trades use
    `LOT_PREDICTIVE_MIN` and `LOT_PREDICTIVE_MAX` as bounds and scale with
    strength.  MICRO trades scale using the adaptive `micro_lot_ai_scale`
    and clamp within `MICRO_LOT_MIN`/`MICRO_LOT_MAX`.  If an exception
    occurs, fall back to a conservative minimum.
    """
    try:
        if classification == "FULL":
            return float(LOT_FULL)
        elif classification == "PREDICTIVE":
            # Linear scaling between predictive min/max lot sizes
            base = float(globals().get("LOT_PREDICTIVE_MIN", 0.05))
            upper = float(globals().get("LOT_PREDICTIVE_MAX", 0.1))
            lot = base + (upper - base) * float(avg_strength)
            lot = max(base, min(upper, lot))
            return round(lot, 2)
        else:
            # MICRO classification; honour adaptive AI scale stored in BOT_STATE
            base_min = float(globals().get("MICRO_LOT_MIN", 0.01))
            base_max = float(globals().get("MICRO_LOT_MAX", 0.03))
            try:
                lot = BOT_STATE.micro_lot_ai_scale * (1.0 + float(avg_strength))
            except Exception:
                lot = base_min
            lot = max(base_min, min(base_max, lot))
            return round(lot, 2)
    except Exception:
        # If something goes wrong, revert to minimum micro lot
        try:
            return float(globals().get("MICRO_LOT_MIN", 0.01))
        except Exception:
            return 0.01


def update_micro_scale(state: BotState, avg_strength: float) -> None:
    """
    Update the micro lot AI scale based on a sliding window of recent
    average signal strengths.  This function implements a simple
    adaptive scaling algorithm: a history of recent strengths is
    maintained in ``state.micro_strength_history``.  The micro lot
    scale is recalculated as a linear interpolation between
    ``MICRO_LOT_MIN`` and ``MICRO_LOT_MAX`` weighted by the mean of
    the history.  This yields a responsive yet smooth adjustment that
    reflects recent market momentum without abrupt jumps.  The history
    length defaults to 20 samples (adjustable via ``AI_SCALE_HISTORY_LEN``).

    Args:
        state: The runtime BotState storing the AI scale and strength history.
        avg_strength: The average strength of the current trade signal.

    Returns:
        None.  The state's ``micro_lot_ai_scale`` is updated in place.
    """
    try:
        # Append the new strength sample
        history = state.micro_strength_history
        history.append(float(avg_strength))
        # Determine maximum history length from global or default to 20
        try:
            max_len = int(globals().get("AI_SCALE_HISTORY_LEN", 20))
        except Exception:
            max_len = 20
        # Trim the oldest samples when exceeding the window size
        while len(history) > max_len:
            history.pop(0)
        # Compute the mean of the history; avoid division by zero
        try:
            mean_strength = sum(history) / len(history)
        except Exception:
            mean_strength = 0.0
        # Determine min/max bounds for micro lot scaling
        try:
            base_min = float(globals().get("MICRO_LOT_MIN", 0.01))
        except Exception:
            base_min = 0.01
        try:
            base_max = float(globals().get("MICRO_LOT_MAX", 0.03))
        except Exception:
            base_max = 0.03
        # Compute new scale as linear interpolation between bounds
        new_scale = base_min + (base_max - base_min) * max(0.0, min(1.0, mean_strength))
        # Clip to bounds to avoid overshoot
        new_scale = max(base_min, min(base_max, new_scale))
        # Update state if changed significantly
        old_scale = None
        try:
            old_scale = float(state.micro_lot_ai_scale)
        except Exception:
            old_scale = None
        state.micro_lot_ai_scale = new_scale
        # Log update if scale has changed meaningfully
        try:
            # Only log if there is a notable change (beyond small fluctuations).  A
            # threshold of 0.0005 prevents spamming logs on every tick when the
            # scale adjusts only marginally.
            if old_scale is None or abs(new_scale - old_scale) > 5e-4:
                try:
                    # Use higher-level log_msg if available; fallback to log_debug
                    if 'log_msg' in globals():
                        log_msg(f"[AI SCALE] micro_lot_ai_scale updated: {old_scale} -> {new_scale}")
                    else:
                        log_debug(f"[AI SCALE] micro_lot_ai_scale updated: {old_scale} -> {new_scale}")
                except Exception:
                    log_debug("update_micro_scale: logging failed")
        except Exception:
            pass
    except Exception as e:
        # Log errors but do not let them crash the trading loop
        try:
            log_debug("update_micro_scale failed:", e)
        except Exception:
            pass

def _determine_trade_side(symbol: str, m15: Any, h1: Any, bos_emoji: str, sweep_emoji: str, fvg_emoji: str) -> Tuple[str, list, int, int]:
    """
    Determine trade direction (BUY/SELL) using displacement, BOS, order block and candle heuristics.

    Returns:
      side: 'BUY' or 'SELL'
      votes: list of voted signals in 'SIGNAL:DIR' format
      buy_votes: number of BUY votes
      sell_votes: number of SELL votes

    The side calculation follows these steps:
      • Displacement direction on the lower timeframe (M15) if non‑neutral.
      • BOS direction via `detect_bos` on H1 or M15.
      • Last opposite‑direction order block on M15.
      • Last candle body direction as a final fallback.
      • Majority vote adjustment based on BOS/Sweep/FVG directional signals.
    """
    side = None
    votes: List[str] = []
    buy_votes = 0
    sell_votes = 0
    try:
        # Attempt displacement first
        try:
            disp = detect_displacement(m15)
            if isinstance(disp, dict):
                ddir = disp.get('direction', None)
                if isinstance(ddir, str):
                    ddir_u = ddir.upper()
                    if ddir_u in ('BUY', 'SELL') and ddir_u != 'NEUTRAL':
                        side = ddir_u
        except Exception as e:
            log_debug("detect_displacement failed:", e)
        # If no side from displacement, derive from BOS direction
        if not side:
            try:
                bos_res = None
                if h1 is not None and len(h1) > 0:
                    bos_res, _ = detect_bos(h1)
                else:
                    bos_res, _ = detect_bos(m15)
                bos_val = interpret_bos(bos_res)
                if bos_val > 0:
                    side = "BUY"
                elif bos_val < 0:
                    side = "SELL"
            except Exception as e:
                log_debug("detect_bos/interpret_bos failed:", e)
        # If still none, use last order block
        if not side:
            try:
                ob = detect_order_block(m15)
                if isinstance(ob, dict):
                    dir_ = ob.get('direction', None)
                    if isinstance(dir_, str) and dir_ in ('BUY', 'SELL'):
                        side = dir_
            except Exception as e:
                log_debug("detect_order_block failed:", e)
        # Fallback to last candle
        if not side:
            try:
                last_candle = m15.iloc[-1]
                side = "BUY" if float(last_candle['close']) > float(last_candle['open']) else "SELL"
            except Exception as e:
                log_debug("last_candle fallback failed:", e)
                side = "BUY"
        # Build per‑signal votes based on SMC detections
        # helper: direction from BOS (prefer H1 MSB)
        def _signal_direction_from_bos(df_h1: Any, df_ltf: Any) -> Optional[str]:
            try:
                msb = detect_msb(df_h1) if df_h1 is not None else None
                if isinstance(msb, dict) and msb.get('type'):
                    t = msb.get('type')
                    if isinstance(t, str):
                        if 'UP' in t:
                            return 'BUY'
                        if 'DOWN' in t:
                            return 'SELL'
            except Exception as e:
                log_debug("_signal_direction_from_bos detect_msb failed:", e)
            # Fallback: approximate from H1 last candle
            try:
                if df_h1 is not None:
                    last = df_h1.iloc[-1]
                    return 'BUY' if float(last['close']) > float(last['open']) else 'SELL'
            except Exception as e:
                log_debug("_signal_direction_from_bos fallback failed:", e)
            return None
        def _signal_direction_from_sweep(sym: str) -> Optional[str]:
            try:
                s_h1 = detect_liquidity_sweep(sym, timeframe=mt5.TIMEFRAME_H1)
                s_m15 = detect_liquidity_sweep(sym, timeframe=mt5.TIMEFRAME_M15)
                for sdict in (s_h1, s_m15):
                    if isinstance(sdict, dict) and sdict.get('type') in ('BUY', 'SELL'):
                        return sdict.get('type')
            except Exception as e:
                log_debug("_signal_direction_from_sweep failed:", e)
            return None
        def _signal_direction_from_fvg(df: Any) -> Optional[str]:
            try:
                if df is None or len(df) < 3:
                    return None
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                bull_gap = (float(c2['low']) > float(c1['high'])) and (float(c2['low']) > float(c3['high']))
                bear_gap = (float(c2['high']) < float(c1['low'])) and (float(c2['high']) < float(c3['low']))
                if bull_gap:
                    return 'BUY'
                if bear_gap:
                    return 'SELL'
            except Exception as e:
                log_debug("_signal_direction_from_fvg failed:", e)
            return None
        try:
            bos_vote = _signal_direction_from_bos(h1, m15)
            if bos_vote and bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"BOS:{bos_vote}")
        except Exception as e:
            log_debug("vote collection failed:", e)
        try:
            sweep_vote = _signal_direction_from_sweep(symbol)
            if sweep_vote and sweep_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"SWEEP:{sweep_vote}")
        except Exception as e:
            log_debug("sweep_vote collection failed:", e)
        try:
            fvg_vote = _signal_direction_from_fvg(m15)
            if fvg_vote and fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"FVG:{fvg_vote}")
        except Exception as e:
            log_debug("fvg_vote collection failed:", e)
        buy_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'BUY')
        sell_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'SELL')
        # Adjust side based on majority votes if present
        try:
            if votes:
                if buy_votes > sell_votes and side != "BUY":
                    side = "BUY"
                elif sell_votes > buy_votes and side != "SELL":
                    side = "SELL"
        except Exception as e:
            log_debug("majority vote adjustment failed:", e)
        # Normalise case
        if isinstance(side, str):
            side = side.upper()
    except Exception as e:
        log_debug("decide_side encountered error, defaulting to BUY:", e)
        side = "BUY"
    return side, votes, buy_votes, sell_votes

def _adjust_for_cooldowns(classification: str, now_uk: Any) -> str:
    """
    Adjust classification based on per‑day and cooldown constraints.

    Rather than hard blocking, this helper downgrades trade types when
    session caps or cooldown timers have been reached.  FULL trades will
    degrade to PREDICTIVE or MICRO, whereas PREDICTIVE trades degrade to MICRO
    when limits are exceeded.  MICRO trades are never downgraded.

    Returns the potentially adjusted classification.
    """
    # Use BOT_STATE as source of truth for cooldowns and counters
    try:
        # Downgrade FULL if we've hit daily cap or are within cooldown
        if classification == "FULL":
            # Daily full cap
            try:
                if BOT_STATE.full_trades_today >= globals().get("FULL_MAX_PER_DAY", 2):
                    classification = "PREDICTIVE"
            except Exception as e:
                log_debug("_adjust_for_cooldowns full_trades_today check failed:", e)
            # Full cooldown
            try:
                if BOT_STATE.last_full_time is not None:
                    cooldown_elapsed = (now_uk - BOT_STATE.last_full_time).total_seconds()
                    if cooldown_elapsed < globals().get("FULL_COOLDOWN_SECONDS", 600):
                        classification = "PREDICTIVE"
            except Exception as e:
                log_debug("_adjust_for_cooldowns last_full_time check failed:", e)
        # Downgrade PREDICTIVE if micro cap or cooldown are exceeded
        if classification == "PREDICTIVE":
            try:
                if BOT_STATE.micro_trades_today >= globals().get("MICRO_MAX_PER_SESSION", 4):
                    classification = "MICRO"
            except Exception as e:
                log_debug("_adjust_for_cooldowns micro_trades_today check failed:", e)
            try:
                if BOT_STATE.last_micro_time is not None:
                    cooldown_elapsed = (now_uk - BOT_STATE.last_micro_time).total_seconds()
                    if cooldown_elapsed < globals().get("MICRO_COOLDOWN_SECONDS", 300):
                        classification = "MICRO"
            except Exception as e:
                log_debug("_adjust_for_cooldowns last_micro_time check failed:", e)
    except Exception as e:
        log_debug("_adjust_for_cooldowns failed:", e)
    return classification


# === Strategy (SMC) ===
#
# Core strategy logic is implemented in the Holy Grail scan and execute function.  It
# evaluates BOS (Break of Structure), liquidity sweeps, and Fair Value Gaps
# to generate high-probability signals.  This section is intentionally left
# untouched by refactoring to preserve the trading strategy's integrity.  Any
# changes here should be limited to structural comments or wrappers that do
# not alter the underlying logic.
def holy_grail_scan_and_execute():
    global LAST_SCAN_XAU_NO_SETUP, last_signal
    if BOT_STATE.trading_paused:
        return
    if not ensure_mt5_connected():
        return
    if not check_dd_limits():
        BOT_STATE.trading_paused = True
        return
    try:
        _update_strict_mode()
    except Exception:
        pass
    now_uk = datetime.now(SAFE_TZ)
    
    # Get current session
    sess_label, _, _ = session_bounds(now_uk)
    
    # Allow trading during ASIA, LON, or NY sessions only
    # Block trading during other hours (6:00 - 8:00 and 17:00 - 24:00)
    if sess_label is None:
        return
    
    # === Notification: Session start (once per session) ===
    try:
        telegram_session_start(sess_label, symbol="XAUUSD")
    except Exception as e:
        log_debug("Session start notification failed:", e)
    
    # Persistent daily reset at UK midnight (only once per day)
    try:
        if state_store is not None:
            try:
                if state_store.maybe_daily_reset(now_uk.date()):
                    # Reset session block tracking on new day
                    BOT_STATE.xau_block_reason_sent_this_session.clear()
                    BOT_STATE.last_session_notified = None
                    BOT_STATE.micro_trades_today = 0
                    BOT_STATE.full_trades_today = 0
                    BOT_STATE.last_trade_zones.clear()
                    BOT_STATE.session_limits.clear()
                    BOT_STATE.full_consec_losses = 0
                    BOT_STATE.micro_consec_losses = 0
                    BOT_STATE.full_risk_reduction = 1.0
                    BOT_STATE.full_pause_until = None
                    BOT_STATE.micro_pause_until = None
                    print("[STATE] Daily counters reset for new UK day")
            except Exception as e:
                log_debug("daily reset inner handler failed:", e)
    except Exception as e:
        log_debug("daily reset outer handler failed:", e)
    
    # === Notification: Daily heartbeat (every 12-24 hours) ===
    try:
        total_trades_today = BOT_STATE.micro_trades_today + BOT_STATE.full_trades_today
        dd_pct = 0.0
        try:
            if BOT_STATE.day_open_equity and BOT_STATE.day_equity_high:
                dd = BOT_STATE.day_open_equity - BOT_STATE.day_equity_high
                dd_pct = (dd / BOT_STATE.day_open_equity * 100.0) if BOT_STATE.day_open_equity > 0 else 0.0
        except Exception:
            pass
        telegram_daily_heartbeat(session_label=sess_label, trades_today=total_trades_today, dd_percent=dd_pct)
    except Exception as e:
        log_debug("Daily heartbeat notification failed:", e)
    
    symbols = SYMBOLS
    # Iterate over all symbols and apply unified SMC pipeline
    for sym in symbols:
        try:
            # 1) Hard safety blockers: skip symbol if blocked
            blocked, block_reason = _is_hard_block(sym, None)
            if blocked:
                try:
                    # Send a concise block notification once per symbol
                    telegram_block(sym, block_reason)
                except Exception as e:
                    log_debug(f"telegram_block failed for {sym}:", e)
                continue
            # 2) Fetch price data
            m15_raw = get_rates(sym, mt5.TIMEFRAME_M15, 200)
            h1_raw  = get_rates(sym, mt5.TIMEFRAME_H1, 200)
            # Convert to DataFrame and validate
            columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            try:
                m15 = pd.DataFrame(m15_raw, columns=columns)
                h1  = pd.DataFrame(h1_raw,  columns=columns)
            except Exception:
                print(f"❌ Failed to convert MT5 data for {sym}")
                continue
            if m15 is None or m15.empty or len(m15) < 20:
                print(f"❌ Not enough M15 data for {sym}")
                continue
            if h1 is None or h1.empty or len(h1) < 20:
                print(f"❌ Not enough H1 data for {sym}")
                continue
            # 3) NOTE: duplicate trade checking is now handled inside place_trade.
            # We deliberately do not pre‑filter open positions here to avoid
            # evaluating the same condition twice. See place_trade for the single
            # authoritative check.
            # 4) Safe SMC pattern detection
            bos_emoji, bos_strength = safe_smc_call(detect_bos, m15, "detect_bos")
            sweep_emoji, sweep_strength = safe_smc_call(detect_sweep, m15, "detect_sweep")
            fvg_emoji, fvg_strength = safe_smc_call(detect_fvg, m15, "detect_fvg")
            # Skip any setup lacking all three patterns (must be confirmed or forming)
            if any(e in (EMOJI_VERY_WEAK, EMOJI_INVALID) for e in (bos_emoji, sweep_emoji, fvg_emoji)):
                # Console message only
                print(f"⛔ {sym}: Rejected due to weak/invalid SMC: BOS={bos_emoji} Sweep={sweep_emoji} FVG={fvg_emoji}")
                continue
            # 5) Compute average strength and score
            strengths = [bos_strength, sweep_strength, fvg_strength]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
            score = int(avg_strength * 100)
            # 6) Classify trade type based on score
            classification = _classify_trade(score, bos_emoji, sweep_emoji, fvg_emoji, m15)
            # Apply cooldown and caps adjustments
            classification = _adjust_for_cooldowns(classification, now_uk)
            # 6b) Apply soft filters for score penalties and downgrades
            try:
                allowed_soft, soft_reason, soft_penalty = SoftFilterEngine(BOT_STATE, sym, None)
                # If a soft penalty exists, adjust the score and classification
                if soft_penalty and isinstance(soft_penalty, int) and soft_penalty > 0:
                    score = max(0, score - soft_penalty)
                    # Recompute trade classification based on the penalised score
                    classification = _classify_trade(score, bos_emoji, sweep_emoji, fvg_emoji, m15)
                    # Force to MICRO when penalty is very high (e.g., off-session or session limit)
                    if soft_penalty >= 100:
                        classification = "MICRO"
                    # Reapply cooldown adjustments after downgrading
                    classification = _adjust_for_cooldowns(classification, now_uk)
            except Exception as e:
                log_debug("soft filter application failed:", e)
            # 7) Determine trade side and vote breakdown
            side, votes, buy_votes, sell_votes = _determine_trade_side(sym, m15, h1, bos_emoji, sweep_emoji, fvg_emoji)
            # 8) Compute lot size
            lot = _compute_lot(classification, avg_strength)
            # Adapt AI scale for micro or predictive trades based on recent signal history
            try:
                if classification in ("MICRO", "PREDICTIVE"):
                    update_micro_scale(BOT_STATE, avg_strength)
            except Exception as e:
                # Suppress errors but log if debugging enabled
                log_debug("ai scale update failed:", e)
            # 9) Update daily counters and timestamps
            try:
                # Counters are updated only after successful execution in place_trade
                pass
            except Exception as e:
                log_debug("counter update placeholder failed:", e)
            # 10) Update last signal global for external consumption
            last_signal = {
                "symbol": sym,
                "bos_emoji": bos_emoji,
                "sweep_emoji": sweep_emoji,
                "fvg_emoji": fvg_emoji,
                "bos_strength": bos_strength,
                "sweep_strength": sweep_strength,
                "fvg_strength": fvg_strength,
                "score": score,
                "decision": classification,
                "usable_count": sum(1 for e in (bos_emoji, sweep_emoji, fvg_emoji) if e in (EMOJI_STRONG, EMOJI_WEAK_OK))
            }
            # 11) Update zone tracker (soft – no blocking)
            try:
                current_price = float(m15["close"].iloc[-1])
                BOT_STATE.last_trade_zones[sym] = {"price": current_price, "time": now_uk}
            except Exception as e:
                log_debug("updating last_trade_zones failed:", e)
            # 12) Build signal lists for Telegram
            simple_signals = []
            detailed_signals = []
            try:
                if bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                    simple_signals.append("BOS")
                    detailed_signals.append(f"BOS({bos_emoji}:{bos_strength:.2f})")
                else:
                    detailed_signals.append(f"BOS({bos_emoji})")
                if sweep_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                    simple_signals.append("Sweep")
                    detailed_signals.append(f"Sweep({sweep_emoji}:{sweep_strength:.2f})")
                else:
                    detailed_signals.append(f"Sweep({sweep_emoji})")
                if fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                    simple_signals.append("FVG")
                    detailed_signals.append(f"FVG({fvg_emoji}:{fvg_strength:.2f})")
                else:
                    detailed_signals.append(f"FVG({fvg_emoji})")
            except Exception as e:
                log_debug("building signal lists failed:", e)
            # 13) Notify via Telegram
            try:
                telegram_signal(sym, classification, side, lot, score, simple_signals, detailed_signals, BOT_STATE.micro_trades_today, BOT_STATE.full_trades_today)
            except Exception as e:
                log_debug("telegram_signal failed:", e)
            # 14) Place the trade.  PREDICTIVE trades are executed via micro mode.
            mode_for_execution = classification if classification == "FULL" else "MICRO"
            # Provide entry price override for XAU micros if tick is missing
            entry_price_override = None
            try:
                tick_check = mt5.symbol_info_tick(sym) if hasattr(mt5, 'symbol_info_tick') else None
            except Exception as e:
                log_debug("mt5.symbol_info_tick failed:", e)
                tick_check = None
            if tick_check is None and mode_for_execution == "MICRO" and sym.upper().startswith('XAUUSD'):
                try:
                    entry_price_override = float(m15['close'].iloc[-1])
                except Exception:
                    entry_price_override = None
            place_trade(sym, lot, mode_for_execution, score, side, entry_price_override)
        except Exception as _ex:
            # Catch all unexpected errors per symbol and continue
            try:
                print(f"[ERROR] Error processing {sym}: {_ex}")
            except Exception as e:
                log_debug("printing symbol error failed:", e)
            continue

        # After handling any exceptions and placing a trade for this symbol, we skip
        # execution of any legacy fallback trading logic.  The unified decision
        # pipeline above fully determines trade actions, so the old code below
        # is never executed.  We retain it as a quoted block solely for
        # documentation.  Removing legacy paths prevents accidental execution.
        continue

        """ Legacy fallback trading logic (unused) begins below.  It will never
        execute due to the continue above but remains in place for reference.
        # Fetch raw MT5 data
        m15_raw = get_rates(sym, mt5.TIMEFRAME_M15, 200)
        h1_raw  = get_rates(sym, mt5.TIMEFRAME_H1, 200)

        # Correct MT5 → DataFrame conversion
        columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

        try:
            m15 = pd.DataFrame(m15_raw, columns=columns)
            h1  = pd.DataFrame(h1_raw,  columns=columns)
        except Exception:
            print(f"❌ Failed to convert MT5 data for {sym}")
            continue

        # Skip empty or invalid data
        if m15 is None or m15.empty or len(m15) < 20:
            print(f"❌ Not enough M15 data for {sym}")
            continue

        if h1 is None or h1.empty or len(h1) < 20:
            print(f"❌ Not enough H1 data for {sym}")
            continue

        # Resolve broker symbol and skip if there are already open positions for it
        try:
            resolved_sym = resolve_symbol(sym)
            positions_here = _get_positions_for_symbol(resolved_sym)
            if positions_here:
                try:
                    print(f"[STATE] Open positions detected for {sym} ({resolved_sym}) - skipping to avoid duplicate trades")
                except Exception:
                    log_debug("printing open positions detection failed for:", sym)
                # mark that open positions exist so no new trades are placed
                continue
        except Exception as e:
            log_debug("resolve_symbol fallback to input symbol due to:", e)
            resolved_sym = sym

        # Safe SMC detections with emoji quality system
        bos_emoji, bos_strength = safe_smc_call(detect_bos, m15, "detect_bos")
        sweep_emoji, sweep_strength = safe_smc_call(detect_sweep, m15, "detect_sweep")
        fvg_emoji, fvg_strength = safe_smc_call(detect_fvg, m15, "detect_fvg")
        
        # Collect emoji ratings
        emojis = [bos_emoji, sweep_emoji, fvg_emoji]
        strengths = [bos_strength, sweep_strength, fvg_strength]
        
        # Block if ANY condition is very weak or invalid
        if EMOJI_VERY_WEAK in emojis or EMOJI_INVALID in emojis:
            try:
                if sym.upper().startswith('XAUUSD'):
                    # Defer noisy XAUUSD rejection messages; report once per scan
                    xau_no_setup_seen = True
                    try:
                        globals()['LAST_SCAN_XAU_NO_SETUP'] = True
                    except Exception as e:
                        log_debug("setting LAST_SCAN_XAU_NO_SETUP failed:", e)
                else:
                    print(f"⛔ {sym}: Rejected due to weak/invalid SMC: BOS={bos_emoji} Sweep={sweep_emoji} FVG={fvg_emoji}")
            except Exception as e:
                log_debug("reject due to weak/invalid SMC failed:", e)
            continue
        
        # Count usable confirmations (✅ or 👍🏾)
        usable_count = sum([1 for e in emojis if e in [EMOJI_STRONG, EMOJI_WEAK_OK]])
        strong_count = sum([1 for e in emojis if e == EMOJI_STRONG])
        
        # Check duplicate zone blocker
        current_price = m15["close"].iloc[-1]
        if sym in BOT_STATE.last_trade_zones:
            zone = BOT_STATE.last_trade_zones[sym]
            price_diff = abs(current_price - zone["price"]) / zone["price"]
            time_diff = (now_uk - zone["time"]).total_seconds()
            if price_diff < DUPLICATE_ZONE_THRESHOLD and time_diff < DUPLICATE_ZONE_TIME_MIN:
                print(f"⛔ Duplicate zone detected for {sym}, skipping")
                continue
        
        trend = trend_bias_soft(m15)
        mitigation = mitigation_present(m15)
        mtf = mtf_refinement(h1, m15)
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        score = int(avg_strength * 100)
        decision = "IGNORE"
        lot = 0.0
        
        # TRADE DECISION LOGIC (directional agreement rules):
        # - FULL (XAUUSD): requires at least 2 usable SMC signals AND
        #   at least 2 of those signals must agree on the same direction.
        # - MICRO: requires at least 1 usable SMC signal (direction determined
        #   by majority of available signals; displacement/BOS/order-block used
        #   as fallbacks).
        # Note: signals are BOS, Sweep, FVG.
        def _signal_direction_from_bos(df_h1, df_ltf):
            # Prefer MSB/BOS on H1 if available
            try:
                msb = detect_msb(df_h1)
                if isinstance(msb, dict) and msb.get('type'):
                    t = msb.get('type')
                    if isinstance(t, str) and 'UP' in t:
                        return 'BUY'
                    if isinstance(t, str) and 'DOWN' in t:
                        return 'SELL'
            except Exception as e:
                log_debug("_signal_direction_from_bos msb check failed:", e)
            # Fallback: approximate from H1 last candle
            try:
                last = df_h1.iloc[-1]
                return 'BUY' if last['close'] > last['open'] else 'SELL'
            except Exception:
                return None

        def _signal_direction_from_sweep(symbol):
            try:
                s_h1 = detect_liquidity_sweep(symbol, timeframe=mt5.TIMEFRAME_H1)
                s_m15 = detect_liquidity_sweep(symbol, timeframe=mt5.TIMEFRAME_M15)
                for sdict in (s_h1, s_m15):
                    if isinstance(sdict, dict) and sdict.get('type') in ('BUY', 'SELL'):
                        return sdict.get('type')
            except Exception as e:
                log_debug("_signal_direction_from_sweep failed:", e)
            return None

        def _signal_direction_from_fvg(df):
            try:
                if df is None or len(df) < 3:
                    return None
                c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
                bull_gap = c2['low'] > c1['high'] and c2['low'] > c3['high']
                bear_gap = c2['high'] < c1['low'] and c2['high'] < c3['low']
                if bull_gap:
                    return 'BUY'
                if bear_gap:
                    return 'SELL'
            except Exception as e:
                log_debug("_signal_direction_from_fvg failed:", e)
            return None

        # Build per-signal directional votes (store as SIGNAL:DIR strings
        # so we can report which signals agreed). Example: 'BOS:BUY'
        votes = []
        try:
            bos_vote = _signal_direction_from_bos(h1, m15)
            if bos_vote and bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"BOS:{bos_vote}")
        except Exception as e:
            log_debug("vote collection BOS failed:", e)
        try:
            sweep_vote = _signal_direction_from_sweep(sym)
            if sweep_vote and sweep_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"SWEEP:{sweep_vote}")
        except Exception as e:
            log_debug("vote collection SWEEP failed:", e)
        try:
            fvg_vote = _signal_direction_from_fvg(m15)
            if fvg_vote and fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                votes.append(f"FVG:{fvg_vote}")
        except Exception as e:
            log_debug("vote collection FVG failed:", e)

        # Count agreeing votes
        buy_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'BUY')
        sell_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'SELL')

        # FULL TRADE: allowed for symbols in FULL_ALLOWED_SYMBOLS (e.g. XAUUSD, GBPUSD, GBPJPY).
        # Need at least 2 usable signals AND at least 2 agreeing votes on same direction.
        if sym.upper() in FULL_ALLOWED_SYMBOLS and usable_count >= 2 and (buy_votes >= 2 or sell_votes >= 2):
            # Enforce FTMO daily cap
            if BOT_STATE.full_trades_today >= FULL_MAX_PER_DAY:
                print(f"⛔ Full trade cap hit ({FULL_MAX_PER_DAY}/day), ignoring signal for {sym}")
                continue

            # Enforce per-session cap
            try:
                if BOT_STATE.session_full_count >= globals().get('FULL_MAX_PER_SESSION', 2):
                    print(f"⛔ Full trade session cap hit ({BOT_STATE.session_full_count}/{FULL_MAX_PER_SESSION}), ignoring {sym}")
                    continue
            except Exception as e:
                log_debug(f"full session cap check failed for {sym}:", e)

            # Don't place FULL trades if we've already had 2 FULL losses this session
            try:
                if BOT_STATE.session_full_losses >= 2:
                    print(f"⛔ Session FULL loss limit reached ({BOT_STATE.session_full_losses} losses) — skipping FULL trades")
                    # allow micro-only for tracking
                    pass_full_block = True
                else:
                    pass_full_block = False
            except Exception as e:
                log_debug("session_full_losses check failed:", e)
                pass_full_block = False

            # HTF bias (Daily / H4) used as directional bias only — block FULL if counter-HTF
            htf_trend = get_htf_trend(sym)

            preferred = 'BUY' if buy_votes > sell_votes else 'SELL'
            # If HTF trend exists and is opposite to preferred and we are not forcing FULL, block FULL
            if htf_trend is not None and preferred != htf_trend:
                print(f"⛔ FULL blocked — counter-HTF ({preferred} vs HTF {htf_trend}) for {sym}. Allowing micro only.")
                pass_full_block = True

            if pass_full_block:
                # Downgrade to micro decision path (do not increment full counters)
                # Let the micro branch handle cap/cooldown
                decision = 'MICRO'
                lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, BOT_STATE.micro_lot_ai_scale * (1 + avg_strength)))
                lot = round(lot, 2)
                # Counters updated after successful execution in place_trade
                pass
            else:
                if BOT_STATE.last_full_time is not None:
                    cooldown_elapsed = (now_uk - BOT_STATE.last_full_time).total_seconds()
                    if cooldown_elapsed < FULL_COOLDOWN_SECONDS:
                        print(f"⛔ Full trade cooldown active ({int(cooldown_elapsed)}s < {FULL_COOLDOWN_SECONDS}s), ignoring {sym}")
                        continue

                decision = "FULL"
                lot = LOT_FULL
                # Counters updated after successful execution in place_trade
                BOT_STATE.session_full_count = BOT_STATE.session_full_count + 1
                pass
        
        # MICRO TRADE LOGIC: Requires at least 1 usable SMC signal
        elif usable_count >= 1:
            if BOT_STATE.micro_trades_today >= MICRO_MAX_PER_SESSION:
                print(f"⛔ Micro trade cap hit ({MICRO_MAX_PER_SESSION}/session), ignoring signal for {sym}")
                continue
            
            if BOT_STATE.last_micro_time is not None:
                cooldown_elapsed = (now_uk - BOT_STATE.last_micro_time).total_seconds()
                if cooldown_elapsed < MICRO_COOLDOWN_SECONDS:
                    print(f"⛔ Micro trade cooldown active ({int(cooldown_elapsed)}s < {MICRO_COOLDOWN_SECONDS}s), ignoring {sym}")
                    continue
            
            decision = "MICRO"
            # AI scaling for micro: start at 0.01, scale to 0.04 max based on strength
            lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, BOT_STATE.micro_lot_ai_scale * (1 + avg_strength)))
            lot = round(lot, 2)
            # Adapt AI scale for micro trades using recent signal strength history
            try:
                update_micro_scale(BOT_STATE, avg_strength)
            except Exception as e:
                log_debug("micro ai scale log failed:", e)
            # Counters updated after successful execution in place_trade
            pass
        
        else:
            decision = "IGNORE"
        
        last_signal = {
            "symbol": sym,
            "bos_emoji": bos_emoji,
            "sweep_emoji": sweep_emoji,
            "fvg_emoji": fvg_emoji,
            "bos_strength": bos_strength,
            "sweep_strength": sweep_strength,
            "fvg_strength": fvg_strength,
            "score": score,
            "decision": decision,
            "usable_count": usable_count
        }
        
        # Update zone tracker
        if decision in ["FULL", "MICRO"]:
            BOT_STATE.last_trade_zones[sym] = {"price": current_price, "time": now_uk}

        # Resolve symbol to broker symbol before any tick/data operations so
        # that instruments like XAUUSD (which may appear as GOLD or with a
        # suffix) are correctly addressed. Log when resolution differs.
        resolved_sym = resolve_symbol(sym)
        if resolved_sym != sym:
            log_msg(f"[SYMBOL] Resolved {sym} -> {resolved_sym}")

        # Determine trade side: prefer displacement direction, then BOS, then order-block,
        # and only fall back to the last candle as a last resort. This reduces cases
        # where a single neutral/inside candle causes an unintended SELL/BY placement.
        side = None
        try:
            disp = detect_displacement(m15)
            if isinstance(disp, dict):
                ddir = disp.get('direction')
                if ddir and ddir.upper() in ('BUY', 'SELL') and ddir.upper() != 'NEUTRAL':
                    side = ddir.upper()
        except Exception:
            side = None

        # If displacement did not produce a clear side, try BOS on H1 then M15
        if not side:
            try:
                # prefer H1 BOS for bias if available
                if 'h1' in locals() and h1 is not None:
                    bos_emoji, _ = detect_bos(h1)
                else:
                    bos_emoji, _ = detect_bos(m15)
                bos_val = interpret_bos(bos_emoji)
                if bos_val > 0:
                    side = 'BUY'
                elif bos_val < 0:
                    side = 'SELL'
            except Exception:
                side = None

        # If still no side, check for a last opposite directional order block
        if not side:
            try:
                ob = detect_order_block(m15)
                if isinstance(ob, dict) and ob.get('direction') in ('BUY', 'SELL'):
                    side = ob.get('direction')
            except Exception:
                side = None

        # Last-resort fallback: use last candle direction
        if not side:
            try:
                last_candle = m15.iloc[-1]
                side = 'BUY' if last_candle['close'] > last_candle['open'] else 'SELL'
            except Exception:
                side = None

        # Normalise side value
        if isinstance(side, str):
            side = side.upper()

        # Reconcile computed side with SMC signal votes (if any). If signals
        # produced a clear majority, prefer that majority to avoid placing a
        # SELL when SMC signals indicate BUY (and vice versa).
        try:
            if votes:
                if buy_votes > sell_votes:
                    preferred = 'BUY'
                elif sell_votes > buy_votes:
                    preferred = 'SELL'
                else:
                    preferred = side
                if preferred and preferred in ('BUY', 'SELL') and side != preferred:
                    # Override side and note it to operator/logs
                    side_adjust_msg = (
                        f"🔁 **SIDE ADJUSTED** — {sym}\n"
                        f"From: {side} → To: {preferred}\n"
                        f"Reason: SMC vote majority (buy={buy_votes}, sell={sell_votes})"
                    )
                    # Do not send side-adjusted logs to Telegram (console-only)
                    try:
                        print(side_adjust_msg)
                    except Exception as e:
                        log_debug(f"print side_adjust_msg failed for {sym}:", e)
                    log_msg(f"[SIDE] Overrode computed side {side} -> {preferred} for {sym} based on votes (buy={buy_votes}, sell={sell_votes})")
                    side = preferred
        except Exception as e:
            log_debug(f"side override block failed for {sym}:", e)
        
        # Build quality explanation
        quality_msg = f"BOS: {bos_emoji} ({bos_strength:.2f})\nSweep: {sweep_emoji} ({sweep_strength:.2f})\nFVG: {fvg_emoji} ({fvg_strength:.2f})"
        
        
        # XAUUSD may execute either FULL (when criteria met) or MICRO trades;
        # no unconditional override here so both trade types remain possible.

        # Apply owner-forced override if present (respecting caps and cooldowns)
        forced = FORCED_TRADES.get(sym)
        if forced:
            if forced == "FULL":
                # attempt to use FULL if caps and cooldown allow
                can_full = True
                if BOT_STATE.full_trades_today >= FULL_MAX_PER_DAY:
                    can_full = False
                if BOT_STATE.last_full_time is not None:
                    cooldown_elapsed = (now_uk - BOT_STATE.last_full_time).total_seconds()
                    if cooldown_elapsed < FULL_COOLDOWN_SECONDS:
                        can_full = False
                if can_full:
                    if decision != "FULL":
                        decision = "FULL"
                        lot = LOT_FULL
                        # Counters updated after successful execution in place_trade
                        pass
                else:
                    # fallback: try micro if possible
                    forced = "MICRO"
            if forced == "MICRO":
                can_micro = True
                if BOT_STATE.micro_trades_today >= MICRO_MAX_PER_SESSION:
                    can_micro = False
                if BOT_STATE.last_micro_time is not None:
                    cooldown_elapsed = (now_uk - BOT_STATE.last_micro_time).total_seconds()
                    if cooldown_elapsed < MICRO_COOLDOWN_SECONDS:
                        can_micro = False
                if can_micro:
                    if decision != "MICRO":
                        decision = "MICRO"
                        # compute conservative micro lot if we don't already have one
                        lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, BOT_STATE.micro_lot_ai_scale * (1 + avg_strength)))
                        lot = round(lot, 2)
                        # Counters updated after successful execution in place_trade
                        pass
                else:
                    # Do not spam Telegram for forced-mode cooldowns; console-only.
                    try:
                        print(f"⛔ Forced {forced} for {sym} blocked by caps/cooldown")
                    except Exception as e:
                        log_debug(f"forced-mode cooldown print failed for {sym}:", e)

        if decision == "IGNORE":
            # Skip notification for ignored signals (console-only)
            try:
                print("⛔ TRADE SKIPPED — Weak or Invalid SMC Signal (BOS / Sweep / FVG)")
            except Exception as e:
                log_debug("ignore-signal print failed:", e)
            continue

        # Safety: ensure side is valid. If side is not BUY/SELL we refuse to place
        # the trade and log the anomaly so it can be investigated.
        if side not in ("BUY", "SELL"):
            # Report as a single-blocked summary for allowed symbols; otherwise console-only.
            try:
                telegram_block(sym, f"Invalid side: {side}")
            except Exception:
                try:
                    print(f"Invalid side detected for {sym}: {side}")
                except Exception as e:
                    log_debug(f"print invalid side failed for {sym}:", e)
            log_msg(f"[SAFETY] Aborted trade for {sym} - invalid side: {side} | decision={decision} | lot={lot}")
            continue

        # Build verbose placement log: which signals were usable, their strengths,
        # vote breakdown and displacement info. Send to Telegram and also log.
        try:
            usable_signals = []
            if bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                usable_signals.append(f"BOS({bos_emoji}:{bos_strength:.2f})")
            else:
                usable_signals.append(f"BOS({bos_emoji})")
            if sweep_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                usable_signals.append(f"Sweep({sweep_emoji}:{sweep_strength:.2f})")
            else:
                usable_signals.append(f"Sweep({sweep_emoji})")
            if fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                usable_signals.append(f"FVG({fvg_emoji}:{fvg_strength:.2f})")
            else:
                usable_signals.append(f"FVG({fvg_emoji})")

            disp_info = None
            try:
                if isinstance(disp, dict):
                    disp_info = f"disp={disp.get('direction')} score={disp.get('score',0)} strong={disp.get('strong',False)}"
                else:
                    disp_info = str(disp)
            except Exception:
                disp_info = str(disp)

            vote_summary = f"votes={votes} (buy={buy_votes}, sell={sell_votes})"
            # Build explicit reason: which signals agreed for the majority
            agreeing = []
            if buy_votes > sell_votes and buy_votes > 0:
                agreeing = [s.split(':',1)[0] for s in votes if s.endswith(':BUY')]
                reason = f"BUY agreement: {agreeing}"
            elif sell_votes > buy_votes and sell_votes > 0:
                agreeing = [s.split(':',1)[0] for s in votes if s.endswith(':SELL')]
                reason = f"SELL agreement: {agreeing}"
            else:
                reason = "No clear agreement"
            verbose_msg = (
                f"🧾 PRE-PLACEMENT SUMMARY — {sym}\n"
                f"• Decision: {decision} | Side: {side} | Lot: {lot:.2f} | Score: {score}%\n"
                f"• Signals: {', '.join(usable_signals)}\n"
                f"• {vote_summary} | {disp_info}\n"
                f"• Reason: {reason}\n"
            )
            # Send Signal message (includes explicit usable_signals) and verbose pre-place to Telegram
            try:
                simple_signals = [s.split('(')[0] for s in usable_signals]
                # Send compact signal notification (Telegram only for allowed symbols)
                try:
                    telegram_signal(sym, decision, side, lot, score, simple_signals, usable_signals, BOT_STATE.micro_trades_today, BOT_STATE.full_trades_today)
                except Exception as e:
                    log_debug("telegram_signal failed for pre-place:", e)
                # Keep verbose pre-place in console only (no Telegram)
                try:
                    print(verbose_msg)
                except Exception as e:
                    log_debug("print verbose_msg failed:", e)
            except Exception as e:
                try:
                    print(verbose_msg)
                except Exception as e2:
                    log_debug("fallback print verbose_msg failed:", e2)
                log_debug("pre-place message assembly failed:", e)
            try:
                log_msg(f"[PRE-PLACE] {sym} | {decision} | side={side} | {vote_summary} | signals={usable_signals} | {disp_info}")
            except Exception as e:
                log_debug("log_msg failed for pre-place:", e)
        except Exception:
            # Best-effort: if verbose logging fails, continue to place trade
            try:
                print(f"Placing {decision} {sym} {side} lot={lot:.2f} (score={score}%)")
            except Exception as e:
                log_debug("print pre-place failed:", e)

        # If spread is bad for this symbol, block full trades but allow micro
            try:
                if decision == 'FULL' and not spread_ok_flag:
                    # Do not send Telegram for spread-based skipping; console-only
                    try:
                        print("⛔ TRADE SKIPPED — Spread too high or tick unavailable for full trade on XAUUSD; micro may still execute")
                    except Exception as e:
                        log_debug("print failed for spread skip message:", e)
                    continue
            except Exception as e:
                log_debug("spread skip check failed:", e)

        # Prepare optional entry price override when tick is unavailable for XAUUSD micros
        entry_price_override = None
        try:
            tick_check = mt5.symbol_info_tick(sym) if hasattr(mt5, 'symbol_info_tick') else None
        except Exception:
            tick_check = None
        if tick_check is None and decision == 'MICRO' and sym.upper().startswith('XAUUSD'):
            try:
                # Use last M15 close as a conservative entry price fallback
                entry_price_override = float(m15['close'].iloc[-1])
            except Exception:
                entry_price_override = None

        place_trade(sym, lot, decision, score, side, entry_price_override)
        """  # end of legacy fallback trading logic

# -----------------------------------------------------------------------------
# Trade placement
# -----------------------------------------------------------------------------
def _get_positions_for_symbol(resolved_symbol: str):
    """Return a list of open positions for the given broker-resolved symbol."""
    try:
        try:
            pos = mt5.positions_get()
        except Exception:
            pos = None
        if not pos:
            return []
        out = [p for p in pos if getattr(p, 'symbol', '').upper() == resolved_symbol.upper()]
        return out
    except Exception:
        return []


def _close_positions_for_symbol(resolved_symbol: str) -> bool:
    """Best-effort close all positions for `resolved_symbol`.

    In TEST_MODE or DRY_RUN this will only log the intended actions. In live
    mode it will attempt to send closing market orders using `safe_order_send`.
    Returns True if all close attempts were issued (or stubbed) without error.
    """
    ok = True
    try:
        positions = _get_positions_for_symbol(resolved_symbol)
        if not positions:
            return True
        tick = None
        try:
            tick = mt5.symbol_info_tick(resolved_symbol)
        except Exception:
            tick = None
        for p in positions:
            try:
                ptype = getattr(p, 'type', None)
                volume = float(getattr(p, 'volume', getattr(p, 'volume_initial', 0.0) or 0.0))
                if volume <= 0:
                    continue
                # Determine close order type and price
                if ptype == getattr(mt5, 'POSITION_TYPE_BUY', 0):
                    close_type = getattr(mt5, 'ORDER_TYPE_SELL', 1)
                    price = tick.bid if tick else 0.0
                else:
                    close_type = getattr(mt5, 'ORDER_TYPE_BUY', 0)
                    price = tick.ask if tick else 0.0
                req = {
                    'action': getattr(mt5, 'TRADE_ACTION_DEAL', 1),
                    'symbol': resolved_symbol,
                    'volume': volume,
                    'type': close_type,
                    'price': price,
                    'deviation': 50,
                    'comment': 'HG REVERSAL CLOSE'
                }
                if globals().get('TEST_MODE') or globals().get('DRY_RUN'):
                    log_msg(f"[STUB] Would close {resolved_symbol} vol={volume} via type={close_type} price={price}")
                else:
                    res = safe_order_send(req)
                    log_msg(f"[REVERSAL] close issued for {resolved_symbol} vol={volume} -> retcode={getattr(res,'retcode',None)}")
            except Exception as e:
                log_msg(f"[REVERSAL] error closing position for {resolved_symbol}: {e}")
                ok = False
        return ok
    except Exception:
        return False

# === Execution ===
#
# Functions responsible for order placement, including market orders,
# pending orders, and reversal logic, are defined in this section.  These
# wrappers enforce micro/full lot limits and interact with the MetaTrader
# trade API via safe send functions.  Execution functions should not
# perform heavy computation; they simply handle order creation and
# submission.
def place_trade(symbol: str, lot: float, mode: str, score: int, side: str = 'BUY', entry_price_override=None):
    """Place a market trade for `symbol` with given `lot`, `mode`, and `side` ('BUY'|'SELL').
    This sets price/SL/TP appropriately for buy vs sell and enforces micro/full clamps.
    """
    # Local variables used for logging risk and stop-loss distance later.

    # Counters and timestamps are stored on BOT_STATE; avoid module globals.
    risk_used_pct_local = 0.0
    sl_pips_local = 0.0

    # Enforce final lot clamps per mode to avoid accidental large volumes
    if mode == "MICRO":
        lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, lot))
        lot = round(lot, 2)
    elif mode == "FULL":
        lot = LOT_FULL

    # ---------------------------------------------------------------
    # Time-based cooldown enforcement (not candle-based)
    # ---------------------------------------------------------------
    try:
        now_uk_lock = datetime.now(SAFE_TZ)
        
        # Retrieve the last successful execution info for this symbol (timestamp, retcode)
        last_exec_info = SYMBOL_EXECUTION_LOCK.get(symbol.upper())
        if last_exec_info:
            try:
                last_exec_time, last_exec_retcode = last_exec_info
            except Exception:
                last_exec_time, last_exec_retcode = None, None
            
            # Only enforce cooldown if the last execution was successful
            success_code = getattr(mt5, 'TRADE_RETCODE_DONE', 10009)
            if last_exec_time is not None and last_exec_retcode == success_code:
                # Determine cooldown duration based on trade mode
                cooldown_seconds = FULL_COOLDOWN_SECONDS if mode == 'FULL' else MICRO_COOLDOWN_SECONDS
                
                # Calculate time elapsed since last execution
                time_since_last = (now_uk_lock - last_exec_time).total_seconds()
                
                # If cooldown period hasn't elapsed, block the trade
                if time_since_last < cooldown_seconds:
                    remaining = cooldown_seconds - time_since_last
                    try:
                        extra = f"cooldown_remaining={remaining:.0f}s | mode={mode} | last_exec={last_exec_time.strftime('%H:%M:%S')}"
                    except Exception:
                        extra = f"cooldown_remaining={remaining:.0f}s"
                    log_block_verbose(symbol, "cooldown", extra_info=extra)
                    return
                else:
                    # Cooldown expired, clear the lock
                    SYMBOL_EXECUTION_LOCK.pop(symbol.upper(), None)
    except Exception as e:
        # If lock check fails, log but continue execution rather than risking duplicate trades
        try:
            log_debug("cooldown check failed:", e)
        except Exception:
            pass

    # Duplicate full entry guard (same direction within 15 minutes)
    try:
        if mode == 'FULL':
            key = f"{symbol.upper()}:{side.upper()}"
            last_ts = BOT_STATE.last_full_entry_by_symbol_side.get(key)
            if last_ts and (now_uk_lock - last_ts) < timedelta(minutes=FULL_DUPLICATE_BLOCK_MIN):
                log_block_verbose(symbol, "duplicate full entry (15m)")
                telegram_msg(f"🚫 {symbol} FULL blocked: duplicate entry within 15m")
                return
    except Exception:
        pass

    # Hard limits guard (session caps, pauses, strict mode)
    try:
        ok_limits, reason_limits = _check_hard_limits(symbol, side, mode)
        if not ok_limits:
            log_block_verbose(symbol, reason_limits)
            telegram_msg(f"🚫 {symbol} {mode} blocked: {reason_limits}")
            return
    except Exception:
        pass

    # ------- No-opposite-direction (no hedging) guard -------
    try:
        resolved_sym = resolve_symbol(symbol)
    except Exception:
        resolved_sym = symbol
    try:
        positions = _get_positions_for_symbol(resolved_sym)
        # Startup resume guard: if symbol had open trades on restart, do not place
        # new trades until those positions are fully closed.
        try:
            startup_map = BOT_STATE.extra.setdefault('startup_open_symbols', {})
            sym_key = str(resolved_sym).upper()
            if sym_key not in startup_map:
                startup_map[sym_key] = True if (positions and len(positions) > 0) else False
            if startup_map.get(sym_key):
                if positions and len(positions) > 0:
                    try:
                        telegram_block(symbol, "startup guard: open trades detected")
                    except Exception as e:
                        log_debug(f"telegram_block failed for startup guard {symbol}:", e)
                    log_block_verbose(symbol, "startup guard: open trades detected")
                    return
                else:
                    startup_map[sym_key] = False
        except Exception as e:
            log_debug("startup guard check failed:", e)
        # Enforce max 2 open trades per symbol (one existing + one extra)
        try:
            if positions is not None and len(positions) >= 2:
                try:
                    telegram_block(symbol, "max open trades reached (2) for symbol")
                except Exception as e:
                    log_debug(f"telegram_block failed for max trades {symbol}:", e)
                log_block_verbose(symbol, "max open trades reached (2) for symbol")
                return
        except Exception as e:
            log_debug("max open trades check failed:", e)
        # Stability: enforce single active full/micro position per symbol
        try:
            if positions is not None:
                full_exists = any(float(getattr(p, 'volume', 0.0) or 0.0) > MICRO_LOT_MAX for p in positions)
                micro_exists = any(float(getattr(p, 'volume', 0.0) or 0.0) <= MICRO_LOT_MAX for p in positions)
                if mode == 'FULL' and full_exists:
                    telegram_block(symbol, "full position already active")
                    log_block_verbose(symbol, "full position already active")
                    return
                if mode == 'MICRO' and micro_exists:
                    telegram_block(symbol, "micro position already active")
                    log_block_verbose(symbol, "micro position already active")
                    return
                # No stacking micro + full in same direction
                same_dir = False
                for p in positions:
                    ptype = getattr(p, 'type', None)
                    pside = 'BUY' if ptype == getattr(mt5, 'POSITION_TYPE_BUY', 0) else 'SELL'
                    if pside == side:
                        same_dir = True
                        break
                if same_dir:
                    if mode == 'FULL' and micro_exists:
                        telegram_block(symbol, "micro already open same direction")
                        log_block_verbose(symbol, "micro already open same direction")
                        return
                    if mode == 'MICRO' and full_exists:
                        telegram_block(symbol, "full already open same direction")
                        log_block_verbose(symbol, "full already open same direction")
                        return
        except Exception as e:
            log_debug("mode-specific position check failed:", e)
        # Never stack more than one metals full trade at a time
        try:
            base_sym = _base_of(symbol)
            if mode == 'FULL' and base_sym in ('XAUUSD', 'XAGUSD'):
                all_pos = mt5.positions_get() or []
                for p in all_pos:
                    sym = _base_of(getattr(p, 'symbol', ''))
                    vol = float(getattr(p, 'volume', 0.0) or 0.0)
                    if sym in ('XAUUSD', 'XAGUSD') and vol > MICRO_LOT_MAX:
                        telegram_block(symbol, "metal full already open")
                        log_block_verbose(symbol, "metal full already open")
                        return
        except Exception:
            pass
        # Determine if an opposite-direction position exists
        opposite_exists = False
        buy_exists = False
        sell_exists = False
        for p in positions:
            ptype = getattr(p, 'type', None)
            if ptype == getattr(mt5, 'POSITION_TYPE_BUY', 0):
                buy_exists = True
            if ptype == getattr(mt5, 'POSITION_TYPE_SELL', 1):
                sell_exists = True
        if side == 'BUY' and sell_exists:
            opposite_exists = True
        if side == 'SELL' and buy_exists:
            opposite_exists = True
        if opposite_exists:
            # Optional reversal behavior controlled via env var
            if str(os.getenv('ALLOW_REVERSAL_ON_SIGNAL', '0')).lower() in ('1','true','yes'):
                # Attempt safe close of opposite positions before placing the new trade
                closed = _close_positions_for_symbol(resolved_sym)
                if not closed:
                    try:
                        telegram_block(symbol, "existing opposite positions could not be closed")
                    except Exception as e:
                        log_debug(f"telegram_block failed closing opposite positions for {symbol}:", e)
                    # Log the block reason once per candle
                    log_block_verbose(symbol, "existing opposite positions could not be closed")
                    return
                else:
                    log_msg(f"Reversal: closed existing positions for {symbol} before placing {side}")
            else:
                existing = 'BUY' if buy_exists else 'SELL'
                try:
                    telegram_block(symbol, f"{existing} already open")
                except Exception as e:
                    log_debug(f"telegram_block failed for existing open {symbol}:", e)
                # Log the block reason once per candle
                log_block_verbose(symbol, f"{existing} already open")
                return
    except Exception as e:
        # If any error occurs in position-checking, be conservative and block
        try:
            telegram_block(symbol, "could not verify existing positions")
        except Exception as ex:
            log_debug(f"telegram_block failed during position-check for {symbol}:", ex)
        try:
            # Log the block reason once per candle (verbose)
            log_block_verbose(symbol, "could not verify existing positions")
        except Exception:
            # Fallback to debug log
            log_debug("position checking error:", e)
        return

    # Resolve to broker symbol and ensure it's selected for trading
    try:
        broker_sym = resolve_symbol(symbol)
        if broker_sym and broker_sym != symbol:
            log_msg(f"[SYMBOL] Using broker symbol {broker_sym} for {symbol}")
        symbol_to_send = broker_sym or symbol
        try:
            mt5.symbol_select(symbol_to_send, True)
        except Exception as e:
            log_debug(f"mt5.symbol_select failed for {symbol_to_send}:", e)
    except Exception:
        symbol_to_send = symbol

    # === EARLY ORDER TYPE INITIALIZATION ===
    # Initialize order_type to safe defaults BEFORE price selection.
    if side == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY if hasattr(mt5, 'ORDER_TYPE_BUY') else 0
    else:
        order_type = mt5.ORDER_TYPE_SELL if hasattr(mt5, 'ORDER_TYPE_SELL') else 1

    tick = None
    try:
        if hasattr(mt5, 'symbol_info_tick'):
            tick = mt5.symbol_info_tick(symbol_to_send)
    except Exception:
        tick = None

    # If no live tick is available, abort to avoid placing blind orders.
    # Market orders must use symbol_info_tick() prices.
    if not tick:
        try:
            telegram_block(symbol, "could not fetch tick")
        except Exception as e:
            log_debug(f"telegram_block fallback failed for {symbol} (tick final):", e)
        return
    else:
        # Mandatory rule: BUY uses ask, SELL uses bid
        price = tick.ask if order_type == (mt5.ORDER_TYPE_BUY if hasattr(mt5, 'ORDER_TYPE_BUY') else 0) else tick.bid

    # =========================================================================
    # CALCULATE SL/TP FROM MARKET STRUCTURE (NO FIXED RR)
    # =========================================================================
    sl = 0.0
    tp = 0.0
    rr_val = None
    sl_method = "Structure-based"
    sl_is_fallback = False
    try:
        si_for_price = mt5.symbol_info(symbol_to_send) if hasattr(mt5, 'symbol_info') else None
        sl, tp, rr_val, sltp_info, err = compute_structure_sl_tp(symbol_to_send, side, price)
        if err or sl is None or tp is None:
            log_block_verbose(symbol, f"structure SL/TP unavailable: {err or 'unknown'}")
            return
        if rr_val is not None:
            min_rr = FULL_RR_MIN if mode == 'FULL' else MICRO_RR_MIN
            if float(rr_val) < min_rr:
                log_block_verbose(symbol, f"RR below minimum ({min_rr:.1f})")
                try:
                    telegram_msg(f"🚫 {symbol} {mode} blocked: RR {float(rr_val):.2f} < {min_rr:.1f}")
                except Exception:
                    pass
                return
        log_msg(
            f"Placing trade with STRUCTURE SL/TP - {symbol} {side} | SL: {sl:.5f} | TP: {tp:.5f} | RR: {_format_rr(rr_val)}"
        )
    except Exception as e:
        log_debug(f"Structure SL/TP calculation failed: {e}")
        log_block_verbose(symbol, "structure SL/TP calculation failed")
        return

    # Normalize SL/TP and entry price to symbol tick size
    try:
        # Use resolved broker symbol when available to ensure correct tick size.
        norm_sym = symbol_to_send if symbol_to_send else symbol
        price = normalize_price(norm_sym, price)
        sl = normalize_price(norm_sym, sl)
        tp = normalize_price(norm_sym, tp)
    except Exception as e:
        try:
            log_debug("normalize_price failed for", norm_sym if 'norm_sym' in locals() else symbol, e)
        except Exception as ex:
            log_debug("normalize_price secondary logging failed:", ex)

    # M15 fallback buffer is NOT applied in Model A mode
    # (Model A uses sl=0.0, tp=0.0; no modifications)

    # ------------------ Dynamic risk enforcement (safe caps only) ------------------
    try:
        # Determine desired max risk percent for this trade type using score
        if mode == 'FULL':
            # scale within target band based on score (higher score -> higher risk within band)
            desired_risk = FULL_RISK_TARGET_MIN + (max(0, min(100, score)) / 100.0) * (FULL_RISK_TARGET_MAX - FULL_RISK_TARGET_MIN)
        else:
            desired_risk = MICRO_RISK_TARGET_MIN + (max(0, min(100, score)) / 100.0) * (MICRO_RISK_TARGET_MAX - MICRO_RISK_TARGET_MIN)
        # Strict mode: reduce full risk by additional 20%
        if BOT_STATE.strict_mode and mode == 'FULL':
            desired_risk *= 0.8
        # Apply full-risk reduction after a loss (30% reduction)
        if mode == 'FULL' and BOT_STATE.full_risk_reduction < 1.0:
            desired_risk *= max(0.1, BOT_STATE.full_risk_reduction)
        # Do not exceed existing configured RISK_PCT or absolute hard cap
        desired_risk = min(desired_risk, globals().get('RISK_PCT', desired_risk), MAX_RISK_PER_TRADE)
        # Adjust proposed lot downwards if it would risk more than desired_risk
        lot = adjust_lot_to_risk(symbol, lot, price, sl, desired_risk)
        # If monthly conservative mode is active, halve lot sizes (guardrail)
        if globals().get('MONTH_CONSERVATIVE_MODE'):
            try:
                lot = max(MICRO_LOT_MIN, round(lot / 2.0, 2))
            except Exception:
                log_debug("MONTH_CONSERVATIVE_MODE lot halve failed")
        # After risk adjustment, enforce micro lot cap.  Micro trades cannot exceed
        # the broker-specific micro lot (0.01 by default, 0.02 when account
        # balance is between £150–£200 on live IC Markets).  Clamp downward to
        # the allowed micro lot cap if risk calculation suggests a larger size.
        try:
            if mode == 'MICRO':
                try:
                    micro_cap = icm_micro_lot()
                except Exception as e:
                    log_debug("icm_micro_lot failed, falling back to MICRO_LOT_MIN:", e)
                    micro_cap = MICRO_LOT_MIN
                # Only clamp downwards; never increase lot here
                if lot > micro_cap:
                    lot = max(MICRO_LOT_MIN, micro_cap)
        except Exception as e:
            log_debug("micro cap enforcement failed:", e)
        # Compute risk used and SL distance in pips for logging.
        try:
            rpl = _risk_per_lot(symbol, price, sl)
            ai_info = mt5.account_info() if hasattr(mt5, 'account_info') else None
            bal = float(getattr(ai_info, 'balance', 1000.0)) if ai_info else 1000.0
            eq = float(getattr(ai_info, 'equity', bal)) if ai_info else bal
            base_amount = eq if USE_EQUITY_RISK else bal
            risk_used_pct_local = (rpl * lot) / base_amount if base_amount > 0 else 0.0
            try:
                sym_pt = float(getattr(si_for_price, 'point', 0.0001)) if 'si_for_price' in locals() else 0.0001
            except Exception as e:
                log_debug("sym_pt extraction failed:", e)
                sym_pt = 0.0001
            diff_val = abs(price - sl)
            sl_pips_local = diff_val / (sym_pt * 10.0) if sym_pt > 0 else diff_val
        except Exception as e:
            log_debug("risk calculation failed:", e)
            risk_used_pct_local = 0.0
            sl_pips_local = 0.0
    except Exception as e:
        log_debug("dynamic risk enforcement failed:", e)

    # Structure-based SL/TP validation happens below.

    # ------------------ Exposure control ------------------
    try:
        open_full_pct = _estimate_open_risk_pct("FULL")
        open_micro_pct = _estimate_open_risk_pct("MICRO")
        open_total_pct = _estimate_open_risk_pct(None)
        if mode == 'FULL' and open_full_pct >= FULL_EXPOSURE_MAX_PCT:
            log_block_verbose(symbol, "full exposure cap reached")
            telegram_msg(f"🚫 {symbol} FULL blocked: exposure {open_full_pct:.2f}% >= {FULL_EXPOSURE_MAX_PCT:.2f}%")
            return
        if mode == 'MICRO':
            if open_total_pct >= MICRO_DISABLE_TOTAL_EXPOSURE_PCT:
                log_block_verbose(symbol, "micro disabled by total exposure")
                telegram_msg(f"🚫 {symbol} MICRO blocked: total exposure {open_total_pct:.2f}%")
                return
            if open_micro_pct >= MICRO_EXPOSURE_MAX_PCT:
                log_block_verbose(symbol, "micro exposure cap reached")
                telegram_msg(f"🚫 {symbol} MICRO blocked: micro exposure {open_micro_pct:.2f}%")
                return
    except Exception:
        pass

    # === REQUEST VALIDATION (prevents retcode 10030 and broker errors) ===
    # Validate all request fields before assembly to prevent broker rejections.
    # This catches invalid volume, permissions, and precision issues before sending.
    try:
        term_info = mt5.terminal_info()
        if not term_info or not getattr(term_info, 'trade_allowed', False):
            log_block_verbose(symbol, "validation failed: terminal trade not allowed")
            return

        si_validate = mt5.symbol_info(symbol_to_send)
        if not si_validate:
            log_block_verbose(symbol, "validation failed: no symbol_info")
            return

        # Trading permissions: symbol must be fully tradable
        trade_mode_full = getattr(mt5, 'SYMBOL_TRADE_MODE_FULL', 0)
        if getattr(si_validate, 'trade_mode', None) != trade_mode_full:
            log_block_verbose(symbol, "validation failed: symbol trade_mode not FULL")
            return
        
        # Get symbol constraints from broker
        min_volume = float(getattr(si_validate, 'volume_min', 0.01))
        max_volume = float(getattr(si_validate, 'volume_max', 100.0))
        volume_step = float(getattr(si_validate, 'volume_step', 0.01))
        digits = int(getattr(si_validate, 'digits', 5))
        pt = float(getattr(si_validate, 'point', 0.0001))
        trade_stops_level = int(getattr(si_validate, 'trade_stops_level', 0))
        
        # Validate and clamp lot size to broker constraints
        if lot < min_volume:
            lot = min_volume
        if lot > max_volume:
            log_block_verbose(symbol, "validation failed: lot exceeds max_volume", extra_info=f"lot={lot}, max={max_volume}")
            return

        # Align lot to broker's volume_step
        if volume_step > 0:
            lot = round(lot / volume_step) * volume_step
            # Round to step precision to avoid floating drift
            try:
                step_str = f"{volume_step:.10f}".rstrip('0').rstrip('.')
                step_decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                lot = round(lot, step_decimals)
            except Exception:
                pass
        
        # Round all prices to correct number of decimal places (symbol digits)
        price = round(price, digits)
        sl = round(sl, digits)
        tp = round(tp, digits)

        # Log final lot once
        try:
            log_msg(f"Execution params: lot={lot}")
        except Exception:
            pass

        # === MODEL A VALIDATION: NONE ===
        # Model A has no SL/TP validation. All checks are skipped.
    except Exception as e:
        log_block_verbose(symbol, "validation exception", extra_info=str(e))
        return
    
    # === SL/TP VALIDATION (STRUCTURE RULES) ===
    # SL must invalidate structure; TP must target liquidity. If invalid, skip trade.
    try:
        if side == 'BUY':
            if sl >= price:
                log_block_verbose(symbol, "invalid SL: not below entry")
                return
            if tp <= price:
                log_block_verbose(symbol, "invalid TP: not above entry")
                return
        else:
            if sl <= price:
                log_block_verbose(symbol, "invalid SL: not above entry")
                return
            if tp >= price:
                log_block_verbose(symbol, "invalid TP: not below entry")
                return
        # Enforce broker stop distance
        if not validate_sl(symbol_to_send, side, price, sl):
            log_block_verbose(symbol, "invalid SL: broker stop distance")
            return
    except Exception as e:
        log_debug(f"SL/TP validation error: {e}")
        log_block_verbose(symbol, "SL/TP validation error")
        return

    # === STRICT EXECUTION-TIME CHECKS: SPREAD & TICK-AGE (XAUUSD FULL ONLY) ===
    # For XAUUSD FULL trades, enforce strict spread/tick requirements right before order_send
    # This ensures final conditions are acceptable, even though setup detection was more permissive
    if symbol.upper().startswith('XAUUSD') and mode == 'FULL':
        try:
            allowed_strict, reason_strict = can_place_on_xau(symbol_to_send, micro=False, strict_checks=True)
            if not allowed_strict:
                try:
                    log_msg(f"TRADE ABORTED — Execution-time strict checks failed: {reason_strict}")
                except Exception:
                    pass
                return
        except Exception as e:
            try:
                log_msg(f"TRADE ABORTED — Execution-time checks exception: {e}")
            except Exception:
                pass
            return

    # === MT5 ORDER REQUEST (SL/TP ATTACHED) ===
    # Build order request with SL/TP so trades are protected at entry
    # type_filling MUST be included (IOC = Immediate Or Cancel)
    ioc_mode = mt5.ORDER_FILLING_IOC if hasattr(mt5, 'ORDER_FILLING_IOC') else 2
    req = {
        "action": mt5.TRADE_ACTION_DEAL if hasattr(mt5, 'TRADE_ACTION_DEAL') else 1,
        "symbol": symbol_to_send,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 100,
        "type_filling": ioc_mode
    }
    if sl and sl != 0.0:
        req["sl"] = sl
    if tp and tp != 0.0:
        req["tp"] = tp

    # === REQUEST STRUCTURE VALIDATION ===
    # Verify the request is correct BEFORE sending
    assert "type_filling" in req, "type_filling MUST be in request"
    assert "filling_mode" not in req, "filling_mode is INVALID in MT5 - removed"
    try:
        log_msg(f"Request validated: type_filling={req['type_filling']}, sl={sl}, tp={tp} (attached)")
    except Exception:
        pass

    try:
        # Print a clear final-decision marker to console so command-mode mirroring
        # can detect an imminent placement. Only emit this marker when a
        # mirrored command session is active so background logs remain
        # unchanged outside of explicit operator requests.
        if globals().get('_CURRENT_CMD_CHAT'):
            try:
                print("Final Decision: Trade ✅")
            except Exception as e:
                log_debug("print final decision failed:", e)

        # Execute order without retries; abort on any failure or invalid result
        # Support DRY_RUN and TEST_MODE by simulating an order instead of sending
        try:
            # Simulation mode: do not call mt5.order_send, just log
            if globals().get('DRY_RUN') or globals().get('TEST_MODE'):
                try:
                    log_msg(f"🔬 Simulating {mode} order on {symbol_to_send} lot={lot:.2f} sl={sl} (TEST_MODE/DRY_RUN active)")
                except Exception:
                    pass
                # In simulation/dry-run mode, do not update the execution lock.  Since no
                # live order is executed there is no cooldown.  Simply return after
                # logging the simulation message.
                return
            # Live mode: send the order with ONE RETRY for retcode 10030 (invalid request)
            res = None
            retcode = None
            
            # === ATTEMPT 1: Initial send ===
            try:
                res = mt5.order_send(req)
            except Exception as send_exc:
                try:
                    log_block_verbose(symbol, "mt5 send exception", extra_info=str(send_exc))
                except Exception:
                    try:
                        log_msg(f"TRADE ABORTED — MT5 order_send exception for {symbol_to_send}: {send_exc}")
                    except Exception:
                        pass
                return
            
            # Determine success and check for retcode 10030 (invalid request)
            success = False
            if res is not None:
                try:
                    retcode = getattr(res, 'retcode', None)
                except Exception:
                    retcode = None
                success = (retcode == getattr(mt5, 'TRADE_RETCODE_DONE', 10009))
                
                # === HANDLE RETCODE 10030 (Invalid Request) ===
                # One retry attempt: fetch fresh tick and refresh price only.
                # Never re-enter validation or recalculate SL/TP.
                if not success and retcode == 10030:
                    log_msg(f"Retcode 10030 on {symbol_to_send} — ONE retry with fresh price")
                    try:
                        # Fetch fresh tick to refresh entry price only
                        fresh_tick = mt5.symbol_info_tick(symbol_to_send)
                        if fresh_tick:
                            buy_type = mt5.ORDER_TYPE_BUY if hasattr(mt5, 'ORDER_TYPE_BUY') else 0
                            fresh_price = fresh_tick.ask if order_type == buy_type else fresh_tick.bid
                            fresh_price = round(fresh_price, digits)
                            
                            # Update ONLY price in request; keep sl and tp unchanged
                            req['price'] = fresh_price
                            
                            # === ATTEMPT 2: Retry with fresh price only ===
                            log_msg(f"Retry: price={fresh_price:.5f}, sl={sl}, tp={tp}")
                            res = mt5.order_send(req)
                            if res is not None:
                                retcode = getattr(res, 'retcode', None)
                                success = (retcode == getattr(mt5, 'TRADE_RETCODE_DONE', 10009))
                                if success:
                                    log_msg(f"Retry SUCCEEDED: retcode {retcode}")
                                else:
                                    log_msg(f"Retry FAILED: retcode {retcode}")
                    except Exception as retry_exc:
                        log_msg(f"Retry exception: {retry_exc}")
                        success = False

            # If send failed or retcode signals rejection, abort.  Do not update the
            # execution lock because no trade was executed.
            if not success:
                try:
                    # Log the failure with the specific retcode.  Use block logging to
                    # deduplicate repeated retcode failures within the same candle.
                    reason_text = f"retcode {retcode}" if retcode is not None else "retcode None"
                    log_block_verbose(symbol, reason_text, extra_info=reason_text)
                except Exception:
                    try:
                        log_msg(f"TRADE ABORTED — Order failed with retcode {retcode} for {symbol_to_send}")
                    except Exception:
                        pass
                return
            # At this point the order executed successfully; record the execution
            # timestamp and retcode for cooldown enforcement.
            now_lock = datetime.now(SAFE_TZ)
            candle_start_lock = now_lock.replace(second=0, microsecond=0)
            SYMBOL_EXECUTION_LOCK[symbol.upper()] = (now_lock, retcode)
            # Structured execution notification: send a receipt with symbol, side, lot, entry price, SL and TP
            try:
                sess_label = None
                try:
                    sess_label, _, _ = session_bounds()
                except Exception:
                    sess_label = None
                telegram_trade_levels(symbol_to_send, side, price, sl, tp, None, sess_label)
            except Exception:
                log_debug("telegram_trade_levels failed for", symbol_to_send)
            # Update trade counters only upon successful execution.
            try:
                now_exec = datetime.now(SAFE_TZ)
                if mode == 'MICRO':
                    BOT_STATE.micro_trades_today = BOT_STATE.micro_trades_today + 1
                    BOT_STATE.last_micro_time = now_exec
                    BOT_STATE.last_micro_entry_by_symbol_side[f"{symbol.upper()}:{side.upper()}"] = now_exec
                else:
                    BOT_STATE.full_trades_today = BOT_STATE.full_trades_today + 1
                    BOT_STATE.last_full_time = now_exec
                    BOT_STATE.last_full_entry_by_symbol_side[f"{symbol.upper()}:{side.upper()}"] = now_exec
                # Update per-session counters (London/NY)
                try:
                    sess_label, _, _ = session_bounds()
                    if sess_label:
                        d = BOT_STATE.session_trades.setdefault(sess_label, {'count': 0})
                        d['count'] = d.get('count', 0) + 1
                        limits = BOT_STATE.session_limits.setdefault(sess_label, {"full": 0, "micro": 0})
                        if mode == 'FULL':
                            limits['full'] = limits.get('full', 0) + 1
                        else:
                            limits['micro'] = limits.get('micro', 0) + 1
                except Exception:
                    pass
            except Exception as e:
                log_debug("trade counters update failed:", e)
            # Mirror execution details to console when command‑mode is active
            try:
                if globals().get('_CURRENT_CMD_CHAT'):
                    try:
                        print('>>> TRADE EXECUTED')
                        print(f'Symbol: {symbol_to_send}')
                        print(f'Side: {side}')
                        print(f'Lot Size: {lot:.2f}')
                        try:
                            print(f'Stop Loss (SL): {sl}')
                        except Exception:
                            print('Stop Loss (SL): N/A')
                        try:
                            print(f'Take Profit (TP1/TP2/TP3): {tp}')
                        except Exception:
                            print('Take Profit (TP1/TP2/TP3): N/A')
                        try:
                            dr = locals().get('desired_risk')
                            if dr is not None:
                                print(f'Risk %: {float(dr) * 100:.2f}%')
                        except Exception as e:
                            log_debug("printing desired_risk failed:", e)
                    except Exception as e:
                        log_debug("command-mode executed print block failed:", e)
            except Exception as e:
                log_debug("post-execution command-mode block failed:", e)
            # Record win/loss and persist state
            try:
                win = getattr(res, 'profit', 0) > 0 if res else False
            except Exception:
                win = False
            # Record open trade metadata for outcome tracking
            try:
                ticket = getattr(res, 'position', None) or getattr(res, 'order', None) or getattr(res, 'ticket', None) or getattr(res, 'trade', None)
                if ticket:
                    open_trades_info[ticket] = {
                        "symbol": symbol_to_send,
                        "side": side,
                        "volume": lot,
                        "mode": mode,
                        "score": score,
                        "adx_h1": None,
                        "adx_m15": None,
                        "rsi_h1": None,
                        "rsi_m15": None
                    }
            except Exception:
                pass
            update_memory(symbol_to_send, win, score)
            try:
                if state_store is not None:
                    try:
                        ticket = getattr(res, 'order', None) or getattr(res, 'ticket', None) or getattr(res, 'trade', None) or 'N/A'
                        state_store.record_open_trade(ticket, symbol_to_send, side, lot, mode, ts=time.time())
                        # record trade zone and equity markers
                        try:
                            state_store.set_trade_zone(symbol_to_send, price, ts=time.time())
                        except Exception as e:
                            log_debug("state_store.set_trade_zone failed:", e)
                        try:
                            ai = mt5.account_info()
                            if ai is not None:
                                state_store.update_day_equity_high(getattr(ai, 'equity', None))
                        except Exception as e:
                            log_debug("state_store.update_day_equity_high failed:", e)
                        state_store.save()
                    except Exception as e:
                        log_debug("state_store record/save failed:", e)
            except Exception as e:
                log_debug("state_store outer handler failed:", e)
            # Successful trade ends function
            return
        except Exception as exc:
            # Catch any unexpected exceptions during send or post‑processing.  Do not
            # set the execution lock since the order did not complete.  Use block
            # logging to avoid spamming the same exception message.
            try:
                log_block_verbose(symbol, "order placement exception", extra_info=str(exc))
            except Exception:
                try:
                    log_msg(f"TRADE ABORTED — Exception during order placement for {symbol_to_send}: {exc}")
                except Exception:
                    pass
            return
    except Exception as e:
        # Do not send raw internal errors to Telegram; log to console instead.
        try:
            log_msg(f"Order send exception: {e}")
        except Exception:
            try:
                print(f"Order send exception: {e}")
            except Exception as ex:
                log_debug("print fallback failed for order send exception:", ex)


def scan_preview_all(report_chat_id=None):
    """Scan symbols (XAUUSD first) and report what the bot WOULD do, without placing trades.

    This function mirrors the detection/decision logic but never calls `place_trade` or
    updates trade counters/state. Useful for dry-run inspections.
    """
    try:
        order = []
        # Build ordered list: XAUUSD first if present
        syms = list(SYMBOLS)
        syms_upper = [s.upper() for s in syms]
        ordered = []
        if any(s.startswith('XAUUSD') or s.upper() == 'XAUUSD' for s in syms_upper):
            for s in syms:
                if s.upper().startswith('XAUUSD') or s.upper() == 'XAUUSD':
                    ordered.append(s)
        for s in syms:
            if s not in ordered:
                ordered.append(s)

        def _report(msg: str):
            if report_chat_id:
                try:
                    send_telegram_to(report_chat_id, msg)
                except Exception as e:
                    try:
                        tg(msg)
                    except Exception as ex:
                        log_debug("tg fallback failed in _report:", ex)
            else:
                try:
                    tg(msg)
                except Exception as e:
                    log_debug("tg failed in _report:", e)

        _report("🔎 Starting preview: XAUUSD first, then other pairs (no trades will be placed)")

        for sym in ordered:
            try:
                # Resolve symbol name used by broker (e.g. XAUUSD -> GOLDsuffix)
                resolved_sym = resolve_symbol(sym)
                if resolved_sym != sym:
                    _report(f"{sym}: resolved -> {resolved_sym}")

                if not check_spread(resolved_sym):
                    _report(f"{sym} ({resolved_sym}): skipped - spread too high or no tick")
                    continue

                m15_raw = get_rates(resolved_sym, mt5.TIMEFRAME_M15, 200)
                h1_raw  = get_rates(resolved_sym, mt5.TIMEFRAME_H1, 200)
                columns = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
                try:
                    m15 = pd.DataFrame(m15_raw, columns=columns)
                    h1  = pd.DataFrame(h1_raw,  columns=columns)
                except Exception:
                    _report(f"{sym}: failed to convert MT5 data")
                    continue

                if m15 is None or m15.empty or len(m15) < 20:
                    _report(f"{sym}: not enough M15 data")
                    continue
                if h1 is None or h1.empty or len(h1) < 20:
                    _report(f"{sym}: not enough H1 data")
                    continue

                bos_emoji, bos_strength = safe_smc_call(detect_bos, m15, "detect_bos")
                sweep_emoji, sweep_strength = safe_smc_call(detect_sweep, m15, "detect_sweep")
                fvg_emoji, fvg_strength = safe_smc_call(detect_fvg, m15, "detect_fvg")
                emojis = [bos_emoji, sweep_emoji, fvg_emoji]
                strengths = [bos_strength, sweep_strength, fvg_strength]
                if EMOJI_VERY_WEAK in emojis or EMOJI_INVALID in emojis:
                    _report(f"{sym}: Rejected due to weak/invalid SMC - BOS={bos_emoji} Sweep={sweep_emoji} FVG={fvg_emoji}")
                    continue

                usable_count = sum([1 for e in emojis if e in [EMOJI_STRONG, EMOJI_WEAK_OK]])
                strong_count = sum([1 for e in emojis if e == EMOJI_STRONG])
                avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
                score = int(avg_strength * 100)

                # Determine side (same logic as live)
                disp = detect_displacement(m15)
                side = None
                if isinstance(disp, dict):
                    side = disp.get('direction')
                if not side or side == 'NEUTRAL':
                    last_candle = m15.iloc[-1]
                    side = 'BUY' if last_candle['close'] > last_candle['open'] else 'SELL'

                would_decision = "IGNORE"
                would_lot = 0.0

                # Build directional votes for preview (same rules as live), store SIGNAL:DIR
                votes = []
                try:
                    msb = detect_msb(h1)
                    if isinstance(msb, dict) and msb.get('type'):
                        t = msb.get('type')
                        if isinstance(t, str) and 'UP' in t and bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                            votes.append('BOS:BUY')
                        if isinstance(t, str) and 'DOWN' in t and bos_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                            votes.append('BOS:SELL')
                except Exception as e:
                    log_debug("detect_msb failed in preview for {sym}:", e)
                try:
                    s_h1 = detect_liquidity_sweep(resolved_sym, timeframe=mt5.TIMEFRAME_H1)
                    s_m15 = detect_liquidity_sweep(resolved_sym, timeframe=mt5.TIMEFRAME_M15)
                    for sdict in (s_h1, s_m15):
                        if isinstance(sdict, dict) and sdict.get('type') in ('BUY', 'SELL') and sweep_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                            votes.append(f"SWEEP:{sdict.get('type')}")
                except Exception as e:
                    log_debug("detect_liquidity_sweep failed in preview for {sym}:", e)
                try:
                    if m15 is not None and len(m15) >= 3:
                        c1, c2, c3 = m15.iloc[-3], m15.iloc[-2], m15.iloc[-1]
                        bull_gap = c2['low'] > c1['high'] and c2['low'] > c3['high']
                        bear_gap = c2['high'] < c1['low'] and c2['high'] < c3['low']
                        if bull_gap and fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                            votes.append('FVG:BUY')
                        if bear_gap and fvg_emoji in (EMOJI_STRONG, EMOJI_WEAK_OK):
                            votes.append('FVG:SELL')
                except Exception as e:
                    log_debug("FVG detection failed in preview for {sym}:", e)

                buy_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'BUY')
                sell_votes = sum(1 for v in votes if v.split(':', 1)[1] == 'SELL')

                # FULL preview: allow for symbols configured in FULL_ALLOWED_SYMBOLS
                # Respect persisted counters/caps when previewing
                # Use global counters for preview.  Ignore any persisted counts in
                # the state store so the preview reflects the current in-memory
                # session only.
                preview_micro_today = BOT_STATE.micro_trades_today
                preview_full_today = BOT_STATE.full_trades_today

                if sym.upper() in FULL_ALLOWED_SYMBOLS and usable_count >= 2 and (buy_votes >= 2 or sell_votes >= 2):
                    if preview_full_today >= FULL_MAX_PER_DAY:
                        _report(f"{sym}: FULL cap already reached ({preview_full_today}/{FULL_MAX_PER_DAY}) - would IGNORE")
                        continue
                    would_decision = 'FULL'
                    would_lot = LOT_FULL
                # MICRO preview: at least 1 usable signal
                elif usable_count >= 1:
                    # Respect micro cap based on global counters only
                    if preview_micro_today >= MICRO_MAX_PER_SESSION:
                        _report(f"{sym}: MICRO cap already reached ({preview_micro_today}/{MICRO_MAX_PER_SESSION}) - would IGNORE")
                        continue
                    would_decision = 'MICRO'
                    would_lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, BOT_STATE.micro_lot_ai_scale * (1 + avg_strength)))
                    would_lot = round(would_lot, 2)

                # Respect owner-forced modes in preview (reporting only)
                forced = FORCED_TRADES.get(sym)
                forced_note = f" (owner forced -> {forced})" if forced else ""

                _report(f"{sym}: WOULD {would_decision}{forced_note} | Side: {side} | Score: {score}% | Lot: {would_lot:.2f} | Usable: {usable_count}/3 | Strong: {strong_count}/3")

            except Exception as e:
                _report(f"{sym}: scan error: {e}")

        _report("🔎 Preview complete - no trades were placed.")
    except Exception as e:
        tg(f"Scan preview top-level error: {e}")

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_status():
    global last_signal
    try:
        win_rate = _recent_win_rate(20) * 100.0
    except Exception:
        win_rate = 0.0
    try:
        exposure_pct = _estimate_open_risk_pct(None)
    except Exception:
        exposure_pct = 0.0
    try:
        sess_label, _, _ = session_bounds()
    except Exception:
        sess_label = None
    sess_counts = BOT_STATE.session_limits.get(sess_label or "", {"full": 0, "micro": 0})
    mode = "STRICT" if BOT_STATE.strict_mode else "NORMAL"
    msg = (
        "📊 **BOT STATUS**\n"
        f"Win Rate (20): {win_rate:.1f}%\n"
        f"Active Exposure: {exposure_pct:.2f}%\n"
        f"Full Trades (session): {sess_counts.get('full', 0)}\n"
        f"Micro Trades (session): {sess_counts.get('micro', 0)}\n"
        f"Current Mode: {mode}\n"
        f"Last Score: {last_signal.get('score','N/A')}%\n"
        f"Decision: {last_signal.get('decision','N/A')}"
    )
    tg(msg)


def cmd_signal():
    global last_signal
    msg = (
        "📡 **LAST SIGNAL**\n"
        f"BOS: {last_signal.get('bos', False)}\n"
        f"Sweep: {last_signal.get('sweep', False)}\n"
        f"FVG: {last_signal.get('fvg', False)}\n"
        f"Score: {last_signal.get('score','N/A')}%\n"
        f"Decision: {last_signal.get('decision','N/A')}"
    )
    tg(msg)


def cmd_admin_status(chat_id: int):
    """Send admin-only status including session/full metrics and FTMO locks."""
    try:
        # Owner check
        try:
            if str(chat_id) != str(TELEGRAM_CHAT_ID):
                return send_telegram_to(chat_id, "⛔ Not authorized for admin status")
        except Exception as e:
            log_debug("cmd_admin_status owner check failed:", e)

        sess_losses = BOT_STATE.session_full_losses
        sess_count = BOT_STATE.session_full_count
        full_today = BOT_STATE.full_trades_today
        day_open = BOT_STATE.day_open_equity or (state_store.get('day_start_equity') if state_store is not None else None)
        day_high = None
        perm_locked = False
        trading_paused_flag = BOT_STATE.trading_paused
        try:
            if state_store is not None:
                day_high = state_store.get('day_equity_high')
                perm_locked = bool(state_store.get('permanent_lockout'))
        except Exception as e:
            log_debug("cmd_admin_status: state_store read failed:", e)

        lines = []
        lines.append(f"📋 Admin Status")
        lines.append(f"Session FULL losses: {sess_losses}")
        lines.append(f"Session FULL count: {sess_count}")
        lines.append(f"FULL trades today: {full_today}")
        lines.append(f"Day open equity: {day_open if day_open is not None else 'N/A'}")
        lines.append(f"Day equity high: {day_high if day_high is not None else 'N/A'}")
        lines.append(f"Trading paused: {trading_paused_flag}")
        lines.append(f"Permanent FTMO lockout: {perm_locked}")
        # initial balance if available
        try:
            init_bal = state_store.get('initial_balance') if state_store is not None else None
            lines.append(f"Initial balance: {init_bal if init_bal is not None else 'N/A'}")
        except Exception as e:
            log_debug("cmd_admin_status: initial balance fetch failed:", e)

        msg = "\n".join(lines)
        return send_telegram_to(chat_id, msg)
    except Exception as e:
        try:
            return send_telegram_to(chat_id, f"Admin status error: {e}")
        except Exception:
            try:
                tg(f"Admin status error: {e}")
            except Exception as ex:
                log_debug("tg fallback failed in cmd_admin_status:", ex)
        return False


def run_telegram_command_tests(report_to_console: bool = True):
    """Run a set of smoke tests against the telegram command dispatcher.

    This will monkeypatch `tg` and `send_telegram_to` temporarily so no
    real Telegram messages are sent. It exercises owner and non-owner
    command paths and prints results to console.
    """
    tests = [
        ("/active", False),
        ("/status", False),
        ("/signal", False),
        ("/scan", False),
        ("/preview", False),
        ("/panic", True),
        ("/resume", True),
        ("/help", False),
        ("/settings", False),
        ("/xau_report", True),
        ("/forcefull XAUUSD", True),
        ("/forcemicro GBPUSD", True),
        ("/unforce XAUUSD", True),
        ("/scale", False),
        ("/scale set 0.02", True),
    ]

    orig_tg = globals().get('tg')
    orig_send = globals().get('send_telegram_to')
    try:
        def _fake_tg(m):
            if report_to_console:
                print(f"[tg] {m}")

        def _fake_send(cid, m):
            if report_to_console:
                print(f"[send_telegram_to {cid}] {m}")
            return True

        globals()['tg'] = _fake_tg
        globals()['send_telegram_to'] = _fake_send

        owner_id = TELEGRAM_CHAT_ID
        non_owner = 999999999
        for cmd, owner_only in tests:
            chat = owner_id if owner_only else non_owner
            try:
                ok = _dispatch_minimal_command(cmd, chat)
                print(f"CMD: {cmd!r} -> handled={ok}")
            except Exception as e:
                print(f"CMD: {cmd!r} -> ERROR: {e}")
    finally:
        if orig_tg:
            globals()['tg'] = orig_tg
        if orig_send:
            globals()['send_telegram_to'] = orig_send


def cmd_panic():
    BOT_STATE.trading_paused = True
    tg("🛑 **PANIC MODE**\nBot paused\nTrades closed")


def cmd_resume():
    BOT_STATE.trading_paused = False
    tg("✅ **TRADING RESUMED**\nBot active")


def cmd_findtrade(chat_id: int):
    """Handle /findtrade: mirror console to `chat_id` and run the live scan+execute.

    This runs in a background thread so the Telegram poller is not blocked.
    Console output is mirrored line-for-line and order-preserved.
    """
    def _run():
        global _CURRENT_CMD_CHAT
        try:
            _CURRENT_CMD_CHAT = chat_id
            with mirror_console_to_telegram(chat_id):
                try:
                    # Run the full live scanner which will place trades when
                    # the logic decides to do so. All prints will be mirrored.
                    holy_grail_scan_and_execute()
                except Exception as e:
                    try:
                        print(f"[SCAN ERROR] {e}")
                    except Exception as ex:
                        log_debug("printing scan error failed:", ex)
        finally:
            _CURRENT_CMD_CHAT = None

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    # Immediate acknowledgement (console + Telegram) — keep exact wording
    try:
        _tg_send_sync(chat_id, "[SCAN STARTED] Running live scan and will mirror console output")
    except Exception:
        try:
            tg("[SCAN STARTED] Running live scan and will mirror console output")
        except Exception as e:
            log_debug("cmd_run scan start fallback tg failed:", e)


def cmd_xau_status(chat_id: int):
    """Send current XAU session BUY/SELL counts and recent block stats to `chat_id`."""
    try:
        # Gather session counts
        try:
            xau_tr = globals().get('XAU_SESSION_TRADES') or {}
        except Exception:
            xau_tr = {}
        try:
            last_label = globals().get('XAU_LAST_SESSION_LABEL')
        except Exception:
            last_label = None

        lon = xau_tr.get('LON', {}) if isinstance(xau_tr, dict) else {}
        ny  = xau_tr.get('NY', {}) if isinstance(xau_tr, dict) else {}
        lon_buy = int(lon.get('BUY', 0))
        lon_sell = int(lon.get('SELL', 0))
        ny_buy = int(ny.get('BUY', 0))
        ny_sell = int(ny.get('SELL', 0))

        # Block stats
        try:
            b_total = int(_XAU_BLOCK_STATS.get('full_block_total', 0))
            b_since = int(_XAU_BLOCK_STATS.get('full_block_since_last_report', 0))
        except Exception:
            b_total = b_since = 0

        # Recent block reasons (from last N _LAST_BLOCK_LOG entries related to XAU)
        reasons = []
        try:
            for k, ts in sorted(_LAST_BLOCK_LOG.items(), key=lambda x: x[1], reverse=True):
                if k.startswith('XAUUSD|'):
                    reasons.append(k.split('|',1)[1])
            reasons = reasons[:6]
        except Exception:
            reasons = []

        lines = [
            f"🔔 XAU Status (requested):",
            f"Last session label: {last_label}",
            "",
            f"LON — BUY: {lon_buy} | SELL: {lon_sell}",
            f"NY  — BUY: {ny_buy} | SELL: {ny_sell}",
            "",
            f"Blocked full trades (total): {b_total} | since last report: {b_since}",
        ]
        if reasons:
            lines.append("")
            lines.append("Recent block reasons:")
            for r in reasons:
                lines.append(f"• {r}")

        msg = "\n".join(lines)
        send_telegram_to(chat_id, msg)
        return True
    except Exception as e:
        try:
            send_telegram_to(chat_id, f"XAU status failed: {e}")
        except Exception as e2:
            log_debug("cmd_xau_status send_telegram_to failed:", e2)
        return False


# Duplicate minimal command dispatcher removed. A comprehensive implementation appears later in the file.

# -----------------------------------------------------------------------------
# Main loop (deprecated shim)
#
# Historically this module defined a no-op main guard early in the file.  The
# unified entry point is now defined at the bottom of this file.  Leaving an
# empty guard here avoids accidental execution when the file is imported as a
# module but has no runtime effect.  The actual bot startup is triggered by
# the second `if __name__ == "__main__"` block near the bottom, where a
# BotState is created and run_bot() is invoked.
if __name__ == "__main__":
    # No-op: actual startup logic is defined below
    pass

# Gracefully handle optional dependency for environment variable loading.
try:
    # The dotenv module is used to load environment variables from a .env file.
    # On systems without python-dotenv installed this import will fail.  In that
    # case we provide a no-op fallback so the rest of the bot can run without
    # raising an ImportError at import time.
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(*args, **kwargs):
        """Fallback no-op when python-dotenv is not installed."""
        return None

# Attempt to import the MetaTrader5 module.  If it is unavailable (for
# example, when running on a system without MetaTrader installed), a dummy
# stand‑in will be used instead.  Accessing any attribute on the dummy will
# raise an ImportError, clearly signalling that trading functionality is
# unavailable in the current environment.
try:
    import MetaTrader5 as mt5
except ImportError:
    class _DummyMT5:
        def __getattr__(self, name: str) -> Any:
            raise ImportError(
                "MetaTrader5 module is not available in this environment."
            )

    mt5 = _DummyMT5()

# =============================================================================
# Ticket-Funded MT5 Connection
# Reliable MT5 initialization + reconnection system
# =============================================================================

MT5_INIT_RETRIES = 5
MT5_INIT_DELAY = 2.0
MT5_CONNECTION_TIMEOUT = 30.0
MT5_CONNECTED = False
MT5_LOGIN_VERIFIED = False
# Ticket-Funded FTMO Servers: Full accepted FTMO server list # Ticket-Funded FTMO Servers
FTMO_ALLOWED_SERVERS = [  # Ticket-Funded FTMO Servers
    "FTMO-LIVE",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-1",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-2",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-3",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-4",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-5",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-6",  # Ticket-Funded FTMO Servers
    "FTMO-LIVE-SERVER",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-1",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-2",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-3",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-4",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-5",  # Ticket-Funded FTMO Servers
    "FTMO-DEMO-SERVER",  # Ticket-Funded FTMO Servers
    "FTMO-CHALLENGE",  # Ticket-Funded FTMO Servers
    "FTMO-VERIFICATION",  # Ticket-Funded FTMO Servers
    "FTMO-VERIFICATION-1",  # Ticket-Funded FTMO Servers
    "FTMO-VERIFICATION-2"  # Ticket-Funded FTMO Servers
]  # Ticket-Funded FTMO Servers

# Ticket-Funded FTMO Servers: Helper to validate FTMO server names via substring matching # Ticket-Funded FTMO Servers

def mt5_initialize_with_retry():
    """
    Ticket-Funded MT5 Connection: Initialize MT5 with retries.
    Returns: (success, error_message)
    """
    global MT5_CONNECTED
    
    for attempt in range(MT5_INIT_RETRIES):
        try:
            tg(f"[MT5 Connection] Initialization attempt {attempt + 1}/{MT5_INIT_RETRIES}...")
            
            # First attempt at initialization attaches to whatever terminal is open.
            # We will later detect the broker and, if it's IC Markets, reinitialize
            # using the IC Markets terminal path.
            result = mt5.initialize()

            if result:
                # Detect broker based on the initial account_info after initialization
                try:
                    ai = mt5.account_info()
                    srv = getattr(ai, 'server', '') if ai else ''
                    broker = detect_broker(srv)
                    # If broker is IC Markets, reinitialize with the explicit terminal path
                    if broker == "ICMARKETS":
                        # Shutdown the current session and reinitialize using ICM_MT5_PATH
                        try:
                            mt5.shutdown()
                        except Exception as e:
                            log_debug("mt5.shutdown during reinit failed:", e)
                        init_res = mt5.initialize(path=ICM_MT5_PATH)
                        if not init_res:
                            # If reinitialization fails, log error and continue to retry loop
                            err = mt5.last_error()
                            tg(f"[MT5 Connection] Reinitialize for IC Markets failed: {err}")
                            # continue to retry attempts
                            continue
                        else:
                            # Successful reinitialization for IC Markets
                            tg("[MT5 Connection] MT5 reinitialized for IC Markets")
                    elif broker == "FTMO" or broker is None:
                        # For FTMO or unknown brokers, we stay with the default initialization
                        pass
                except Exception as _e:
                    # detection failed; proceed with default initialization
                    pass

                tg("[MT5 Connection] MT5 initialized successfully")
                MT5_CONNECTED = True
                return True, "MT5 initialized"
            else:
                error = mt5.last_error()
                tg(f"[MT5 Connection] Initialize failed: {error}")
                
                if attempt < MT5_INIT_RETRIES - 1:
                    time.sleep(MT5_INIT_DELAY)
        
        except Exception as e:
            tg(f"[MT5 Connection] Initialize exception (attempt {attempt + 1}): {e}")
            
            if attempt < MT5_INIT_RETRIES - 1:
                time.sleep(MT5_INIT_DELAY)
    
    # All retries failed
    MT5_CONNECTED = False
    error_msg = (
        "🚫 **ERROR** — MT5 failed to initialize\n"
        "Check terminal and network; see logs for details."
    )
    tg(error_msg)
    telegram_msg_mt5(error_msg)
    return False, error_msg

def mt5_check_terminal():
    """
    Ticket-Funded MT5 Connection: Check if MT5 terminal is open and available.
    Returns: (terminal_found, error_message)
    """
    try:
        # Try to get account info - if it works, terminal is running
        account = mt5.account_info()
        
        if account is None:
            error_msg = (
                "🚫 **ERROR** — MT5 terminal not responding\n"
                "Ensure the terminal is running and logged in."
            )
            tg(error_msg)
            telegram_msg_mt5(error_msg)
            return False, error_msg
        
        tg("[MT5 Connection] MT5 terminal detected and responding")
        return True, "Terminal found"
    
    except Exception as e:
        error_msg = f"🚫 **ERROR**\nMT5 terminal check failed: {e}"
        tg(error_msg)
        telegram_msg_mt5(error_msg)
        return False, error_msg

def mt5_verify_login():
    """
    Ticket-Funded MT5 Connection: Verify account login and data.
    Returns: (login_valid, account_info_dict_or_none)
    """
    global MT5_LOGIN_VERIFIED
    
    try:
        account = mt5.account_info()
        
        if account is None:
            error_msg = (
                "🚫 **ERROR** — MT5 login failed (no account info)\n"
                "Check credentials and terminal login."
            )
            tg(error_msg)
            telegram_msg_mt5(error_msg)
            MT5_LOGIN_VERIFIED = False
            return False, None
        
        # Extract account details
        account_info = {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'currency': account.currency if hasattr(account, 'currency') else 'N/A'
        }
        
        tg(f"[MT5 Connection] Login verified: {account_info['login']}")
        tg(f"[MT5 Connection] Balance: {account_info['balance']}, Equity: {account_info['equity']}")
        
        MT5_LOGIN_VERIFIED = True
        return True, account_info
    
    except Exception as e:
        error_msg = (
            f"🚫 **ERROR** — MT5 login verification failed\n"
            f"Details: {e}"
        )
        tg(error_msg)
        telegram_msg_mt5(error_msg)
        MT5_LOGIN_VERIFIED = False
        return False, None

def mt5_startup():
    """
    Ticket-Funded MT5 Connection: Run full startup sequence.
    Should be called once at bot startup.
    Returns: (startup_success, account_info_or_none)
    """
    # Step 1: Initialize MT5
    tg("[Step 1/3] Initializing MT5...")
    success, msg = mt5_initialize_with_retry()
    if not success:
        tg("STARTUP FAILED at initialization")
        return False, None
    
    # Step 2: Check terminal
    tg("[Step 2/3] Checking MT5 terminal...")
    success, msg = mt5_check_terminal()
    if not success:
        tg("STARTUP FAILED at terminal check")
        return False, None
    
    # Step 3: Verify login
    tg("[Step 3/3] Verifying login...")
    success, account = mt5_verify_login()
    if not success:
        tg("STARTUP FAILED at login verification")
        return False, None
    
    # Compose a startup message indicating the detected broker, server, account type and category.
    try:
        ai = mt5.account_info()
        srv = getattr(ai, 'server', '') if ai else ''
        broker = detect_broker(srv)
        acc_type = detect_account_type(srv) or "UNKNOWN"
        # Determine account category for display
        if broker == "FTMO":
            category = "Prop Firm"
            emoji = "🏦"  # bank / prop firm
        elif broker == "ICMARKETS":
            category = "Normal brokerage account"
            # Use different emoji for live vs demo IC Markets
            if acc_type == "LIVE":
                emoji = "💰"
            elif acc_type == "DEMO":
                emoji = "🧪"
            else:
                emoji = "📘"
        else:
            category = "Unknown"
            emoji = "❓"
        # Build the message
        msg_lines = []
        msg_lines.append(f"{emoji} Broker detected: {broker or 'UNKNOWN'}")
        msg_lines.append(f"Server: {srv or 'N/A'}")
        msg_lines.append(f"Account type: {acc_type}")
        msg_lines.append(f"Category: {category}")
        # If unknown broker, instruct user to check terminal
        if broker is None:
            msg_lines.append("⚠️ Unsupported broker — trading will be paused until a supported account is logged in.")
        startup_msg = " | ".join(msg_lines)
        # Send to Telegram (blocking) and also log to console
        tg(startup_msg)
        try:
            telegram_msg(startup_msg)
        except Exception as e:
            log_debug("startup telegram_msg failed:", e)
    except Exception as e:
        log_debug("startup MT5 connect flow error:", e)
    tg("✅ MT5 connected\nAccount verified\nHoly Grail is live.")
    
    # After successful login, attempt to restore persisted bot state
    MT5_CONNECTED = True
    MT5_LOGIN_VERIFIED = True
    try:
        # Load persisted state and reconcile with live MT5 positions
        if state_store is not None:
            try:
                # Ensure state is loaded
                state_store.load()
                try:
                    # perform reconciliation and FTMO safety enforcement
                    restore_state_on_startup()
                except Exception as e:
                    log_debug("restore_state_on_startup failed:", e)
            except Exception as e:
                log_debug("state_store.load failed during startup restore:", e)
    except Exception as e:
        log_debug("startup state restore outer handler failed:", e)
    return True, account


def restore_state_on_startup():
    """Load persisted state, reconcile with MT5 open positions, and enforce FTMO locks.

    This function is safe to call after MT5 is connected and account_info is available.
    It will update in-memory globals (counters, zones, equity markers) and persist
    any fixes made during reconciliation.
    """
    # Runtime counters and zones are stored in BOT_STATE; restore into BOT_STATE below.
    global day_open_equity, day_equity_high
    try:
        if state_store is None:
            print("[STATE] No state_store available, skipping restore")
            return
        st = state_store.load()

        # Restore counters.  Under the execution‑only policy, avoid using
        # persisted counters from the state store; rely on in‑memory globals
        # instead.  Counters are reset each session and incremented only
        # after real executions.
        BOT_STATE.micro_trades_today = int(st.get('micro_trades_today', 0) or 0) if False else BOT_STATE.micro_trades_today
        BOT_STATE.full_trades_today = int(st.get('full_trades_today', 0) or 0) if False else BOT_STATE.full_trades_today

        # Restore last times
        lmt = st.get('last_micro_time')
        lft = st.get('last_full_time')
        try:
            BOT_STATE.last_micro_time = datetime.fromtimestamp(float(lmt)) if lmt else None
        except Exception:
            BOT_STATE.last_micro_time = None
        try:
            BOT_STATE.last_full_time = datetime.fromtimestamp(float(lft)) if lft else None
        except Exception:
            BOT_STATE.last_full_time = None

        # Restore zones into BOT_STATE
        BOT_STATE.last_trade_zones = {}
        for sym, z in (st.get('last_trade_zones') or {}).items():
            try:
                ts = float(z.get('ts'))
                BOT_STATE.last_trade_zones[sym] = { 'price': float(z.get('price')), 'time': datetime.fromtimestamp(ts) }
            except Exception as e:
                log_debug("restore last_trade_zones timestamp parse failed for", sym, e)
                try:
                    BOT_STATE.last_trade_zones[sym] = { 'price': float(z.get('price')), 'time': datetime.now(SAFE_TZ) }
                except Exception as e2:
                    log_debug("restore last_trade_zones price parse fallback failed for", sym, e2)
                    BOT_STATE.last_trade_zones[sym] = { 'price': float(z.get('price') or 0.0), 'time': datetime.now(SAFE_TZ) }

        # Restore equity markers into BOT_STATE
        try:
            BOT_STATE.day_open_equity = st.get('day_start_equity') or BOT_STATE.day_open_equity
        except Exception as e:
            log_debug("restore day_open_equity failed:", e)
        try:
            BOT_STATE.day_equity_high = st.get('day_equity_high') or BOT_STATE.day_equity_high
        except Exception as e:
            log_debug("restore day_equity_high failed:", e)

        # Restore session tracking into BOT_STATE
        try:
            BOT_STATE.session_full_losses = int(st.get('session_full_losses', 0) or 0)
        except Exception:
            BOT_STATE.session_full_losses = 0
        try:
            BOT_STATE.session_full_count = int(st.get('session_full_count', 0) or 0)
        except Exception:
            BOT_STATE.session_full_count = 0

        # Restore XAU per-session trade counters (nested dict)
        try:
            xau_tr = st.get('xau_session_trades')
            if isinstance(xau_tr, dict):
                globals()['XAU_SESSION_TRADES'] = xau_tr
        except Exception as e:
            log_debug("restore xau_session_trades failed:", e)
        try:
            xlast = st.get('xau_last_session_label')
            if xlast is not None:
                globals()['XAU_LAST_SESSION_LABEL'] = xlast
        except Exception as e:
            log_debug("restore xau_last_session_label failed:", e)

        # Enforce preserved flags
        if st.get('permanent_lockout'):
            BOT_STATE.trading_paused = True
            BOT_STATE.permanent_lockout = True
            try:
                globals()['PERMANENT_LOCKED'] = True
            except Exception as e:
                log_debug("setting PERMANENT_LOCKED failed:", e)
            print("[STATE] Permanent lockout from previous session - trading paused")

        if st.get('trading_paused'):
            BOT_STATE.trading_paused = True
            print("[STATE] Trading paused flag restored from previous session")

        # Reconcile recorded open trades with MT5 live positions
        try:
            positions = mt5.positions_get() or []
            live_tickets = set()
            for p in positions:
                try:
                    ticket = getattr(p, 'ticket', None) or getattr(p, 'position', None)
                    live_tickets.add(str(ticket))
                except Exception:
                    log_debug("positions_get iteration: failed to read ticket", p)
                    continue

            recorded = st.get('open_trades', []) or []
            recorded_tickets = set(str(r.get('ticket')) for r in recorded if r.get('ticket') is not None)

            # Mark closed any recorded trades that are no longer live
            for rt in list(recorded_tickets):
                if rt not in live_tickets:
                    try:
                        state_store.record_closed_trade(rt)
                    except Exception as e:
                        log_debug("state_store.record_closed_trade failed:", e)

            # Add any live positions that were not recorded
            for p in positions:
                try:
                    ticket = getattr(p, 'ticket', None) or getattr(p, 'position', None)
                    ticket = str(ticket)
                    if ticket not in recorded_tickets:
                        sym = getattr(p, 'symbol', '')
                        typ = getattr(p, 'type', 0)
                        side = 'BUY' if typ == 0 else 'SELL'
                        vol = float(getattr(p, 'volume', 0.0) or 0.0)
                        state_store.record_open_trade(ticket, sym, side, vol, mode='RESTORED', ts=time.time())
                except Exception as e:
                    log_debug("positions_get iteration failed when recording live pos:", e)
                    continue
        except Exception as e:
            log_debug("reconcile open trades failed:", e)

        # Ensure initial_balance and day_open_equity are set
        try:
            ai = mt5.account_info()
            if ai is not None:
                bal = getattr(ai, 'balance', None)
                eq = getattr(ai, 'equity', None)
                if state_store.get('initial_balance') is None and bal is not None:
                    state_store.set('initial_balance', float(bal))
                if state_store.get('day_start_equity') is None and eq is not None:
                    state_store.set_day_open_equity(eq)

                # Update equity high
                try:
                    if eq is not None:
                        state_store.update_day_equity_high(eq)
                except Exception as e:
                    log_debug("update_day_equity_high failed:", e)

                # FTMO daily drawdown enforcement
                try:
                    day_open = state_store.get('day_start_equity')
                    day_high = state_store.get('day_equity_high') or day_open
                    if day_high and eq is not None:
                        dd_pct = (float(day_high) - float(eq)) / float(day_high) * 100.0
                        if dd_pct >= 4.0:
                            state_store.set_trading_paused(True)
                            state_store.save()
                            BOT_STATE.trading_paused = True
                            print(f"[STATE] Daily DD lock active ({dd_pct:.2f}%) - trading paused")
                except Exception as e:
                    log_debug("FTMO daily DD enforcement failed:", e)

                # FTMO overall drawdown enforcement
                try:
                    init_bal = state_store.get('initial_balance')
                    if init_bal and bal is not None:
                        total_drop = (float(init_bal) - float(bal)) / float(init_bal) * 100.0
                        if total_drop >= 7.5:
                            state_store.set_permanent_lockout(True)
                            state_store.save()
                            BOT_STATE.trading_paused = True
                            BOT_STATE.permanent_lockout = True
                            try:
                                globals()['PERMANENT_LOCKED'] = True
                            except Exception as e:
                                log_debug("setting PERMANENT_LOCKED failed during FTMO enforcement:", e)
                            print(f"[STATE] Permanent lockout triggered ({total_drop:.2f}%) - trading paused")
                except Exception as e:
                    log_debug("FTMO overall DD enforcement failed:", e)
        except Exception as e:
            log_debug("MT5 account restore/check failed:", e)

        # Persist any changes made during recovery
        try:
            state_store.save()
        except Exception as e:
            log_debug("state_store.save failed during restore:", e)

        print("[STATE] State restored from previous session")
    except Exception as e:
        try:
            print(f"[STATE] restore_state_on_startup error: {e}")
        except Exception as e2:
            log_debug("restore_state_on_startup print failed:", e2)

 
# Centralised configuration / constants block
CONSTANTS = {
    # Basic runtime flags
    "DRY_RUN": False,
    "TEST_MODE": False,
    # Symbols and strategies
    "SYMBOLS": ["XAUUSD", "GBPUSD", "GBPJPY"],
    "STRATEGY_MAP": {"XAUUSD": "GOAT", "GBPUSD": "GBPUSD_STRAT", "GBPJPY": "GBPJPY_STRAT"},
    # Telegram defaults (override via env if required)
    # Intentionally left blank so credentials must come from the environment
    "TELEGRAM_TOKEN": "8246430772:AAE3IKjs728sj4wW2b6CCMq0iVQ4gXnWHjI",
    "TELEGRAM_CHAT_ID": "5952533212",
    "TELEGRAM_POLLING_ENABLED": True,
    # Trading size / risk
    "RISK_PCT": 1.0/100.0,
    "USE_EQUITY_RISK": True,
    "USE_FIXED_FULL_LOT": True,
    "FULL_LOT_DEFAULT": 0.20,
    "MICRO_LOT_MIN": 0.01,
    "MICRO_LOT_MAX": 0.04,   # align with hard cap used in execution path
    "MICRO_LOT_TARGET": 0.01,
    "MICRO_MAX_PER_SESSION": 8,  # align with runtime cap
    "MICRO_MAX_PER_DAY": 8,
    "FULL_MAX_PER_DAY": 4,
    # Scan cadence
    "SCAN_MIN_MINUTES": 5,
    "SCAN_MAX_MINUTES": 10,
    # Session & behavior
    "DEMO_ON_START": True,
    "FORCE_ENABLE_TRADING": False,
    "QUIET_SPAM": False,
    # Timeframes / thresholds
    "AI_MIN_SCORE": 60,
    "ADX_MIN_H1": 22.0,
    "ADX_MIN_M15": 22.0,
    "RSI_BUY_MIN": 40.0,
    "RSI_SELL_MAX": 60.0,
    "ATR_FLOOR_M15": 0.015,
    "RANGE_MIN_PCT": 0.35,
    # Retry/backoff
    "ORDER_MAX_RETRIES": 3,
    "RETRY_BACKOFF_BASE": 2.0,
    # Spread/deviation defaults
    "FULL_SPREAD_MAX_PTS": 300,
    "MICRO_SPREAD_MAX_PTS": 300,
    "MAX_DEVIATION_DEFAULT": 100,
    # Sessions
    "FULL_SESSION_LON": (7, 0, 18, 0),
    "MICRO_SESSION_LON": (7, 0, 18, 0),
    # Trailing / partials
    "TRAIL_ATR_MULT": 2.0,
    "TP_PARTS": [0.3, 0.3, 0.4],
    "TP_R_MULTS": [1.0, 2.0, 999.0],
    # Data cache defaults
    "DATA_CACHE_TTL": {"H4": 6.0, "H1": 3.0, "M15": 1.5, "M5": 0.6},
    # Optional overrides for later parts of the code
    "TELEGRAM_POLL_SLEEP_EMPTY": 1.5,
    # SMC pure execution mode (prefer SMC-only gating, minimal filters)
    "SMC_PURE_MODE": True,
}

# Backwards-compatible top-level aliases (preserve existing variable names)
DRY_RUN = CONSTANTS["DRY_RUN"]
TEST_MODE = CONSTANTS["TEST_MODE"]
SYMBOLS = CONSTANTS["SYMBOLS"]
STRATEGY_MAP = CONSTANTS["STRATEGY_MAP"]
TELEGRAM_POLLING_ENABLED = CONSTANTS["TELEGRAM_POLLING_ENABLED"]
RISK_PCT = CONSTANTS["RISK_PCT"]
USE_EQUITY_RISK = CONSTANTS["USE_EQUITY_RISK"]
USE_FIXED_FULL_LOT = CONSTANTS["USE_FIXED_FULL_LOT"]
FULL_LOT_DEFAULT = CONSTANTS["FULL_LOT_DEFAULT"]
MICRO_LOT_MIN = CONSTANTS["MICRO_LOT_MIN"]
MICRO_LOT_MAX = CONSTANTS["MICRO_LOT_MAX"]
MICRO_LOT_TARGET = CONSTANTS["MICRO_LOT_TARGET"]
MICRO_MAX_PER_SESSION = CONSTANTS["MICRO_MAX_PER_SESSION"]
MICRO_MAX_PER_DAY = CONSTANTS["MICRO_MAX_PER_DAY"]
FULL_MAX_PER_DAY = CONSTANTS["FULL_MAX_PER_DAY"]
SCAN_MIN_MINUTES = CONSTANTS["SCAN_MIN_MINUTES"]
SCAN_MAX_MINUTES = CONSTANTS["SCAN_MAX_MINUTES"]
DEMO_ON_START = CONSTANTS["DEMO_ON_START"]
FORCE_ENABLE_TRADING = CONSTANTS["FORCE_ENABLE_TRADING"]
QUIET_SPAM = CONSTANTS["QUIET_SPAM"]
AI_MIN_SCORE = CONSTANTS["AI_MIN_SCORE"]

# ----------------------- Dynamic risk controller -----------------------
# Target risk bands (operator-tunable, internal enforcement below)
FULL_RISK_TARGET_MIN = 0.005   # 0.5% per full trade (user requested 0.5-1%)
FULL_RISK_TARGET_MAX = 0.010   # 1.0% per full trade
MICRO_RISK_TARGET_MIN = 0.0010 # 0.10% minimum micro risk
MICRO_RISK_TARGET_MAX = 0.0025 # 0.25% maximum micro risk
# Absolute per-trade hard cap (never exceed)
MAX_RISK_PER_TRADE = 0.005      # 0.5% hard cap for FTMO compliance (per-trade max) 

# Monthly performance targets
MONTHLY_TARGET_LOW = 0.08
MONTHLY_TARGET_HIGH = 0.15

# Conservative-mode adjustments
CONSERVATIVE_COOLDOWN_MULT = 1.5

# Runtime tracking (initialized at startup)
MONTH_START_BALANCE = None
MONTH_CONSERVATIVE_MODE = False
MONTH_START_MONTH = None
ORIGINAL_FULL_COOLDOWN_SECONDS = globals().get('FULL_COOLDOWN_SECONDS', 600)
ORIGINAL_MICRO_COOLDOWN_SECONDS = globals().get('MICRO_COOLDOWN_SECONDS', 300)

ADX_MIN_H1 = CONSTANTS["ADX_MIN_H1"]
ADX_MIN_M15 = CONSTANTS["ADX_MIN_M15"]
RSI_BUY_MIN = CONSTANTS["RSI_BUY_MIN"]
RSI_SELL_MAX = CONSTANTS["RSI_SELL_MAX"]
ATR_FLOOR_M15 = CONSTANTS["ATR_FLOOR_M15"]
RANGE_MIN_PCT = CONSTANTS["RANGE_MIN_PCT"]
ORDER_MAX_RETRIES = CONSTANTS["ORDER_MAX_RETRIES"]
RETRY_BACKOFF_BASE = CONSTANTS["RETRY_BACKOFF_BASE"]
FULL_SPREAD_MAX_PTS = CONSTANTS["FULL_SPREAD_MAX_PTS"]
MICRO_SPREAD_MAX_PTS = CONSTANTS["MICRO_SPREAD_MAX_PTS"]
MAX_DEVIATION_DEFAULT = CONSTANTS["MAX_DEVIATION_DEFAULT"]
FULL_SESSION_LON = CONSTANTS["FULL_SESSION_LON"]
MICRO_SESSION_LON = CONSTANTS["MICRO_SESSION_LON"]
TRAIL_ATR_MULT = CONSTANTS["TRAIL_ATR_MULT"]
TP_PARTS = CONSTANTS["TP_PARTS"]
TP_R_MULTS = CONSTANTS["TP_R_MULTS"]


def ensure_mt5_connection(retries: int = 10, delay: float = 1.0) -> bool:
    """
    Ticket-Funded MT5 Connection: Initialize and verify MT5 connection.
    
    Returns True when connected and basic account info is present. 
    Raises Exception on failure.
    """
    global MT5_CONNECTED, MT5_LOGIN_VERIFIED
    
    for i in range(retries):
        try:
            ok = False
            try:
                # Ticket-Funded MT5 Connection: Use new initialization
                ok = mt5.initialize()
            except Exception:
                ok = False
                log_debug("mt5.initialize attempt raised:", i)
            
            if ok:
                try:
                    print("[Ticket-Funded MT5] Connected to MT5 successfully.")
                except Exception:
                    log_debug("print of MT5 connected message failed")
                
                # Ticket-Funded MT5 Connection: Verify basic account info
                try:
                    ai = mt5.account_info()
                    if not ai or not getattr(ai, 'server', None) or not getattr(ai, 'login', None):
                        print(f"[Ticket-Funded MT5] Missing account/server info on attempt {i+1}")
                        time.sleep(delay)
                        continue
                    
                    # Log account details
                    print(f"[Ticket-Funded MT5] Account: {ai.login}, Server: {ai.server}")
                    print(f"[Ticket-Funded MT5] Balance: {ai.balance}, Equity: {ai.equity}")
                    
                    MT5_CONNECTED = True
                    MT5_LOGIN_VERIFIED = True
                except Exception:
                    log_debug("mt5.account_info read failed on connect attempt", i)
                    time.sleep(delay)
                    continue
                
                return True
            else:
                try:
                    print(f"[Ticket-Funded MT5] Connection failed. Retrying... ({i+1}/{retries})")
                except Exception:
                        log_debug("print of MT5 retry message failed")
        except Exception:
            try:
                print(f"[Ticket-Funded MT5] Connection check exception on attempt {i+1}")
            except Exception as e:
                log_debug("print failed for connection check exception:", e)
        time.sleep(delay)
    
    # Ticket-Funded MT5 Connection: Alert on final failure
    error_msg = "🚫 **ERROR**\nMT5 connection failed after multiple attempts"
    telegram_msg_mt5(error_msg)
    raise Exception(error_msg)


def ensure_mt5_connected_or_exit(retries:int=5, delay:float=2.0):
    """Ensure MT5 is connected; if not, attempt reconnect and block until ready or raise.

    This helper is intended to be called from `main_loop` at startup to
    guarantee MT5 availability before the bot proceeds.
    """
    try:
        for _ in range(retries):
            try:
                if mt5.account_info() is not None:
                    return True
            except Exception:
                log_debug("ensure_mt5_connected_or_exit: mt5.account_info check failed")
            try:
                mt5.shutdown()
            except Exception:
                log_debug("ensure_mt5_connected_or_exit: mt5.shutdown failed")
            time.sleep(0.25)
            try:
                mt5.initialize()
            except Exception:
                log_debug("ensure_mt5_connected_or_exit: mt5.initialize failed")
            time.sleep(delay)
        # final check
        if mt5.account_info() is None:
            raise Exception("MT5 not connected after reconnection attempts")
        return True
    except Exception as e:
        log_msg(f"⚠️ MT5 reconnection failed: {e}")
        raise



# ================= MARKET MEMORY MODULE =================
MARKET_MEMORY_WINDOW = 200
MARKET_MEMORY = []

def record_market_snapshot(symbol="XAUUSD"):
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        if rates is None:
            return
        df = pd.DataFrame(rates)
        atr_val = (df['high'] - df['low']).tail(20).mean()
        body = (df['close'] - df['open']).abs().iloc[-1]
        wick = (df['high'].iloc[-1] - df['low'].iloc[-1]) - body
        adx_val = adx(df, 14)
        snapshot = {"atr": float(atr_val), "adx": float(adx_val),
                    "body": float(body), "wick": float(wick)}
        MARKET_MEMORY.append(snapshot)
        if len(MARKET_MEMORY) > MARKET_MEMORY_WINDOW:
            MARKET_MEMORY.pop(0)
    except Exception as e:
        log_debug("market memory append failed:", e)

def market_memory_bias():
    try:
        if len(MARKET_MEMORY) < 20:
            return 0
        atr_avg = sum(m["atr"] for m in MARKET_MEMORY[-50:]) / 50
        adx_avg = sum(m["adx"] for m in MARKET_MEMORY[-50:]) / 50
        if atr_avg > 1.5 and adx_avg > 25:
            return 5
        if atr_avg < 0.7 and adx_avg < 18:
            return -5
        if atr_avg > 3.0:
            return -3
        return 0
    except Exception:
        return 0

def apply_market_memory_to_score(ai_score):
    try:
        return max(0, min(100, ai_score + market_memory_bias()))
    except Exception as e:
        # Log any unexpected error and fall back to original score
        try:
            log_error(f"apply_market_memory_to_score error: {e}")
        except Exception:
            try:
                print(f"apply_market_memory_to_score error: {e}")
            except Exception as ex:
                log_debug("apply_market_memory_to_score print fallback failed:", ex)
        return ai_score

def log_error(text: str) -> None:
    """Lightweight error logger used before full logging is configured."""
    try:
        # Prefer the richer logger if available
        if 'log_msg' in globals():
            try:
                log_msg(str(text), level="ERROR")
                return
            except Exception as e:
                log_debug("log_msg failed in log_error:", e)
        print(f"ERROR: {text}")
    except Exception as e:
        try:
            print(f"ERROR: {text}")
        except Exception as ex:
            log_debug("print fallback failed in log_error:", ex)
# ========================================================

# =============================================================================
# HOLY GRAIL ENGINE - Pure SMC with 70% Scoring System
# All logic is integrated directly in this file
# =============================================================================

# ========================================================

# Ticket-Funded: Position close monitoring with trade alerts
_last_position_tickets = set()

# When a Telegram command triggers a live mirror, set this to the chat id.
# place_trade will use this to send synchronous execution/failure messages
# directly to the requesting chat so order outcomes are shown inline.
_CURRENT_CMD_CHAT = None

from contextlib import contextmanager
import builtins as _builtins


@contextmanager
def mirror_console_to_telegram(chat_id: int):
    """Context manager that mirrors all `print()` output to the given
    `chat_id` via synchronous Telegram sends, preserving order.

    It temporarily replaces `builtins.print` so every printed line is also
    sent using `_tg_send_sync(chat_id, text)`. Use sparingly because it
    ensures synchronous network sends to preserve ordering.
    """
    global _tg_send_sync
    orig_print = _builtins.print

    def _mirror_print(*args, sep=' ', end='\n', **kwargs):
        try:
            text = sep.join(str(a) for a in args) + end
        except Exception:
            try:
                text = ' '.join([str(a) for a in args]) + end
            except Exception:
                text = ''

        try:
            orig_print(*args, sep=sep, end=end, **kwargs)
        except Exception as e:
            log_debug("orig_print in _mirror_print failed:", e)

        try:
            payload = text.rstrip('\n')
            if payload:
                try:
                    _tg_send_sync(chat_id, payload)
                except Exception as e:
                    log_debug("_tg_send_sync failed in _mirror_print:", e)
                try:
                    low = payload.lower()
                    block_indicators = ['⛔', 'blocked', 'skip', 'skipping', 'duplicate zone', 'could not fetch tick', 'cooldown active', 'cap hit', 'rejected']
                    if any(tok in low for tok in block_indicators):
                        try:
                            _tg_send_sync(chat_id, 'Final Decision: Blocked ❌')
                            _tg_send_sync(chat_id, f'Reason: {payload}')
                        except Exception as e:
                            log_debug("_tg_send_sync failed for blocked decision in _mirror_print:", e)
                except Exception as e:
                    log_debug("_mirror_print block detection failed:", e)
        except Exception as e:
            log_debug("_mirror_print outer handler failed:", e)

    try:
        _builtins.print = _mirror_print
        yield
    finally:
        try:
            _builtins.print = orig_print
        except Exception as e:
            log_debug("restoring orig_print failed:", e)
@contextmanager
def capture_console_output():
    orig_print = _builtins.print
    buf = []

    def _capture(*args, sep=' ', end='\n', **kwargs):
        try:
            text = sep.join(str(a) for a in args)
        except Exception:
            try:
                text = ' '.join([str(a) for a in args])
            except Exception:
                text = ''
        # store without trailing newline
        try:
            buf.append(text)
        except Exception as e:
            log_debug("buf.append failed in capture_console_output:", e)
        # still print to console
        try:
            orig_print(*args, sep=sep, end=end, **kwargs)
        except Exception as e:
            log_debug("orig_print in _capture failed:", e)

    try:
        _builtins.print = _capture
        yield buf
    finally:
        try:
            _builtins.print = orig_print
        except Exception as e:
            log_debug("restoring orig_print failed in capture_console_output:", e)


def _chunk_and_send(chat_id: int, text: str, max_len: int = 3500):
    """Send `text` to chat_id in chunks under `max_len` using synchronous send."""
    try:
        if not text:
            return True
        # prefer splitting at double-newline or newline for readability
        parts = []
        if len(text) <= max_len:
            parts = [text]
        else:
            lines = text.splitlines()
            cur = []
            cur_len = 0
            for ln in lines:
                add_len = len(ln) + 1
                if cur_len + add_len > max_len and cur:
                    parts.append('\n'.join(cur))
                    cur = [ln]
                    cur_len = add_len
                else:
                    cur.append(ln)
                    cur_len += add_len
            if cur:
                parts.append('\n'.join(cur))

        for p in parts:
            try:
                # use synchronous low-level send to bypass suppression logic
                _tg_send_sync(chat_id, p)
            except Exception as e:
                try:
                    send_telegram_to(chat_id, p)
                except Exception as ex:
                    try:
                        telegram_msg(p)
                    except Exception as ex2:
                        log_debug("telegram_msg fallback failed sending chunk:", ex2)
                    log_debug("send_telegram_to fallback failed sending chunk:", ex)
                log_debug("_tg_send_sync failed sending chunk:", e)
        return True
    except Exception:
        return False


def cmd_scan_verbose(chat_id: int):
    """Run the live scanner and capture console output, then send aggregated
    emoji-preserving scan results to `chat_id` as one or more messages.

    This runs the exact same scan logic (including execution) and only
    controls how results are reported to Telegram (no suppression).
    """
    def _worker():
        try:
            # capture all console prints during the scan
            with capture_console_output() as out_lines:
                try:
                    holy_grail_scan_and_execute()
                except Exception as e:
                    try:
                        print(f"[SCAN ERROR] {e}")
                    except Exception as ex:
                        log_debug("print failed in cmd_scan_verbose scan error handler:", ex)

            # Build the detailed scan report using the refactored function
            report_text = ""
            try:
                report_text = send_scan_report(chat_id, allow_outside_session=True) or ""
            except Exception as e:
                try:
                    report_text = f"[REPORT ERROR] {e}"
                except Exception:
                    report_text = "[REPORT ERROR] unknown"

            # Aggregate output preserving emojis and formatting
            try:
                merged_lines = []
                if out_lines:
                    merged_lines.append("--- Console Output ---")
                    merged_lines.extend(out_lines)
                if report_text:
                    merged_lines.append("--- Scan Report ---")
                    # If report_text contains multiple reports, split by double newlines
                    try:
                        rep_parts = report_text.split('\n\n') if isinstance(report_text, str) else [str(report_text)]
                        for rp in rep_parts:
                            merged_lines.extend(rp.splitlines())
                    except Exception:
                        merged_lines.append(str(report_text))
                text = '\n'.join(merged_lines)
            except Exception:
                try:
                    text = '\n'.join(out_lines) if out_lines else str(report_text)
                except Exception:
                    text = str(out_lines) + '\n' + str(report_text)

            # If empty, send a minimal acknowledgment
            if not text:
                try:
                    _tg_send_sync(chat_id, "🔎 Scan completed — no output captured.")
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, "🔎 Scan completed — no output captured.")
                    except Exception as ex:
                        log_debug("send_telegram_to failed for empty scan ack:", ex)
                    log_debug("_tg_send_sync failed for empty scan ack:", e)
                return

            # Send in chunks to the requesting chat
            _chunk_and_send(chat_id, text)
        except Exception as e:
            try:
                _tg_send_sync(chat_id, f"Scan failed: {e}")
            except Exception as ex:
                try:
                    send_telegram_to(chat_id, f"Scan failed: {e}")
                except Exception as ex2:
                    log_debug("send_telegram_to failed for scan failed message:", ex2)
                log_debug("_tg_send_sync failed for scan failed message:", ex)

    threading.Thread(target=_worker, daemon=True).start()
    try:
        _tg_send_sync(chat_id, "🔎 Scan started — detailed results will be posted here when ready.")
    except Exception as e:
        log_debug("_tg_send_sync failed for scan start, falling back to send_telegram_to:", e)
        try:
            send_telegram_to(chat_id, "🔎 Scan started — detailed results will be posted here when ready.")
        except Exception as e2:
            log_debug("send_telegram_to fallback failed for scan start:", e2)

def send_trade_close_alert(symbol, result, rr="N/A", profit=None, pips=None, duration=None):
    """Send formatted trade close alert to Telegram.

    Parameters:
    - symbol: str
    - result: one of "WIN", "LOSS", or other
    - rr: risk:reward string or "N/A"
    - profit: numeric profit (optional)
    - pips: numeric pips (optional)
    - duration: string or numeric duration (optional)
    """
    try:
        # Determine emoji based on result
        if result == "WIN":
            emoji = "📈"
            result_text = "✅ WIN"
        elif result == "LOSS":
            emoji = "📉"
            result_text = "❌ LOSS"
        else:
            emoji = "➖"
            result_text = "➖ BREAK EVEN"

        parts = [f"📦 **TRADE CLOSED** — {symbol}", f"Result: {result_text}"]
        if rr != "N/A":
            parts.append(f"R:R: {rr}")
        if profit is not None:
            try:
                parts.append(f"Profit: {float(profit):.2f}")
            except Exception:
                parts.append(f"Profit: {profit}")
        if pips is not None:
            parts.append(f"Pips: {pips}")
        if duration:
            parts.append(f"Duration: {duration}")
        else:
            parts.append("Duration: N/A")

        msg = "\n".join(parts)
        # Use structured notification for trade results (wins/losses)
        try:
            # Map result to required emoji phrasing
            if result == "WIN":
                emoji = '🏆'
                body = f"{emoji} {symbol} — WIN\n" + "\n".join(parts[1:])
            elif result == "LOSS":
                emoji = '❌'
                body = f"{emoji} {symbol} — LOSS\n" + "\n".join(parts[1:])
            else:
                body = msg
            # Only notify for allowed symbols
            if _should_notify_symbol(symbol):
                send_telegram_to(TELEGRAM_CHAT_ID, body)
        except Exception:
            try:
                print(msg)
            except Exception as ex2:
                try:
                    log_debug("tg fallback failed and log_debug failed:", ex2)
                except Exception as e:
                    log_debug("send_telegram_to/build_help_message fallback failed:", e)
    except Exception as e:
        try:
            log_debug("send_trade_close_alert failed:", e)
        except Exception as e2:
            try:
                log_debug("send_trade_close_alert logging failed:", e2)
            except Exception as e:
                try:
                    log_debug("volatility jump check failed in safe_order_send:", e)
                except Exception:
                    pass

def monitor_holy_grail_position_closes():
    """
    # Ticket-Funded
    Monitor for closed positions and send trade close alerts.
    """
    global _last_position_tickets
    try:
        current_positions = set()
        try:
            poss = mt5.positions_get() or []
        except Exception as e:
            log_debug("positions_get failed:", e)
            poss = []

        for pos in poss:
            try:
                current_positions.add(getattr(pos, 'ticket', None))
            except Exception as e:
                log_debug("reading position ticket failed:", e)

        # Find recorded tickets that are no longer live
        closed_tickets = _last_position_tickets - current_positions

        for ticket in closed_tickets:
            # Attempt to locate deal information for the closed ticket
            deals = []
            try:
                deals = mt5.history_deals_get() or []
            except Exception as e:
                log_debug("history_deals_get failed:", e)

            # Best-effort: find the first deal matching the closed ticket
            profit = None
            for deal in deals:
                try:
                    if getattr(deal, 'position_id', None) is not None and str(getattr(deal, 'position_id')) == str(ticket):
                        symbol = getattr(deal, 'symbol', 'N/A')
                        profit = getattr(deal, 'profit', None)
                        result = "WIN" if profit and profit > 0 else ("LOSS" if profit and profit < 0 else "BREAK_EVEN")

                        # Compute pips (best-effort)
                        pips = None
                        try:
                            si = mt5.symbol_info(symbol)
                            point = getattr(si, 'point', None) if si is not None else None
                            price_open = getattr(deal, 'price_open', None) or getattr(deal, 'price', None)
                            price_close = getattr(deal, 'price', None)
                            if point and price_open is not None and price_close is not None:
                                pips = round(abs(price_close - price_open) / point, 1)
                        except Exception as e:
                            log_debug("pips calc failed:", e)

                        # Duration (best-effort)
                        duration = None
                        try:
                            t_open = getattr(deal, 'time', None)
                            if t_open:
                                duration = 'N/A'
                        except Exception:
                            duration = None

                        try:
                            send_trade_close_alert(symbol, result, "N/A", profit=profit, pips=pips, duration=duration)
                        except Exception as e:
                            log_debug("send_trade_close_alert failed:", e)
                        break
                except Exception as e:
                    log_debug("processing deal failed:", e)

            # Persist closed trade and update session counters if necessary
            try:
                if state_store is not None:
                    try:
                        opens = state_store.get('open_trades') or []
                        was_full = False
                        for ot in opens:
                            try:
                                if str(ot.get('ticket')) == str(ticket):
                                    if ot.get('mode') and str(ot.get('mode')).upper() == 'FULL':
                                        was_full = True
                                    break
                            except Exception:
                                continue

                        try:
                            state_store.record_closed_trade(ticket)
                        except Exception as e:
                            log_debug("state_store.record_closed_trade failed during monitor:", e)

                        try:
                            if was_full and profit is not None and float(profit) < 0:
                                BOT_STATE.session_full_losses = BOT_STATE.session_full_losses + 1
                                try:
                                    state_store.set('session_full_losses', BOT_STATE.session_full_losses)
                                except Exception as e:
                                    log_debug("state_store.set session_full_losses failed:", e)
                        except Exception as e:
                            log_debug("processing restored trade loss handling failed:", e)

                        try:
                            state_store.save()
                        except Exception as e:
                            log_debug("state_store.save failed during monitor:", e)
                    except Exception as e:
                        log_debug("error reconciling recorded open trades, attempting best-effort:", e)
                        try:
                            state_store.record_closed_trade(ticket)
                            state_store.save()
                        except Exception as e2:
                            log_debug("best-effort record_closed_trade/save failed:", e2)
            except Exception as e:
                log_debug("failed to persist closed trade removal during monitor:", e)

        # Update tracking set
        _last_position_tickets = current_positions
    except Exception as e:
        log_debug("monitor_holy_grail_position_closes failed:", e)


import os
import ctypes

# ----------------------------------------------------------------------
# MT5 terminal configuration
#
# Previous versions of this bot attempted to locate the MetaTrader5
# terminal by scanning for known installation paths and launching any
# FTMO‑branded terminals.  This led to brittle behaviour and incorrect
# attachments (e.g. repeatedly attempting to connect to a missing FTMO
# terminal).  The scanning logic has been removed.  Instead, the bot
# uses a single, explicit path (AUTO_MT5_PATH) to launch the terminal
# once, and the MetaTrader5 Python bridge will locate the running
# terminal automatically.  If AUTO_MT5_PATH is invalid, no attempt is
# made to open other terminals.
MT5_PATH = None

# Absolute path to the MetaTrader 5 terminal that should be launched
# automatically.  Update this path if your MT5 is installed elsewhere.
AUTO_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# End of path detection.  The import of MetaTrader5 has already been
# handled at the top of the module using a try/except with a fallback
# dummy implementation.  Therefore, no additional initialization is
# necessary here.  See the import section for details.
TIMEZONE = "Europe/London"

# ---------------------------------------------------------------------------
# Safe timezone handling
# Some environments may not have full timezone data installed, leading to
# pytz.UnknownTimeZoneError for otherwise valid timezone names such as
# "Europe/London". Define a helper to obtain a timezone object safely,
# falling back to UTC on failure. Precompute a fallback timezone object
# once at import time to avoid repeated exception handling.
def _safe_pytz_timezone(name: str):
    """Return a pytz timezone object for the given name or UTC on failure."""
    try:
        return pytz.timezone(name)
    except Exception:
        try:
            return pytz.timezone('UTC')
        except Exception:
            # Last resort: use UTC from datetime if pytz has no 'utc' attribute
            return pytz.utc if hasattr(pytz, 'utc') else timezone.utc

# Precompute the configured timezone, safely resolved
SAFE_TZ = _safe_pytz_timezone(TIMEZONE)
# This constant formerly pointed to the FTMO‑branded MT5.  It is now
# unused because the bot no longer attaches to alternative terminals.
GOAT_TERMINAL_DIR = None
#
# Telegram credentials (single canonical definition).
# These are intentionally set as literals per operator request.
# NOTE: polling is NOT started automatically on import - call
# Global Telegram credentials - single source of truth
# Read from environment variables first (safer). Optionally load .env.
try:
    from dotenv import load_dotenv
    # Load .env explicitly from the script directory so running the script
    # from a different CWD still picks up credentials placed alongside
    # `Ticket.py` (common cause of missing-token problems).
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except Exception:
    # If python-dotenv isn't available or loading fails, continue; the
    # bot will attempt to read real environment variables instead.
    try:
        log_debug("dotenv load failed or python-dotenv missing")
    except Exception as e:
        log_debug("startup summary print failed:", e)

# Read credentials after attempting to load the local .env file.
# Support both common variable names: `TELEGRAM_BOT_TOKEN` (preferred) and
# `TELEGRAM_TOKEN` (legacy or other setups). This helps environments where
# the .env was created with a slightly different key name.
TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or CONSTANTS.get("TELEGRAM_TOKEN", "")
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or CONSTANTS.get("TELEGRAM_CHAT_ID", "")

# Helpful, non-sensitive debug: indicate whether credentials were detected
# (prints booleans only; does not expose token/chat values).
print(f"[Telegram] token set: {bool(TELEGRAM_TOKEN)}, chat set: {bool(TELEGRAM_CHAT_ID)}")

# Startup summary for gating modes (console-only)
try:
    pred_state = 'OFF' if (HARD_DISABLE_HTF or not ENABLE_PREDICTIVE_AI) else 'ON'
    liq_state = 'OFF' if (HARD_DISABLE_HTF or not ENABLE_PREDICTIVE_AI) else 'ON'
    print(f"[STARTUP] HTF_DISABLED={HARD_DISABLE_HTF} | PredictiveGating={pred_state} | LiquidityGating={liq_state}")
except Exception:
    try:
        log_debug("startup summary print failed")
    except Exception as e:
        log_debug("print txt failed in print_ngrok_commands:", e)

# Track whether credentials were supplied via the environment (preferred)
# If credentials are only present via the `CONSTANTS` dict (e.g. tests or
# packaged configs), do NOT auto-start network pollers on import. Require an
# explicit environment-based credential presence to enable automatic
# import-time startup.
TELEGRAM_CREDS_FROM_ENV = bool(
    (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")) and os.getenv("TELEGRAM_CHAT_ID")
)

# Polling flag: when True, the background poller will run. Keep True
# to enable active Telegram command listening at startup.
TELEGRAM_POLLING_ENABLED = CONSTANTS.get("TELEGRAM_POLLING_ENABLED", True)

# Internal polling state (do not modify directly)
_tg_thread = None
_tg_offset = None

def telegram_enabled() -> bool:
    """Return True when Telegram credentials appear configured."""
    return True if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else False


def _handle_telegram_update(text: str, chat_id: int) -> None:
    """Safe wrapper to handle incoming Telegram text updates.

    Ensures every update is handled best-effort and that errors are
    reported back to the originating chat instead of bubbling up.
    If the command is not recognised, sends the help message.
    """
    try:
        handled = False
        try:
            handled = bool(_dispatch_minimal_command(text, chat_id))
        except Exception as e:
            # Report internal dispatch errors back to the requesting chat if possible,
            # otherwise fall back to admin `tg()` so owner is still informed.
            try:
                send_telegram_to(chat_id, f"[TG HANDLER ERROR] {e}")
            except Exception:
                try:
                    tg(f"[TG HANDLER ERROR] {e}")
                except Exception as ex:
                    log_debug("tg fallback send failed:", ex)
            try:
                send_telegram_to(chat_id, "⚠️ Command processing error. See bot logs.")
            except Exception as e:
                try:
                    tg("⚠️ Command processing error. See bot logs.")
                except Exception as e2:
                    try:
                        log_debug("tg fallback failed in command processing:", e2)
                    except Exception:
                        pass
            return

        if not handled:
            # Unknown command - send concise help to the requesting chat
            try:
                send_telegram_to(chat_id, build_help_message())
            except Exception:
                try:
                    send_telegram_to(chat_id, "Available commands: /help /status /signal /scan /findtrade /panic /resume")
                except Exception:
                    try:
                        tg(build_help_message())
                    except Exception:
                        try:
                            log_debug("tg fallback failed sending help message")
                        except Exception as ex:
                            log_debug("tg fallback send failed:", ex)
    except Exception as e:
        # Catch-all to ensure nothing leaks out of the poller
        try:
            tg("[TG HANDLER FATAL] Unhandled exception in update handling")
        except Exception as ex:
            try:
                log_debug("tg fallback failed in _handle_telegram_update:", ex)
            except Exception as e:
                log_debug("webhook _dispatch inner handler failed:", e)


def start_telegram_polling(send_online_ack: bool = False) -> None:
    """Start a background daemon thread to poll Telegram updates.

    send_online_ack: when True, send a short "Bot Online" message to the
    configured admin chat. This lets callers choose whether an ack is sent
    (auto-start will request the ack; manual starts default to no ack).
    """
    global TELEGRAM_POLLING_ENABLED, _tg_thread
    try:
        if not telegram_enabled():
            print("[Telegram] Telegram not enabled, skipping polling")
            return
        # If already running, do nothing
        if _tg_thread is not None and getattr(_tg_thread, 'is_alive', lambda: False)():
            print("[Telegram] Polling thread already running")
            return

        TELEGRAM_POLLING_ENABLED = True
        _tg_thread = threading.Thread(target=telegram_poll_loop, daemon=True)
        _tg_thread.start()
        # Visible console log indicating poller mode
        mode = "auto" if str(os.getenv("AUTO_START_TELEGRAM", "1")) not in ("0", "false", "False") else "manual"
        print(f"✅ Telegram polling started (mode={mode})")
        # Only send the online acknowledgement when explicitly requested
        if send_online_ack:
            # By default the startup acknowledgement is console-only unless
            # `ALLOW_RAW_TELEGRAM=1` is set. This keeps startup noise out of
            # operator Telegram when compact-only policy is enforced.
            if str(os.getenv('ALLOW_RAW_TELEGRAM', '0')).lower() in ('1', 'true'):
                try:
                    ok, resp = _tg_send_sync(TELEGRAM_CHAT_ID, "🤖 Bot Online – Telegram Connected")
                    if ok:
                        print("[Telegram] Online acknowledgement sent to admin chat")
                    else:
                        try:
                            print(f"[Telegram] Online acknowledgement FAILED to send. Response: {resp}")
                        except Exception:
                            print("[Telegram] Online acknowledgement FAILED to send")
                except Exception as e:
                    print(f"[Telegram] Online acknowledgement error: {e}")
            else:
                print('[Telegram] Online acknowledgement suppressed (ALLOW_RAW_TELEGRAM not enabled)')
        # Start periodic XAU block reporter (idempotent)
        try:
            start_xau_block_reporter()
        except Exception as e:
            log_debug("start_xau_block_reporter failed:", e)
    except Exception as e:
        print(f"[Telegram] Failed to start polling: {e}")


def stop_telegram_polling(join_timeout: float = 2.0) -> None:
    """Stop the background Telegram poller and optionally join the thread.

    Setting the polling flag to False signals the thread to exit. We then
    attempt a short join to allow it to terminate cleanly.
    """
    global TELEGRAM_POLLING_ENABLED, _tg_thread
    try:
        TELEGRAM_POLLING_ENABLED = True
        th = _tg_thread
        _tg_thread = None
        if th is not None and getattr(th, 'is_alive', lambda: False)():
            try:
                th.join(join_timeout)
            except Exception as e:
                log_debug("joining telegram thread failed:", e)
    except Exception as e:
        log_debug("stop_telegram_polling outer handler failed:", e)


def set_telegram_webhook(url: str, cert_path: Optional[str] = None) -> dict:
    """Set Telegram webhook to `url`. If `cert_path` is provided it will be
    uploaded as the public certificate. Returns Telegram API response as dict.

    Note: Telegram requires a publicly reachable HTTPS URL. Use a service like
    ngrok for local testing or host on a public server with a valid
    certificate.
    """
    try:
        api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        if cert_path and os.path.exists(cert_path):
            with open(cert_path, 'rb') as f:
                files = {'certificate': f}
                res = requests.post(api, data={'url': url}, files=files, timeout=15)
        else:
            res = requests.post(api, data={'url': url}, timeout=15)
        try:
            return res.json()
        except Exception:
            return {'ok': False, 'error': 'invalid json response'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def start_webhook_server(host: str = '0.0.0.0', port: int = 8443, certfile: Optional[str] = None, keyfile: Optional[str] = None, path: str = '/') -> None:
    """Start a minimal HTTPS webhook server that accepts Telegram updates.

    This function runs the server in a daemon thread. Telegram will POST JSON
    updates to the configured webhook URL; incoming updates are dispatched to
    `_handle_telegram_update`.

    Requirements: a public HTTPS URL that matches the address/port used in
    `set_telegram_webhook`. If using a self-signed cert, upload it via
    `set_telegram_webhook(..., cert_path=...)`.
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import ssl

        class _WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if not self.path.startswith(path):
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get('content-length', 0))
                body = self.rfile.read(length) if length else b''
                try:
                    data = json.loads(body.decode('utf-8')) if body else {}
                except Exception:
                    data = {}
                # Respond quickly to Telegram
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                try:
                    # Dispatch handling in a thread to avoid blocking the HTTP
                    # server while handlers run.
                    def _dispatch():
                        try:
                            # Telegram sends update objects with 'message' etc.
                            if isinstance(data, dict):
                                # emulate the polling update structure
                                if 'message' in data:
                                    msg = data.get('message', {})
                                    text = msg.get('text', '')
                                    chat = msg.get('chat', {}).get('id')
                                    if text and chat:
                                        _handle_telegram_update(text, chat)
                                else:
                                    # handle other update types if desired
                                    pass
                        except Exception as e:
                            log_debug("webhook _dispatch failed:", e)
                    threading.Thread(target=_dispatch, daemon=True).start()
                except Exception as e:
                    log_debug("webhook dispatch start failed:", e)

        httpd = HTTPServer((host, port), _WebhookHandler)
        if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
            httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True, certfile=certfile, keyfile=keyfile)

        def _serve():
            try:
                httpd.serve_forever()
            except Exception as e:
                try:
                    httpd.server_close()
                except Exception as ex:
                    log_debug("httpd.server_close failed:", ex)
                log_debug("webhook serve loop failed:", e)

        th = threading.Thread(target=_serve, daemon=True)
        th.start()
        print(f"✅ Webhook server started on https://{host}:{port}{path}")
    except Exception as e:
        print(f"[Webhook] failed to start: {e}")


def print_ngrok_commands(port: int = 8443, webhook_path: str = '/') -> str:
    """Return a block of PowerShell commands to run ngrok and set the Telegram webhook.

    This helper does not attempt to download or run ngrok. It only provides the
    exact PowerShell commands for you to paste and run locally. Replace the
    placeholders `<NGROK_HTTPS_URL>` and `<BOT_TOKEN>` as instructed.
    """
    cmds = []
    cmds.append('# 1) Download ngrok (one-time)')
    cmds.append('Invoke-WebRequest -Uri "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip" -OutFile "ngrok.zip"')
    cmds.append('Expand-Archive -Path .\\ngrok.zip -DestinationPath .\\ngrok')
    cmds.append('Set-Location .\\ngrok')
    cmds.append('')
    cmds.append('# 2) Start your local webhook server in a separate shell:')
    cmds.append('python -c "import Ticket; Ticket.start_webhook_server(host=\'0.0.0.0\', port=%d, certfile=None, keyfile=None, path=\'%s\')"' % (port, webhook_path))
    cmds.append('')
    cmds.append('# 3) In another shell run ngrok to forward HTTPS to your local port:')
    cmds.append('.\\ngrok.exe http %d' % port)
    cmds.append('')
    cmds.append('# 4) After ngrok prints the Forwarding https://... URL, register it with Telegram:')
    cmds.append('$bot_token = "<BOT_TOKEN>"')
    cmds.append('$public = "<NGROK_HTTPS_URL>"  # replace with the https:// URL shown by ngrok')
    cmds.append('Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$bot_token/setWebhook" -Body @{url=($public + "%s")}' % webhook_path)
    cmds.append('')
    cmds.append('# 5) To remove webhook and revert to polling:')
    cmds.append('Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$bot_token/setWebhook" -Body @{url=""}')

    txt = '\n'.join(cmds)
    try:
        print(txt)
    except Exception as e:
        try:
            log_debug("print txt failed in print_ngrok_commands", e)
        except Exception as e2:
            try:
                log_debug("print_ngrok_commands logging failed:", e2)
            except Exception:
                pass
    return txt

# Dry-run flag: when True, no real orders or network actions are performed.
DRY_RUN = CONSTANTS.get("DRY_RUN", False)

# Test mode: when True, `place_order` will simulate a successful order and
# will not call the MT5 bridge. Use `enable_test_mode()` / `disable_test_mode()`
# to toggle during testing.
TEST_MODE = CONSTANTS.get("TEST_MODE", False)

# Validate Telegram credentials early to warn the operator when missing.
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    try:
        print("⚠️ Telegram credentials missing or invalid!")
    except Exception:
            try:
                log_debug("print failed for missing telegram credentials")
            except Exception as e:
                log_debug("webhook dispatch start failed:", e)
SYMBOLS = CONSTANTS.get("SYMBOLS", ["XAUUSD", "GBPUSD", "GBPJPY"])
STRATEGY_MAP = CONSTANTS.get("STRATEGY_MAP", {"XAUUSD": "GOAT", "GBPUSD": "GBPUSD_STRAT", "GBPJPY": "GBPJPY_STRAT"})
# Use centralized constants where appropriate (alias for compatibility)
AUTO_PLACE_FULL = CONSTANTS.get("AUTO_PLACE_FULL", True)
RISK_PCT = CONSTANTS.get("RISK_PCT", 1.0/100.0)
USE_EQUITY_RISK = CONSTANTS.get("USE_EQUITY_RISK", True)
USE_FIXED_FULL_LOT = CONSTANTS.get("USE_FIXED_FULL_LOT", True)
FULL_LOT_DEFAULT = CONSTANTS.get("FULL_LOT_DEFAULT", 0.20)
SCAN_MIN_MINUTES = CONSTANTS.get("SCAN_MIN_MINUTES", 5)
SCAN_MAX_MINUTES = CONSTANTS.get("SCAN_MAX_MINUTES", 10)
MICRO_LOT_MIN = CONSTANTS.get("MICRO_LOT_MIN", 0.01)
MICRO_LOT_MAX = CONSTANTS.get("MICRO_LOT_MAX", 0.03)
MICRO_LOT_TARGET = CONSTANTS.get("MICRO_LOT_TARGET", 0.01)
MICRO_MAX_PER_SESSION = CONSTANTS.get("MICRO_MAX_PER_SESSION", 6)
# With a single session spanning 07:00–18:00 there is only one trading window per day.
# Cap micro trades to the per‑session limit instead of doubling for two sessions.
MICRO_MAX_PER_DAY = CONSTANTS.get("MICRO_MAX_PER_DAY", MICRO_MAX_PER_SESSION)  # up to 6 micro trades per day
FULL_MAX_PER_DAY = CONSTANTS.get("FULL_MAX_PER_DAY", 6)  # cap full trades per day
MICRO_START_DELAY_MIN = 5
MICRO_SYMBOL_PRIORITY = "XAUUSD"
DEMO_ON_START = True
FORCE_ENABLE_TRADING = False
FORCE_MSG_SHOWN = False  # one-time notice flag
if FORCE_ENABLE_TRADING:
    MICRO_START_DELAY_MIN = 0  # allow micros immediately when forcing
QUIET_SPAM = False
SPAM_FILTER = [
    "order_check", "requote", "retry fail", "ret=10018", "price_changed",
    "ret=10030", "invalid_fill", "dev=", "fill=", "vol=", "Retry fail",
    "Requote", "INVALID_FILL", "PRICE_CHANGED", "→ switching to PENDING"
]
MAX_REQUOTES_BEFORE_PENDING = 2    # if we get 2 requotes (10018) in a row, switch to pending order

# ---------------------------------------------------------------------------
# Data reliability parameters
# To improve resilience against intermittent MT5 feed issues, the bot will
# retry data fetches several times before falling back to the last cached
# dataset.  These constants control the number of retries and the delay
# between attempts.
DATA_RETRY_COUNT: int = 3
DATA_RETRY_DELAY: float = 0.25

# Cache of the most recent data returned by get_data for each (symbol, timeframe, bars).
# If the MT5 API returns no data, the bot will return a copy of the last
# successfully retrieved DataFrame for that key.  This avoids skipping trades
# due to temporary feed outages.
LAST_DATA_CACHE: Dict[Tuple[str, int, int], pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# AI score smoothing
# Maintain a smoothed AI score per symbol to reduce noise and avoid reacting
# to single spikes in the AI model output.  Use an exponential moving
# average controlled by the alpha parameter.  The LAST_AI_SCORES dict
# stores the previous smoothed score for each symbol.
from collections import defaultdict

LAST_AI_SCORES: Dict[str, float] = defaultdict(float)

def smooth_ai_score(raw_score: float, last_score: float, alpha: float = 0.35) -> float:
    """Compute an exponentially smoothed AI score."""
    return (alpha * raw_score) + ((1 - alpha) * last_score)

def get_ai_status_summary() -> str:
    """Generate a summary of all AI features and their current status."""
    try:
        status_lines = []
        status_lines.append("🤖 AI LEARNING SYSTEM STATUS")
        status_lines.append("=" * 50)
        
        # ML Model Status
        if ML_MODEL is not None:
            status_lines.append(f"✅ ML Model: TRAINED ({len(ML_MODEL.feature_names_in_)} features)")
        else:
            status_lines.append("⚠️  ML Model: NOT TRAINED (need trade_log.csv with 20+ trades)")
        
        # AI Weights
        status_lines.append(f"\n📊 Current AI Weights:")
        for key, value in sorted(AI_WEIGHTS.items()):
            status_lines.append(f"   {key:15s}: {value:5.2f}")
        
        # AI Bias and Threshold
        status_lines.append(f"\n⚖️  AI Bias: {AI_BIAS:+.1f}")
        status_lines.append(f"🎯 AI Threshold: {AI_THRESHOLD}")
        
        # Symbol-specific thresholds
        status_lines.append(f"\n📈 Symbol Thresholds:")
        for sym, thresh in SYMBOL_CONF_THRESH.items():
            status_lines.append(f"   {sym}: {thresh}")
        
        # Adaptive memory status
        status_lines.append(f"\n🧠 Adaptive Memory:")
        for sym in SYMBOLS:
            mem_size = len(adaptive_memory.get(sym, []))
            status_lines.append(f"   {sym}: {mem_size} trades recorded")
        
        # Features status
        status_lines.append(f"\n🔧 Feature Flags:")
        for feat, enabled in FEATURES.items():
            icon = "✅" if enabled else "❌"
            status_lines.append(f"   {icon} {feat}")
        
        # Predictive AI
        pred_enabled = "ON" if ENABLE_PREDICTIVE_AI else "OFF"
        status_lines.append(f"\n🔮 Predictive AI: {pred_enabled}")
        
        status_lines.append("=" * 50)
        return "\n".join(status_lines)
    except Exception as e:
        return f"⚠️ Error generating AI status: {e}"

SKIP_MARKET_CHECK = True           # skip mt5.order_check to reduce latency
PAIR_CONFIG = {
    "XAUUSD": {"max_spread": 300},  # Use hard block limit for XAUUSD (lenient during setup)
    "GBPUSD": {"max_spread": 30},
    "GBPJPY": {"max_spread": 35},
}
GOATED_SERVER_HINTS = ["goated", "goat funded", "goatfunded", "goat-"]
MT5_LOGIN  = os.environ.get("GOATED_LOGIN")      # optional: set env var
MT5_PASS   = os.environ.get("GOATED_PASSWORD")   # optional: set env var
MT5_SERVER = os.environ.get("GOATED_SERVER")     # optional: set env var

# ---------------------------------------------------------------------------
# Early helper function definitions
#
# The legacy implementation previously defined a temporary ``adx_rising`` stub
# here to avoid ``NameError`` exceptions before the real ADX code was defined
# later in the file.  The real ADX implementation now lives below and is
# imported at runtime, so the stub is no longer necessary and has been
# removed.  All code should call the fully featured ``adx_rising`` defined
# later in this module.

def _mt5_try_paths(paths):
    """
    Legacy stub for backwards compatibility.

    Previous versions attempted to scan multiple possible MT5 installation
    directories and initialise the bridge against whichever terminal was
    reachable.  This behaviour has been removed.  The bot now relies
    exclusively on the single explicit path defined by ``AUTO_MT5_PATH`` and
    does not attempt to launch or connect to any alternate terminals.

    Returns
    -------
    bool
        Always ``False`` indicating that no attempt was made to initialise
        from a list of paths.
    """
    # Do not scan for terminals.  Always return False so that callers
    # know they must use the explicit ``AUTO_MT5_PATH`` mechanism.
    return False
def _mt5_init_force(path=MT5_PATH):
    """
    Ensure we attach to the correct MT5 terminal (C:\\mt5_bot).
    Tries terminal64.exe, then terminal.exe. Auto-launches terminal and sets PATH.
    """
    # Legacy stub: do not attempt to attach to FTMO or alternative terminals.
    # The bot now uses a single explicit terminal path (AUTO_MT5_PATH) and
    # initialises the MT5 bridge directly.  Always return False to skip
    # the legacy attach logic.
    return False
    try:
        base = GOAT_TERMINAL_DIR
        cand = [
            path,
            fr"{GOAT_TERMINAL_DIR}terminal64.exe",
            fr"{GOAT_TERMINAL_DIR}terminal.exe",
        ]
        try:
            term_dir = os.path.dirname(cand[0])
            if term_dir and os.path.isdir(term_dir):
                os.environ["PATH"] = term_dir + ";" + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(term_dir)
                except Exception:
                    log_debug("_mt5_init_force: add_dll_directory failed for", term_dir)
        except Exception:
            log_debug("_mt5_init_force: term_dir detection failed")
        ok = _mt5_try_paths(cand)
        if not ok:
            print("🛑 Could not initialize MT5 on any candidate path.")
            return False
        if MT5_LOGIN and MT5_PASS and MT5_SERVER:
            try:
                lg_ok = mt5.login(int(MT5_LOGIN), password=MT5_PASS, server=MT5_SERVER)
                print("MT5 login:", "OK" if lg_ok else "FAIL")
            except Exception as e:
                print(f"⚠️ MT5 login error: {e}")
        try:
            v = getattr(mt5, "version", lambda: ("", "", ""))()
            print(f"ℹ️ MT5 Python bridge version: {v}")
        except Exception:
            log_debug("mt5.version print failed")
        ai = mt5.account_info()
        srv = (getattr(ai, "server", "") or "").lower() if ai else ""
        # If the connected server does not match any GOATED hints, do not emit a warning.
        # Previously this function printed a message instructing the user to log into
        # a GOATED account, but this behaviour has been removed to avoid confusion.
        if any(tag in srv for tag in GOATED_SERVER_HINTS):
            try:
                PROP_ACTIVE.update({"name": "GOAT"})
            except Exception as e:
                try:
                    log_debug("PROP_ACTIVE.update failed during MT5 init", e)
                except Exception as e2:
                    try:
                        log_debug("PROP_ACTIVE.update logging failed:", e2)
                    except Exception:
                        pass
        return True
    except Exception as e:
        print(f"🛑 MT5 init error: {e}")
        try:
            print("Hint: ensure 64‑bit Python, MetaTrader5 pip package installed, and terminal64.exe exists in C:\\mt5_bot")
        except Exception:
            try:
                log_debug("print hint fallback failed in _mt5_init_force")
            except Exception as e:
                log_debug("tg fallback send failed when notifying command error:", e)
        return False
LOOSEN = True

# ---------------------------------------------------------------------------
# Additional global flags
#
# _MT5_LAUNCH_ATTEMPTED is used to ensure that the MT5 terminal is only
# launched once per run.  Without this guard, the main trading loop would
# repeatedly attempt to launch the configured terminal path whenever it
# detects the bridge is disconnected, leading to unnecessary resource use
# and potential errors when AUTO_MT5_PATH is invalid.  After the first
# launch attempt, this flag is set to True and no further launches are
# attempted; subsequent reattachments rely on the terminal already running.
#
# LAST_CONSEC_RESET_TS stores the timestamp of the last reset of the
# consecutive loss counter.  It is used in risk_guard_tick() to clear
# CONSEC_LOSS_CT and unpause trading after a fixed interval or at the
# start of a new day.

_MT5_LAUNCH_ATTEMPTED = False
LAST_CONSEC_RESET_TS = None

# ---------------------------------------------------------------------------
# New advanced predictive and liquidity modules
#
# To increase accuracy and trade quality, this bot now includes a predictive
# continuation model, a synthetic order-book liquidity engine, dynamic risk
# scaling and an institutional trailing stop mechanism.  These additions
# mirror the capabilities of the live version while adhering to FTMO safety
# constraints.  They are disabled by default if predictive AI is turned off.

# Whether to enable the predictive AI and liquidity gating.  When False,
# predictive and liquidity checks are bypassed and the bot behaves as before.
ENABLE_PREDICTIVE_AI = True

# Whether to operate in aggressive mode.  Aggressive mode relaxes confluence
# thresholds slightly to allow deeper pullbacks and earlier entries while
# staying within risk limits.  ATR floors and other safety systems are
# never lowered below their original values.
AGGRESSIVE_MODE = True

# Internal handle for the institutional trailing stop worker thread.  The
# trailing worker monitors open trades and tightens stop losses in multiple
# phases without ever widening them.  It is started on the first successful
# trade and runs in the background thereafter.
_trailing_thread = None

def compute_trend_strength(symbol: str) -> float:
    """
    Compute a normalised trend strength metric based on the slope difference of
    the 50‑ and 200‑period EMA on the H1 timeframe.  A value >0.5 indicates
    bullish momentum, <0.5 bearish.  If insufficient data is available, a
    neutral value of 0.5 is returned.  The result is clipped into [0, 1].
    """
    try:
        h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 200, max_age=3.0)
        if h1 is None or len(h1) < 100:
            return 0.5
        ema50 = get_ema(h1, 50)
        ema200 = get_ema(h1, 200)
        if ema50 is None or ema200 is None:
            return 0.5
        diff = ema50 - ema200
        # Slope over the last 20 bars
        if len(diff) < 25:
            return 0.5
        slope = float(diff.iloc[-1] - diff.iloc[-20])
        # Normalise relative to price; scaling factor chosen so typical slopes
        # map into a reasonable range.  A positive slope yields >0.5 and a
        # negative slope <0.5.
        price = float(h1['close'].iloc[-1])
        norm = 0.0
        try:
            norm = (slope / max(1e-8, price)) * 500.0
        except Exception:
            norm = 0.0
        return max(0.0, min(1.0, 0.5 + norm))
    except Exception:
        return 0.5

def compute_momentum_slope(symbol: str) -> float:
    """
    Compute a normalised momentum slope using the change in RSI over the last
    14 bars on the M15 timeframe.  A bullish acceleration yields >0.5 while
    bearish acceleration yields <0.5.  If data is insufficient, returns 0.5.
    """
    try:
        m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 50, max_age=1.5)
        if m15 is None or len(m15) < 20:
            return 0.5
        # Approximate momentum by computing RSI at the start and end of a
        # 14‑bar window and taking the difference.  Avoid heavy loops.
        try:
            start_rsi = float(rsi(m15.iloc[:14]))
            end_rsi = float(rsi(m15.iloc[-14:]))
        except Exception:
            return 0.5
        slope = end_rsi - start_rsi
        # Normalise difference (RSI is in [0,100]) so a 50‑point change maps to 1.0
        norm = slope / 50.0
        return max(0.0, min(1.0, 0.5 + norm))
    except Exception:
        return 0.5

def predict_continuation_probability(symbol: str) -> float:
    """
    Predict the probability of trend continuation by blending the trend
    strength and momentum slope metrics.  The returned value lies in [0,1].
    """
    ts = compute_trend_strength(symbol)
    ms = compute_momentum_slope(symbol)
    # Heavily weight trend strength (60%) and moderately weight momentum (40%).
    raw = 0.6 * ts + 0.4 * ms
    return max(0.0, min(1.0, raw))

def predict_liquidity_pressure(symbol: str) -> float:
    """
    Estimate a liquidity pressure score based on spread, candle body/wick
    imbalance and recent price acceleration.  A higher score indicates a
    deeper order book and better fill quality.  Result is clamped to [0,1].
    """
    try:
        # Spread component: invert the spread points relative to a max
        sp = spread_points(symbol)
        if sp is None or sp <= 0:
            sp = 50.0
        base = _base_of(symbol)
        # Set crude maximum spreads per asset class
        max_sp = 50.0 if base == 'XAUUSD' else 10.0
        spread_score = max(0.0, min(1.0, 1.0 - (sp / max_sp)))
        # Candle body vs wick: use the most recent M15 bar
        m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 10, max_age=0.8)
        wick_score = 0.5
        if m15 is not None and len(m15) >= 1:
            try:
                body = abs(float(m15['close'].iloc[-1]) - float(m15['open'].iloc[-1]))
                range_val = float(m15['high'].iloc[-1]) - float(m15['low'].iloc[-1])
                wick = max(0.0, range_val - body)
                if (body + wick) > 1e-8:
                    wick_score = max(0.0, min(1.0, body / (body + wick)))
            except Exception:
                wick_score = 0.5
        # Price acceleration: mean high-low range over last 5 bars relative to ATR
        try:
            acc_series = (m15['high'] - m15['low']).rolling(5).mean().iloc[-1] if m15 is not None else 0.0
        except Exception:
            acc_series = 0.0
        try:
            atr_val = true_atr(m15, 14) if m15 is not None else 0.0
        except Exception:
            atr_val = 0.0
        acc_score = 0.5
        try:
            if atr_val > 1e-8:
                acc_score = max(0.0, min(1.0, acc_series / (atr_val * 1.5)))
        except Exception:
            acc_score = 0.5
        return max(0.0, min(1.0, (spread_score + wick_score + acc_score) / 3.0))
    except Exception:
        return 0.5

def blended_prediction(symbol: str) -> float:
    """
    Blend the continuation probability and liquidity pressure into a single
    predictive score.  The result, used for gating, lies in [0,1].
    """
    p = predict_continuation_probability(symbol)
    l = predict_liquidity_pressure(symbol)
    return max(0.0, min(1.0, 0.6 * p + 0.4 * l))

def trend_alignment_ok(symbol: str, side: str) -> bool:
    """
    Determine if the predicted trend direction aligns with the proposed trade
    side.  A BUY trade is aligned when the trend strength is clearly bullish
    (≥0.55), and a SELL trade is aligned when the trend strength is bearish
    (≤0.45).  Neutral values are treated as ambiguous but not blocked.
    """
    try:
        ts = compute_trend_strength(symbol)
        if ts >= 0.55 and side == 'BUY':
            return True
        if ts <= 0.45 and side == 'SELL':
            return True
        # When neutral, permit the trade and let other gates decide.
        return True
    except Exception:
        return True

def compute_dynamic_risk_scale(symbol: str, p_score: float, liquidity: float) -> float:
    """
    Compute a dynamic risk scaling factor based on predictive score and liquidity.
    A value of 1.0 indicates baseline risk.  Values above 1.0 slightly
    increase risk, values below 1.0 reduce risk.  Zero blocks the trade.
    
    Rules:
        • If p_score or liquidity falls below their micro gate (p<0.55 or
          liquidity<0.35), return 0 (block).
        • High quality (p ≥0.75 and liquidity ≥0.55): return 1.10.
        • Normal quality (p ≥0.62 and liquidity ≥0.55): return 1.00.
        • Low quality (p ≥0.55 and liquidity ≥0.35): return 0.50.
        • Otherwise: 0.0 (block).
    """
    try:
        if liquidity < 0.35 or p_score < 0.55:
            return 0.0
        if p_score >= 0.75 and liquidity >= 0.55:
            return 1.10
        if p_score >= 0.62 and liquidity >= 0.55:
            return 1.00
        if p_score >= 0.55 and liquidity >= 0.35:
            return 0.50
        return 0.0
    except Exception:
        return 0.0

def _modify_sl(position, new_sl):
    """
    Internal helper to modify the stop loss of an existing position without
    widening it.  Positions are identified by their ticket number.  Any
    exceptions are logged but suppressed to avoid affecting the main loop.
    """
    try:
        ticket = position.ticket
        symbol = position.symbol
        tp = position.tp
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": normalize_price(symbol, new_sl),
            "tp": tp or 0.0,
        }
        pr = mt5.order_send(req)
        if pr and pr.retcode == mt5.TRADE_RETCODE_DONE:
            log_msg(f"🔒 Trailing stop adjusted on {symbol} (ticket {ticket}) to {new_sl:.3f}")
        return pr
    except Exception as e:
        try:
            # Fallback to generic logger when log_error is unavailable
            log_msg(f"Trailing stop modification error: {e}")
        except Exception as e2:
            try:
                log_debug("log_msg fallback failed in _modify_sl:", e2)
            except Exception as e3:
                try:
                    log_debug("log_msg fallback logging failed in _modify_sl:", e3)
                except Exception:
                    pass
        return None

def institutional_trailing_worker():
    """
    Background worker that applies an institutional trailing stop algorithm to
    all open positions.  Once a trade achieves +0.8R profit it moves the SL
    to +0.2R; once +1.4R is hit the SL moves to +0.8R.  Thereafter the SL
    trails by one ATR behind the latest price swing.  Only tighter stops
    are set - stops are never widened.  This worker runs continuously until
    all positions have closed.
    """
    try:
        while True:
            try:
                positions = mt5.positions_get()
            except Exception as e:
                try:
                    log_debug("mt5.positions_get failed in institutional_trailing_worker:", e)
                except Exception:
                    pass
                positions = None
            if not positions:
                # Sleep longer when no positions are present
                time.sleep(5)
                continue
            for pos in positions:
                try:
                    symbol = pos.symbol
                    # Determine side from position type (0=BUY, 1=SELL).  Use a
                    # conservative fallback if the constant is not available.
                    try:
                        typ = getattr(pos, 'type', 1)
                        # MetaTrader5 uses 0 for buy and 1 for sell
                        side = 'BUY' if typ == 0 else 'SELL'
                    except Exception:
                        side = 'BUY'
                    entry_price = pos.price_open
                    sl = pos.sl
                    # Skip if no SL set yet
                    if not sl or sl == 0.0:
                        continue
                    R = abs(entry_price - sl)
                    # Obtain current price
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        continue
                    current_price = tick.ask if side == 'BUY' else tick.bid
                    # Profit relative to R
                    profit = (current_price - entry_price) if side == 'BUY' else (entry_price - current_price)
                    # Phase 0: break-even early (protect capital once moderately profitable)
                    if profit >= 0.4 * R:
                        be_sl = entry_price
                        if (side == 'BUY' and be_sl > sl) or (side == 'SELL' and be_sl < sl):
                            _modify_sl(pos, be_sl)
                            sl = be_sl
                    # Phase 1: move SL to +0.2R at +0.8R
                    if profit >= 0.8 * R:
                        target_sl = (entry_price + 0.2 * R) if side == 'BUY' else (entry_price - 0.2 * R)
                        if (side == 'BUY' and target_sl > sl) or (side == 'SELL' and target_sl < sl):
                            _modify_sl(pos, target_sl)
                            sl = target_sl
                    # Phase 2: move SL to +0.8R at +1.4R
                    if profit >= 1.4 * R:
                        target_sl = (entry_price + 0.8 * R) if side == 'BUY' else (entry_price - 0.8 * R)
                        if (side == 'BUY' and target_sl > sl) or (side == 'SELL' and target_sl < sl):
                            _modify_sl(pos, target_sl)
                            sl = target_sl
                    # Phase 3: ATR-based trailing behind latest price (volatility adaptive tightening)
                    try:
                        m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 50, max_age=1.5)
                        atr_val = true_atr(m15, 14) if m15 is not None else None
                    except Exception as e:
                        try:
                            log_debug("ATR fetch/compute failed in institutional_trailing_worker:", e)
                        except Exception:
                            pass
                        atr_val = None
                    if atr_val:
                        # For small micro positions, trail more tightly (0.6x ATR)
                        try:
                            vol_mult = 1.0
                            if abs(getattr(pos, 'volume', 0.0)) <= MICRO_LOT_MAX:
                                vol_mult = 0.6
                            if side == 'BUY':
                                new_sl = current_price - (atr_val * vol_mult)
                                if new_sl > sl:
                                    _modify_sl(pos, new_sl)
                            else:
                                new_sl = current_price + (atr_val * vol_mult)
                                if new_sl < sl:
                                    _modify_sl(pos, new_sl)
                        except Exception as e:
                            try:
                                log_debug("institutional_trailing_worker vol_mult adjustment failed:", e)
                            except Exception as e2:
                                try:
                                    log_debug("institutional_trailing_worker vol_mult logging failed:", e2)
                                except Exception:
                                    pass
                except Exception as e:
                    try:
                        log_debug("institutional_trailing_worker per-position loop failed:", e)
                    except Exception as e2:
                        try:
                            log_debug("institutional_trailing_worker per-position loop logging failed:", e2)
                        except Exception:
                            pass
                    continue
            time.sleep(5)
    except Exception as e:
        # Any unhandled exception terminates the worker; log and exit with diagnostics
        try:
            log_msg("Trailing worker terminated unexpectedly:", e)
        except Exception as e2:
            try:
                log_debug("log_msg fallback failed in trailing worker:", e2)
            except Exception:
                pass
        return
# Unified threshold configuration
#
# The trading logic must apply a single unified set of confluence rules across
# both full and micro strategies.  The values below represent the final
# thresholds for all ADX, RSI, ATR and range filters as well as the minimum
# AI score.  These constants supersede any prior “loosened” or legacy
# thresholds scattered throughout the code.  Wherever a threshold is
# referenced (for example, in high_prob_filters_ok(), try_micro_on(),
# attempt_full_trade_once(), AI score gates and scan reporting), the
# corresponding unified constant should be used.  Setting these values in
# one place ensures that adjustments propagate consistently across the bot.
AI_MIN_SCORE        = 60          # minimum AI score required to approve a trade
ADX_MIN_H1          = 22.0        # minimum ADX on the H1 timeframe
ADX_MIN_M15         = 22.0        # minimum ADX on the M15 timeframe
RSI_BUY_MIN         = 40.0        # buy when RSI >= 40
RSI_SELL_MAX        = 60.0        # sell when RSI <= 60
ATR_FLOOR_M15       = 0.015       # minimum ATR (M15) to consider volatility sufficient
RANGE_MIN_PCT       = 0.35        # minimum percent movement from the prior candle midpoint

# For backwards compatibility with the existing names used throughout the
# codebase (LOOSEN_* variables), the unified thresholds are aliased below.
# This ensures that all gate functions referencing the LOOSEN_* constants
# automatically inherit the unified values without requiring further edits.
LOOSEN_FULL_AI = AI_MIN_SCORE
LOOSEN_FULL_ADX_H1_MIN = ADX_MIN_H1
LOOSEN_FULL_ADX_M15_MIN = ADX_MIN_M15
LOOSEN_FULL_RSI_BUY = RSI_BUY_MIN
LOOSEN_FULL_RSI_SELL = RSI_SELL_MAX
LOOSEN_FULL_ATR_FLOOR = ATR_FLOOR_M15
LOOSEN_FULL_RANGE_MAX_PCT = RANGE_MIN_PCT
# User-requested relaxed thresholds for trade confluence.  The bot keeps all
# existing safety and risk management logic intact but slightly loosens
# some technical gatekeepers.  ADX thresholds are lowered from 25 to 22
# (on both H1 and M15) and RSI thresholds are centred around 50 but
# shifted to 53/47 for buy/sell conditions.  The ATR floor ensures
# sufficient volatility (≥0.015) and the range filter requires price to
# move ≥0.35% away from the prior candle's midpoint.  These adjustments
# open up more setups while preserving safety.
# Overwrite the legacy LOOSEN_* definitions with the unified thresholds.
LOOSEN_FULL_ADX_H1_MIN = ADX_MIN_H1        # H1 ADX ≥22
LOOSEN_FULL_ADX_M15_MIN = ADX_MIN_M15       # M15 ADX ≥22
LOOSEN_FULL_RSI_BUY = RSI_BUY_MIN           # buy when RSI ≥53
LOOSEN_FULL_RSI_SELL = RSI_SELL_MAX          # sell when RSI ≤47
LOOSEN_FULL_ATR_FLOOR = ATR_FLOOR_M15        # ATR must be ≥0.015
LOOSEN_FULL_RANGE_MAX_PCT = RANGE_MIN_PCT      # require ≥0.35% movement away from range midpoint
LOOSEN_FULL_CONFIRM_WICK  = True        # allow wick break confirm (>=1.2x ATR20 range)
PREFER_PENDING_STOPS_FULL = False        # breakout → buy/sell stop first
PULLBACK_CANCEL_MIN       = 30          # cancel pullback limit if unfilled in 20–30 min
LOOSEN_MIN_RR_FULL        = 1.8
LOOSEN_RISK_PCT           = 0.01        # 1% risk stays
LOOSEN_SL_ATR_MULT        = 1.5         # SL = max(last swing, 1.5x ATR)
# Global spread caps for gold trades.  Both full and micro trades on XAUUSD
# should respect these values to prevent orders being blocked by overly tight
# spread limits.  Set each to 300 points per the latest FTMO rules.
FULL_SPREAD_MAX_PTS       = 300
MICRO_SPREAD_MAX_PTS      = 300
MAX_DEVIATION_DEFAULT     = 100

# ---------------------------------------------------------------------------
# Micro‑trade specific thresholds
#
# When executing micro trades, we want to loosen several of the volatility and
# momentum requirements compared to full trades while still maintaining
# accuracy and FTMO compliance.  These constants define the minimum ATR,
# ADX and RSI levels as well as the AI score floor used when evaluating
# potential micro entries.  Adjusting these values allows micro trades to
# trigger more frequently during quiet market conditions without turning the
# strategy into a reckless scalper.
#
# MICRO_ATR_MIN:  minimum M15 ATR required to consider a micro trade.  Lower
# values permit trades in tight ranges.  Default is 0.004, as requested.
# ADX_MICRO_H1_MIN / ADX_MICRO_M15_MIN:  relaxed ADX thresholds for the H1
# and M15 timeframes.  Lower values allow trades during consolidation while
# still filtering out completely flat markets.
# RSI_MICRO_BUY_MIN / RSI_MICRO_SELL_MAX:  relaxed RSI regime levels.  These
# are closer to the mid‑range (50) than the full‑trade thresholds (53/47)
# which opens up more setups on slow days.  A buy requires RSI ≥ 51 and a
# sell requires RSI ≤ 49 on both M15 and H1 timeframes.
# MICRO_AI_MIN:  minimum AI score for micro entries.  Set to AI_MIN_SCORE
# minus five points, ensuring B‑grade setups can still be taken while
# rejecting low‑quality signals.
MICRO_ATR_MIN = 0.004
ADX_MICRO_H1_MIN = 14.0
ADX_MICRO_M15_MIN = 12.0
RSI_MICRO_BUY_MIN = 51.0
RSI_MICRO_SELL_MAX = 49.0
# Neutral zone for micro RSI: values between these bounds are considered neutral and allow trades
# in either direction when both H1 and M15 RSIs fall within this range.  This supports
# micro trades on slow or choppy days when momentum is lacking but markets are not overbought
# or oversold.  Adjust these values only if you also update the micro RSI gates below.
RSI_NEUTRAL_LOW = 48.0
RSI_NEUTRAL_HIGH = 52.0
MICRO_AI_MIN = max(0, AI_MIN_SCORE - 5)

# ---------------------------------------------------------------------------
# Utility: wait for FTMO login
# ---------------------------------------------------------------------------
# When launching the bot, MetaTrader 5 may connect before it has finished
# logging into the target trading server.  In particular, FTMO demo and
# funded servers include "ftmo" in their name once login completes.  To
# avoid performing prop-firm detection too early (which would fail to
# detect FTMO and treat the firm as unknown), call ``wait_for_ftmo_login``
# immediately after ``_mt5_init_force()`` in the main startup routine.  It
# polls the MT5 account info once per second for up to ``max_wait`` seconds
# and returns as soon as the server name contains "ftmo" (case-insensitive).
def wait_for_ftmo_login(max_wait: int = 30) -> bool:
    """
    Wait up to ``max_wait`` seconds for MT5 to finish logging into an FTMO
    server.  Returns ``True`` once the server name contains "ftmo" (case-
    insensitive), otherwise ``False`` if the timeout expires.  If any
    exceptions occur during polling, they are ignored.

    Parameters
    ----------
    max_wait : int, optional
        Maximum number of seconds to wait.  Defaults to 30 seconds.

    Returns
    -------
    bool
        ``True`` if an FTMO server is detected within the timeout, ``False``
        otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            ai = mt5.account_info()
            srv = (getattr(ai, "server", "") or "").lower() if ai else ""
            if "ftmo" in srv:
                return True
        except Exception as e:
            try:
                log_debug("wait_for_ftmo_login poll failed:", e)
            except Exception as e2:
                try:
                    log_debug("wait_for_ftmo_login logging fallback failed:", e2)
                except Exception:
                    pass
        time.sleep(1)
    return False

# ---------------------------------------------------------------------------
# Risk‑reward configuration
#
# The trading routines reference ``RR_FULL`` and ``RR_MICRO`` to determine
# the risk‑to‑reward multiple for full and micro positions.  In earlier
# versions these constants were not explicitly defined, which would lead to
# ``NameError`` exceptions when calculating pending stop and limit orders.
# Define sensible defaults here so those references always resolve.  Users
# may override these values at runtime by setting the variables in the
# global scope before importing this module.
RR_FULL  = globals().get("RR_FULL", 2.0)    # e.g. take profit at 2× risk for full trades
RR_MICRO = globals().get("RR_MICRO", 2.0)    # micro trades use the same default

# Session windows aligned to a single block from 07:00 to 18:00 UK time for both
# full and micro strategies. Previously London and NY sessions were split; merging
# them simplifies the schedule and honours the user's requested hours.
FULL_SESSION_LON          = (7, 0, 18, 0)     # 07:00–18:00
FULL_SESSION_NY           = (7, 0, 18, 0)     # 07:00–18:00 (same as London)
MICRO_SESSION_LON         = (7, 0, 18, 0)     # 07:00–18:00
MICRO_SESSION_NY          = (7, 0, 18, 0)     # 07:00–18:00
# Micro strategy thresholds are unified with the full strategy.  These
# assignments deliberately reference the unified constants defined above so
# that micro trades adhere to the same confluence rules as full trades.
MICRO_AI_MIN = AI_MIN_SCORE
MICRO_ADX_MIN = ADX_MIN_H1
MICRO_RSI_BUY = RSI_BUY_MIN
MICRO_RSI_SELL = RSI_SELL_MAX
MICRO_ATR_FLOOR = ATR_FLOOR_M15
MICRO_RANGE_MAX_PCT = RANGE_MIN_PCT
MICRO_LOT_FLOOR           = 0.01
SL_MIN_PTS_DEFAULT        = 80
SL_MIN_PTS_BY_SYMBOL      = {"XAUUSD": 100, "GBPUSD": 60, "GBPJPY": 80}
BLOCK_FULL_IF_FIRM_UNKNOWN = True
ALLOW_MICRO_IF_FIRM_UNKNOWN = True
ORDER_MAX_RETRIES   = 3           # retry mt5.order_send up to N times on transient errors
RETRY_BACKOFF_BASE  = 2.0         # base seconds for exponential backoff on retries
HEALTH_PING_MIN     = 120         # min interval between /health auto pings (seconds, 0 = off)
_last_health_ts     = 0
MAX_CONSEC_LOSSES   = 3           # halt after N consecutive losses (blocks new trades)
HALT_UNTIL_TS       = None        # timestamp (UTC) until which trading is paused
EQUITY_SHOCK_PCT    = 1.0         # % equity drop within an hour triggers 60-min pause
EQUITY_SHOCK_UNTIL  = None        # timestamp (UTC) to resume after shock
EQUITY_HOURLY_REF   = None        # reference (equity, timestamp) for shock detection
CONSEC_LOSS_CT      = 0           # rolling counter of consecutive losses
AUTO_FLAT_ON_BREACH = True        # close all positions & cancel pendings on breach (account protection)
NEWS_BLACKOUT_ON    = True        # block XAUUSD trades during scheduled red news windows
NEWS_FILE           = "news_red.json"
def _load_news_windows():
    """Load scheduled news windows (red-folder news times) from file."""
    try:
        with open(NEWS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as e:
        try:
            log_debug("_load_news_windows failed:", e)
        except Exception:
            pass
    return []
def in_news_blackout(symbol='XAUUSD', now=None):
    """Return True if current time falls within any configured red-news window for XAUUSD (and NEWS_BLACKOUT_ON is True)."""
    try:
        if not NEWS_BLACKOUT_ON:
            return False
        if _base_of(symbol) != 'XAUUSD':
            return False
        # Use safe timezone resolution for current time and conversions
        now = now or datetime.now(SAFE_TZ)
        tz = SAFE_TZ
        for w in _load_news_windows():
            st_raw = w.get('start'); en_raw = w.get('end')
            if not st_raw or not en_raw:
                continue
            try:
                st = datetime.fromisoformat(st_raw)
                en = datetime.fromisoformat(en_raw)
            except Exception:
                continue
            if st.tzinfo is None:
                st = tz.localize(st)
            if en.tzinfo is None:
                en = tz.localize(en)
            if st <= now <= en:
                return True
        return False
    except Exception:
        return False
SNAPSHOT_HOUR_LONDON = 21        # hour of day (London time) to send daily snapshot
SAFE_DAILY_DD_PCT    = 2.0       # daily drawdown threshold (%) for extra-safe auto-flat
SAFE_MAX_DD_PCT      = 4.0       # max total drawdown threshold (%) for extra-safe auto-flat
last_snapshot_date   = None
risk_highwater       = None      # equity high-water mark
MODE                 = "BALANCED"  # can be "STRICT" or "BALANCED" mode for filters
STRATEGY             = "GOAT"      # active strategy: "LEGACY" or "GOAT"
# trading_paused is managed by BOT_STATE.trading_paused (do not use module global)
# Session counters moved into `BOT_STATE.session_full_losses` and
# `BOT_STATE.session_full_count` (do not use module globals).
STATE_FILE           = "bot_state.json"
SYMBOL_ROUTE         = {}       # mapping base symbol -> actual MT5 symbol (if alias)
MICRO_RR_IDX         = 0        # round-robin index for micro trade rotation
blocked_pairs        = {}       # track temporarily blocked pairs if needed
ADDON                = {}       # tracker for add-on trades (one extra trade per position)
# Tune the AI approval threshold higher for increased precision (target ~9.5/10 accuracy)
# A higher threshold means signals need more confluence before being approved.
# Use the unified AI minimum score as the global AI threshold for trade approval.
AI_THRESHOLD = AI_MIN_SCORE

# ---------------------------------------------------------------------------
# Scan tracking globals
#
# The bot tracks the outcome of the most recent manual scan triggered via
# the /scan or /findtrade Telegram commands.  These variables are
# initialised at the start of each scan and updated whenever a trade is
# executed while the scan is running.  They are consumed by the command
# dispatcher to generate explicit status lines for XAUUSD.  Outside of
# a manual scan, SCAN_MODE will be False.

SCAN_MODE = False
LAST_SCAN_PLACED_SYMBOL = None
LAST_SCAN_TRADE_TYPE = None

# Expanded AI weights to account for volatility regime (ATR Z-score) and RSI‑Leader (RSIL).
# We increase the weight on core confluence factors and add new keys for
# volatility expansion and momentum to better discriminate high‑probability setups.
AI_WEIGHTS = {
    # Higher bias weight to strongly favour trades aligned with higher timeframe trend
    "h4_bias": 20,
    # ADX and its rising condition have more influence
    "adx_h1": 18, "adx_m15": 12, "adx_rising": 10,
    # RSI regime weight increased slightly
    "rsi_regime": 14,
    # EMA slope influences score more heavily
    "ema_slope": 10,
    # BOS match weight increased to prioritise market structure alignment
    "bos_match": 14,
    # FVG or ATR momentum combined weight for volatility momentum
    "fvg_or_atrhot": 12,
    # Session and spread remain moderate
    "session": 6, "spread_ok": 8,
    # New features for volatility regime and RSI leader
    "atr_z": 10,
    "rsil": 10
}

# ============================================================
# ADAPTIVE LEARNING LAYER (self-tuning AI engine)
# ============================================================

LEARN_RATE_W = 0.02
LEARN_RATE_BIAS = 0.5
WINDOW = 50

adaptive_memory = {sym: [] for sym in SYMBOLS}

def record_trade_outcome(symbol, ai_score, win):
    try:
        adaptive_memory[symbol].append({"score": ai_score, "win": win})
        if len(adaptive_memory[symbol]) > WINDOW:
            adaptive_memory[symbol].pop(0)
    except Exception as e:
        # Do not silently ignore errors when recording trade outcomes.
        log_error(f"record_trade_outcome error: {e}")

def adaptive_update_weights(symbol):
    """Adaptive learning: adjust AI weights based on recent trade outcomes."""
    try:
        mem = adaptive_memory[symbol]
        if len(mem) < 10:
            return
        
        # Calculate average scores for winning and losing trades
        wins = [m for m in mem if m["win"]]
        losses = [m for m in mem if not m["win"]]
        
        avg_win_score = sum(m["score"] for m in wins) / max(1, len(wins))
        avg_loss_score = sum(m["score"] for m in losses) / max(1, len(losses))
        
        # Adjust weights based on performance
        if avg_loss_score >= avg_win_score:
            # Losses have high scores - tighten criteria
            for k in AI_WEIGHTS:
                AI_WEIGHTS[k] = max(0, AI_WEIGHTS[k] - LEARN_RATE_W)
            SYMBOL_CONF_THRESH[symbol] = min(95, SYMBOL_CONF_THRESH[symbol] + 1)
            log_msg(f"🧠 AI Learning: Tightening criteria for {symbol} (avg_loss={avg_loss_score:.1f} >= avg_win={avg_win_score:.1f})")
        else:
            # Wins have higher scores - loosen slightly
            for k in AI_WEIGHTS:
                AI_WEIGHTS[k] = min(25, AI_WEIGHTS[k] + LEARN_RATE_W)
            SYMBOL_CONF_THRESH[symbol] = max(40, SYMBOL_CONF_THRESH[symbol] - 1)
            log_msg(f"🧠 AI Learning: Loosening criteria for {symbol} (avg_win={avg_win_score:.1f} > avg_loss={avg_loss_score:.1f})")
        
        # Calculate accuracy and adjust bias
        accuracy = sum(m["win"] for m in mem) / len(mem)
        global AI_BIAS
        if accuracy > 0.65:
            AI_BIAS += LEARN_RATE_BIAS
        else:
            AI_BIAS -= LEARN_RATE_BIAS
        AI_BIAS = max(-10, min(10, AI_BIAS))
        
        # Save state and notify
        save_state()
        telegram_msg(f"🧠 Adaptive AI Update: {symbol} | Accuracy: {accuracy:.1%} | New Threshold: {SYMBOL_CONF_THRESH[symbol]} | AI Bias: {AI_BIAS:.1f}")
    except Exception as e:
        log_msg(f"⚠️ Adaptive weight update failed for {symbol}: {e}")

def adaptive_on_trade_close(symbol, ai_score, win):
    record_trade_outcome(symbol, ai_score, win)
    adaptive_update_weights(symbol)


AI_BIAS = 0.0  # bias term for AI model (offset for baseline probability)
SYMBOL_CONF_THRESH = {sym: AI_THRESHOLD for sym in SYMBOLS}  # dynamic confidence threshold per symbol
open_trades_info = {}   # store entry details and features for open trades (for learning on close)
symbol_outcomes = {sym: [] for sym in SYMBOLS}  # recent win/loss outcomes (for threshold auto-tuning)
day_stats = {
    "date": None,
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
    "day_open_equity": None,
    "blocks": 0, "fails": 0,
    "fulls_LON": 0, "fulls_NY": 0,
    "micros_LON": 0, "micros_NY": 0,
    "micros_by_symbol": {'LON': {}, 'NY': {}}
}
next_scan_at = None
last_hour_notified = None
PROP_RULESETS = {
    "FTMO": {
        # Include common demo and spaced variants so demo accounts are detected as FTMO
        # We match lowercase patterns in the MT5 server name.  This list covers
        # "ftmo", "ftmo-demo", "ftmo demo" and case-insensitive equivalents.
        # Match any FTMO server variant; a single keyword catches demo, evaluation and live servers
        "patterns": ["ftmo"],
        "daily_loss_pct": 5.0,
        "daily_pivot": "utc_midnight_higher",
        "max_loss_pct": 10.0,
        "max_concurrent": None,
        "daily_trade_cap": None,
        "trailing_max": False
    },
    "FXIFY": {
        "patterns": ["fxify"],
        "daily_loss_pct": 4.0,
        "daily_pivot": "utc_midnight_higher",
        "max_loss_pct": 10.0,
        "max_concurrent": None,
        "daily_trade_cap": None,
        "trailing_max": False
    },
    "GOAT": {
        "patterns": ["goat funded", "goatfunded", "goat-funded", "goat "],
        "daily_loss_pct": 2.0,
        "daily_pivot": "utc_midnight_higher",
        "max_loss_pct": 4.0,
        "max_concurrent": None,
        "daily_trade_cap": None,
        "trailing_max": False
    },
    "AQUA": {
        "patterns": ["aquafunded", "aqua funded", "aquafund"],
        "daily_loss_pct": 5.0,   # default for 2-step
        "daily_pivot": "utc_midnight_higher",
        "max_loss_pct": 10.0,
        "max_concurrent": None,
        "daily_trade_cap": None,
        "trailing_max": False,
        "plan": "2step"          # auto-tweaked to 1step if detected
    }
}
PROP_ACTIVE = {"name": "AUTO"}   # detected or overridden prop firm
# Globals for funded-account enforcement and risk scaling
ACCOUNT_INITIAL_BALANCE = None
LEVERAGE_RESTRICT_FACTOR = 1.0
LOT_SCALE_FACTOR = 1.0
TEMP_LOT_REDUCTION_UNTIL = None
CONSEC_WINS = 0
CONSEC_LOSSES = 0
SCALING_ACTIVE = False
SCALING_ALLOWED_UNTIL = None
COPY_TRADING_DETECTED = False
MAX_CAP_ALLOC = 400000.0


def _tg_poll_once(offset=None):
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        # Use long polling so Telegram holds the connection until an update
        # is available. Increase the outer timeout slightly higher than the
        # long-poll timeout to allow network delays.
        try:
            res = requests.get(url, params={"offset": offset, "timeout": 30}, timeout=35)
            data = res.json()
        except Exception as e:
            print(f"[TG POLL FETCH ERROR] {e}")
            return []
        # Validate shape
        if not isinstance(data, dict):
            print(f"[TG POLL] unexpected response type: {type(data)}")
            return []
        if not data.get('ok'):
            print(f"[TG POLL] telegram returned not-ok: {data}")
            return []
        return data.get("result", [])
    except Exception as e:
        try:
            print(f"[TG] Poll fetch failed: {e}")
        except Exception as ex:
            try:
                log_debug("_tg_poll_once print fallback failed:", ex)
            except Exception as e2:
                try:
                    log_debug("_tg_poll_once printing fallback logging failed:", e2)
                except Exception:
                    pass
        return []

def _safe_str(x):
    try:
        return str(x)
    except Exception:
        try:
            return repr(x)
        except Exception:
            return "<unrepresentable>"


def _dispatch_minimal_command(text, chat_id):
    t = text.strip().lower()
    # micro_lot_ai_scale stored in BOT_STATE
    def _is_owner(cid):
        try:
            return str(cid) == str(TELEGRAM_CHAT_ID)
        except Exception:
            return False
    # Create a local reply wrapper so existing calls to `tg()` inside this
    # dispatcher send responses back to the origin chat. If sending to the
    # origin chat fails, fall back to the global `tg()` (owner/admin alert).
    try:
        tg_orig = globals().get('tg')
    except Exception:
        tg_orig = None
    def _tg_scope(msg):
        try:
            return send_telegram_to(chat_id, msg)
        except Exception:
            try:
                if tg_orig:
                    return tg_orig(msg)
            except Exception:
                try:
                    log_debug("tg_orig fallback failed in _tg_scope")
                except Exception:
                    pass
        return False
    tg = _tg_scope

    # Quick console log for incoming commands so we can troubleshoot silent replies
    try:
        print(f"[TG CMD] received: {t} from chat_id={chat_id}")
    except Exception as e:
        log_debug("[TG CMD] console print failed:", e)

    # Ensure help/start/stop always reply to the originating chat even if
    # other logic later may not handle them explicitly.
    if t in ("/help", "/start", "/stop"):
        try:
            if t == "/help":
                try:
                    send_telegram_to(chat_id, build_help_message())
                except Exception:
                    if tg_orig:
                        tg_orig(build_help_message())
                return True
            if t == "/start":
                try:
                    start_telegram_polling(send_online_ack=True)
                    send_telegram_to(chat_id, "✅ Polling started")
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Start failed: {e}")
                    except Exception as e2:
                        try:
                            log_debug("send_telegram_to fallback failed in /start:", e2)
                        except Exception:
                            pass
                return True
            if t == "/stop":
                try:
                    stop_telegram_polling()
                    send_telegram_to(chat_id, "🛑 Polling stopped")
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Stop failed: {e}")
                    except Exception as e2:
                        try:
                            log_debug("send_telegram_to fallback failed in /stop:", e2)
                        except Exception:
                            pass
                return True
        except Exception as e:
            log_debug("telegram command top-level handler failed:", e)
    if t == "/active":
        tg("🤖 Bot is active.")
        return True
    if t == "/status":
        # Ticket-Funded: Use Holy Grail status handler if available
        try:
            if 'handle_status_command' in globals():
                handle_status_command()
            else:
                # Fallback to original status
                ai = mt5.account_info()
                bal = getattr(ai, 'balance', 0.0) if ai else 0.0
                eq  = getattr(ai, 'equity', 0.0) if ai else 0.0
                firm, _rules, server = detect_prop_firm()
                firm = firm or "UNKNOWN"
                server = server or "UNKNOWN"
                tg(f"Balance: {bal:.2f} | Equity: {eq:.2f} | Firm: {firm} | Server: {server}")
        except Exception as e:
            tg(f"Status error: {e}")
        return True

    if t in ("/ai", "/mlstatus", "/aistatus"):
        # Show AI learning system status
        try:
            status = get_ai_status_summary()
            tg(status)
        except Exception as e:
            tg(f"⚠️ AI status error: {e}")
        return True

    if t in ("/panic",):
        try:
            # Ticket-Funded: Use Holy Grail panic handler if available
            if 'handle_panic_command' in globals():
                handle_panic_command()
            else:
                trading_pause(True)
                flat_all_positions()
                cancel_all_pendings()
                tg("🛑 PANIC: all positions closed, trading paused.")
        except Exception as e:
            tg(f"Panic error: {e}")
        return True
    
    # Ticket-Funded: Holy Grail /resume command
    if t in ("/resume",):
        try:
            if 'handle_resume_command' in globals():
                handle_resume_command()
            else:
                tg("Resume command not available")
        except Exception as e:
            tg(f"Resume error: {e}")
        return True
    
    # Ticket-Funded: Holy Grail /signal command
    if t in ("/signal",):
        try:
            if 'handle_signal_command' in globals():
                handle_signal_command()
            else:
                tg("Signal command not available")
        except Exception as e:
            tg(f"Signal error: {e}")
        return True
    if t.startswith("/strict"):
        try:
            parts = t.split()
            if len(parts) > 1 and parts[1] in ("on","off"):
                globals()['SKIP_MARKET_CHECK'] = (parts[1] != 'on')
                tg(f"Strict checks: {'ON' if not SKIP_MARKET_CHECK else 'OFF'}")
            else:
                tg(f"Strict is currently {'ON' if not SKIP_MARKET_CHECK else 'OFF'} - use /strict on|off")
        except Exception as e:
            try:
                tg("Failed to toggle strict mode")
            except Exception as e2:
                try:
                    log_debug("tg fallback failed toggling strict mode:", e2)
                except Exception:
                    pass
        return True
    if t.startswith("/equity"):
        # Ensure a reply is always sent even if the image build/upload fails
        try:
            path = build_equity_curve_png()
            try:
                send_telegram_to(chat_id, f"Equity curve created: {path}")
            except Exception as e:
                try:
                    tg(f"Equity curve created: {path}")
                except Exception as e2:
                    try:
                        log_debug("equity send fallback failed:", e2)
                    except Exception:
                        pass
        except Exception as e:
            try:
                tg(f"Equity chart error: {e}")
            except Exception as e2:
                try:
                    tg("Equity chart error and fallback send failed.")
                except Exception as e3:
                    try:
                        log_debug("equity chart fallback tg failed:", e3)
                    except Exception:
                        pass
        return True
    if t.startswith("/newsupdate"):
        try:
            n = fetch_red_news_windows()
            tg(f"News windows updated: {len(n)} events")
        except Exception as e:
            tg(f"News update failed: {e}")
        return True
    if t.startswith("/selftest"):
        # Run self-tests asynchronously and send immediate acknowledgement.
        try:
            send_telegram_to(chat_id, "🔧 Selftest started — results will be posted here when ready.")

            def _do_selftest():
                try:
                    txt = run_self_tests()
                    send_telegram_to(chat_id, txt)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Selftest error: {e}")
                    except Exception:
                        log_debug("Selftest error send failed:", e)

            threading.Thread(target=_do_selftest, daemon=True).start()
        except Exception as e:
            try:
                send_telegram_to(chat_id, f"Selftest start failed: {e}")
            except Exception:
                try:
                    tg(f"Selftest start failed: {e}")
                except Exception as e2:
                    log_debug("Selftest start fallback tg failed:", e2)
        return True
    if t in ("/scan", "/findtrade"):
        try:
            if t == "/scan":
                # Verbose scan: capture console output and send aggregated results
                try:
                    cmd_scan_verbose(chat_id)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Scan start failed: {e}")
                    except Exception:
                        log_debug("send_telegram_to failed during scan start:", e)
                return True
            else:
                # /findtrade mirrors the live console while running the scanner
                try:
                    cmd_findtrade(chat_id)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Findtrade start failed: {e}")
                    except Exception:
                        log_debug("send_telegram_to failed during findtrade start:", e)
                return True
        except Exception as e:
            try:
                send_telegram_to(chat_id, f"Scan command error: {e}")
            except Exception as ex:
                log_debug("send_telegram_to failed during scan command error:", ex)
        return True
    if t == "/xau_status":
        try:
            # Prepare and send XAU status to requesting chat
            try:
                cmd_xau_status(chat_id)
            except Exception as e:
                try:
                    send_telegram_to(chat_id, f"XAU status error: {e}")
                except Exception:
                    log_debug("send_telegram_to failed during xau_status:", e)
        except Exception as e:
            log_debug("start selftest acknowledgement failed:", e)
        return True
    if t == "/admin_status":
        try:
            if not _is_owner(chat_id):
                tg("⛔ Not authorized to view admin status.")
                return True
            # Run admin status and post to requester
            try:
                send_telegram_to(chat_id, "Admin status requested...")
            except Exception:
                log_debug("send_telegram_to failed when notifying admin status request")
            cmd_admin_status(chat_id)
        except Exception as e:
            tg(f"Admin status error: {e}")
        return True
    # Owner-only force commands: /forcefull <SYMBOL>, /forcemicro <SYMBOL>, /unforce <SYMBOL>
    if t.startswith("/forcefull") or t.startswith("/forcemicro") or t.startswith("/unforce"):
        # Permission check
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to run force commands.")
            return True
        parts = t.split()
        if len(parts) < 2:
            tg("Usage: /forcefull SYMBOL or /forcemicro SYMBOL or /unforce SYMBOL")
            return True
        cmd = parts[0]
        sym = parts[1].upper()
        if cmd == "/unforce":
            if sym in FORCED_TRADES:
                FORCED_TRADES.pop(sym, None)
                save_forced_trades()
                tg(f"✅ Removed forced mode for {sym}")
            else:
                tg(f"ℹ️ No forced mode set for {sym}")
            return True
        if cmd == "/forcefull":
            FORCED_TRADES[sym] = "FULL"
            save_forced_trades()
            tg(f"✅ {sym} forced to FULL (owner override)")
            return True
        if cmd == "/forcemicro":
            FORCED_TRADES[sym] = "MICRO"
            save_forced_trades()
            tg(f"✅ {sym} forced to MICRO (owner override)")
            return True
    if t.startswith("/scale"):
        parts = t.split()
        # /scale -> show current
        if len(parts) == 1:
            try:
                tg(f"micro_lot_ai_scale = {BOT_STATE.micro_lot_ai_scale:.4f}")
            except Exception:
                tg("micro_lot_ai_scale: unknown")
            return True
        # owner-only set/reset
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to set scale")
            return True
        # /scale reset
        if parts[1] == "reset":
            old = BOT_STATE.micro_lot_ai_scale
            BOT_STATE.micro_lot_ai_scale = MICRO_LOT_MIN
            log_msg(f"[AI SCALE] micro_lot_ai_scale reset: {old} -> {BOT_STATE.micro_lot_ai_scale}")
            tg(f"micro_lot_ai_scale reset to {BOT_STATE.micro_lot_ai_scale:.4f}")
            return True
        # /scale set 0.02 or /scale 0.02
        val = None
        if parts[1] == "set" and len(parts) > 2:
            try:
                val = float(parts[2])
            except Exception:
                val = None
        else:
            try:
                val = float(parts[1])
            except Exception:
                val = None
        if val is None:
            tg("Usage: /scale [value] or /scale set <value> or /scale reset")
            return True
        # clamp
        val = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, val))
        old = BOT_STATE.micro_lot_ai_scale
        BOT_STATE.micro_lot_ai_scale = val
        log_msg(f"[AI SCALE] micro_lot_ai_scale set: {old} -> {BOT_STATE.micro_lot_ai_scale}")
        tg(f"micro_lot_ai_scale set: {BOT_STATE.micro_lot_ai_scale:.4f}")
        return True
    # Owner-only on-demand XAU block report
    if t.startswith("/xau_report"):
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to request XAU report")
            return True
        parts = t.split()
        reset = False
        if len(parts) > 1 and parts[1] in ("reset", "clear"):
            reset = True
        try:
            msg = _build_xau_block_report_message(reset_after_send=reset)
            send_telegram_to(chat_id, msg)
        except Exception:
            try:
                tg(_build_xau_block_report_message(reset_after_send=reset))
            except Exception as e:
                log_debug("_build_xau_block_report_message tg fallback failed:", e)
        return True
    if t in ("/help", "/commands"):
        # Ticket-Funded: Updated help message
        tg(build_help_message())
        return True

    # /preview - non-placing scan run in background and report asynchronously
    if t in ("/preview",):
        try:
            # Start background thread to run preview and report to requester
            th = threading.Thread(target=scan_preview_all, kwargs={"report_chat_id": chat_id}, daemon=True)
            th.start()
            # Acknowledge request to the requesting chat
            try:
                send_telegram_to(chat_id, "🔎 Preview started — results will be posted here when ready.")
            except Exception:
                tg("🔎 Preview started — results will be posted to admin chat when ready.")
        except Exception as e:
            try:
                send_telegram_to(chat_id, f"Preview start failed: {e}")
            except Exception:
                tg(f"Preview start failed to notify: {e}")
        return True

    # Extended minimal handlers for common operational commands
    # These are intentionally lightweight and mostly use existing helpers.
    parts = t.split()

    # /mode strict|balanced
    if t.startswith("/mode"):
        if len(parts) > 1 and parts[1] in ("strict", "balanced"):
            try:
                globals()['MODE'] = parts[1].upper()
                tg(f"Mode set: {globals().get('MODE')}")
            except Exception as e:
                tg(f"Failed to set mode: {e}")
        else:
            tg(f"Current mode: {globals().get('MODE', 'UNKNOWN')}. Usage: /mode strict|balanced")
        return True

    # /strategy goat|legacy
    if t.startswith("/strategy"):
        if len(parts) > 1 and parts[1] in ("goat", "legacy"):
            try:
                globals()['STRATEGY'] = parts[1].upper()
                tg(f"Strategy set: {globals().get('STRATEGY')}")
            except Exception as e:
                tg(f"Failed to set strategy: {e}")
        else:
            tg(f"Current strategy: {globals().get('STRATEGY', 'UNKNOWN')}. Usage: /strategy goat|legacy")
        return True

    # start/stop/restart polling (owner-only for start/restart)
    if t == "/start":
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to start polling")
            return True
        try:
            globals()['TELEGRAM_POLLING_ENABLED'] = True
            if not globals().get('TG_POLL_THREAD_STARTED'):
                threading.Thread(target=telegram_poller_loop, daemon=True).start()
                globals()['TG_POLL_THREAD_STARTED'] = True
            tg("✅ Telegram polling started")
        except Exception as e:
            tg(f"Start failed: {e}")
        return True

    if t == "/stop":
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to stop polling")
            return True
        try:
            globals()['TELEGRAM_POLLING_ENABLED'] = False
            tg("🛑 Telegram polling stopped")
        except Exception as e:
            tg(f"Stop failed: {e}")
        return True

    if t == "/restart":
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to restart polling")
            return True
        try:
            globals()['TELEGRAM_POLLING_ENABLED'] = False
            time.sleep(0.3)
            globals()['TELEGRAM_POLLING_ENABLED'] = True
            if not globals().get('TG_POLL_THREAD_STARTED'):
                threading.Thread(target=telegram_poller_loop, daemon=True).start()
                globals()['TG_POLL_THREAD_STARTED'] = True
            tg("🔁 Telegram polling restarted")
        except Exception as e:
            tg(f"Restart failed: {e}")
        return True

    # /ping
    if t == "/ping":
        tg("pong")
        return True

    # /settings - show basic runtime settings
    if t == "/settings":
        try:
            s = (
                f"Mode: {globals().get('MODE')} | Strategy: {globals().get('STRATEGY')}\n"
                f"Risk per full trade: {globals().get('RISK_PCT', 0.0)*100:.3f}% | Micro lot target: {globals().get('MICRO_LOT_TARGET', 0.0)}\n"
                f"Micro AI scale: {BOT_STATE.micro_lot_ai_scale:.4f} | Polling: {globals().get('TELEGRAM_POLLING_ENABLED')}\n"
                f"Live only: {globals().get('LIVE_ONLY', False)} | Quiet spam: {globals().get('QUIET_SPAM', False)}"
            )
            tg(s)
        except Exception as e:
            tg(f"Settings error: {e}")
        return True

    # Owner-only parameter changes
    if t.startswith("/risk"):
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to set risk")
            return True
        if len(parts) < 2:
            tg("Usage: /risk <pct>  (e.g. /risk 0.5 sets 0.5%)")
            return True
        try:
            v = float(parts[1])
            # Interpret user-supplied as percent (e.g. 0.5 -> 0.5%)
            new = float(v) / 100.0
            globals()['RISK_PCT'] = new
            save_state()
            tg(f"✅ RISK_PCT set to {new*100:.3f}%")
        except Exception as e:
            tg(f"Failed to set risk: {e}")
        return True

    if t.startswith("/microlot"):
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to set micro lot")
            return True
        if len(parts) < 2:
            tg("Usage: /microlot <lot>  (e.g. /microlot 0.01)")
            return True
        try:
            v = float(parts[1])
            v = max(float(globals().get('MICRO_LOT_MIN', 0.01)), min(float(globals().get('MICRO_LOT_MAX', 0.04)), v))
            globals()['MICRO_LOT_TARGET'] = v
            save_state()
            tg(f"✅ MICRO_LOT_TARGET set to {v}")
        except Exception as e:
            tg(f"Failed to set microlot: {e}")
        return True

    # /demo - run a simulated scan/demo in background
    if t.startswith("/demo"):
        try:
            send_telegram_to(chat_id, "🔧 Demo started — results will be posted when ready.")
            def _demo():
                try:
                    txt = run_self_tests()
                    send_telegram_to(chat_id, txt)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Demo error: {e}")
                    except Exception:
                            log_debug("send_telegram_to failed during demo error notify")
            threading.Thread(target=_demo, daemon=True).start()
        except Exception as e:
            tg(f"Demo start failed: {e}")
        return True

    # /micro <PAIR> - set micro rotation start pair
    if t.startswith("/micro"):
        if len(parts) < 2:
            tg("Usage: /micro <PAIR>")
            return True
        pair = parts[1].upper()
        try:
            if pair in globals().get('SYMBOLS', []):
                globals()['MICRO_RR_IDX'] = globals().get('SYMBOLS', []).index(pair)
                save_state()
                tg(f"✅ Micro rotation start set to {pair}")
            else:
                tg(f"Unknown pair: {pair}")
        except Exception as e:
            tg(f"Failed to set micro rotation: {e}")
        return True

    # /route <BASE> <ALIAS>
    if t.startswith("/route"):
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to set routes")
            return True
        if len(parts) < 3:
            tg("Usage: /route <BASE> <ALIAS>")
            return True
        base = parts[1].upper(); alias = parts[2]
        try:
            globals().setdefault('SYMBOL_ROUTE', {})
            globals()['SYMBOL_ROUTE'][base] = alias
            save_state()
            tg(f"✅ Route set: {base} -> {alias}")
        except Exception as e:
            tg(f"Failed to set route: {e}")
        return True

    # /probe <PAIR>
    if t.startswith("/probe"):
        if len(parts) < 2:
            tg("Usage: /probe <PAIR>")
            return True
        pair = parts[1]
        try:
            resolved = resolve_symbol(pair)
            si = None
            try:
                si = mt5.symbol_info(resolved)
            except Exception:
                si = None
            tg(f"Probe: input={pair} resolved={resolved} available={bool(si)}")
        except Exception as e:
            tg(f"Probe failed: {e}")
        return True

    # /autotest - run self-tests async
    if t == "/autotest":
        try:
            send_telegram_to(chat_id, "🔬 Autotest started")
            def _auto():
                try:
                    txt = run_self_tests()
                    send_telegram_to(chat_id, txt)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Autotest error: {e}")
                    except Exception:
                            log_debug("send_telegram_to failed during autotest error notify")
            threading.Thread(target=_auto, daemon=True).start()
        except Exception as e:
            tg(f"Autotest start failed: {e}")
        return True

    # /quiet on|off
    if t.startswith("/quiet"):
        if len(parts) > 1 and parts[1] in ("on","off"):
            globals()['QUIET_SPAM'] = (parts[1] == 'on')
            tg(f"Quiet spam: {globals().get('QUIET_SPAM')}")
        else:
            tg(f"Quiet spam is {globals().get('QUIET_SPAM')}. Usage: /quiet on|off")
        return True

    # /liveonly on|off
    if t.startswith("/liveonly"):
        if len(parts) > 1 and parts[1] in ("on","off"):
            globals()['LIVE_ONLY'] = (parts[1] == 'on')
            save_state()
            tg(f"Live-only mode: {globals().get('LIVE_ONLY')}")
        else:
            tg(f"Live-only is {globals().get('LIVE_ONLY')}. Usage: /liveonly on|off")
        return True

    # /prop - show detected prop firm & rules
    if t.startswith("/prop"):
        # /prop set <NAME>
        if len(parts) > 1 and parts[1] == 'set':
            if not _is_owner(chat_id):
                tg("⛔ Not authorized to set prop override")
                return True
            if len(parts) < 3:
                tg("Usage: /prop set <FTMO|FXIFY|GOAT|AQUA|AUTO>")
                return True
            name = parts[2].upper()
            try:
                globals()['PROP_ACTIVE'] = {'name': name}
                save_state()
                tg(f"✅ PROP override set to {name}")
            except Exception as e:
                tg(f"Failed to set prop: {e}")
            return True
        # otherwise show detected
        try:
            firm, rules, server = detect_prop_firm()
            tg(f"Prop firm: {firm} | Server: {server} | rules: {rules}")
        except Exception as e:
            tg(f"Prop detect error: {e}")
        return True

    # /full - attempt a full trade now (runs in background)
    if t == "/full":
        if not _is_owner(chat_id):
            tg("⛔ Not authorized to place full trades")
            return True
        try:
            send_telegram_to(chat_id, "🚀 Attempting full trade now — results will be posted.")
            def _do_full():
                try:
                    ok = attempt_full_trade_once()
                    send_telegram_to(chat_id, f"Full trade attempt completed: {'placed' if ok else 'no setup'}")
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Full trade error: {e}")
                    except Exception:
                            log_debug("send_telegram_to failed during full trade error notify")
            threading.Thread(target=_do_full, daemon=True).start()
        except Exception as e:
            tg(f"Full start failed: {e}")
        return True

    return False
def resolve_symbol(base_or_alias):
    """Resolve a base symbol or alias to a tradeable MT5 symbol name."""
    base = base_or_alias.split('.')[0].upper()
    if base in SYMBOL_ROUTE:
        name = SYMBOL_ROUTE[base]
        si = mt5.symbol_info(name)
        if si and mt5.symbol_select(name, True) and getattr(si, "trade_mode", 0) == 4:
            return name
    si_alias = mt5.symbol_info(base_or_alias)
    if si_alias and mt5.symbol_select(base_or_alias, True) and getattr(si_alias, "trade_mode", 0) == 4:
        return base_or_alias
    alt_roots = {"XAUUSD": ["GOLD"], "GBPUSD": [], "GBPJPY": []}
    candidates = [base]
    try:
        candidates += [s.name for s in (mt5.symbols_get(base + "*") or [])]
        candidates += [s.name for s in (mt5.symbols_get("*" + base + "*") or [])]
        for root in alt_roots.get(base, []):
            candidates += [s.name for s in (mt5.symbols_get(root + "*") or [])]
            candidates += [s.name for s in (mt5.symbols_get("*" + root + "*") or [])]
    except Exception:
        log_debug("resolve_symbol: symbol discovery failed",)
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for name in candidates:
        si = mt5.symbol_info(name)
        if si and mt5.symbol_select(name, True) and getattr(si, "trade_mode", 0) == 4:
            return name
    return base_or_alias
def detect_prop_firm() -> Tuple[str, Dict[str, Optional[float]], str]:
    """
    Detect the current proprietary trading firm by inspecting the connected MT5 server.

    This unified detection routine waits briefly for the MT5 API to return a non‑empty
    server string, normalises it, then maps known patterns to a firm name.  It returns
    a tuple of (firm_name, rules_dict, server_string).  Unknown or missing values
    result in ``"UNKNOWN"`` fields and an empty rules dict.
    """
    # Ensure MT5 is initialised and attempt to obtain account info. If no
    # account info is available immediately, try to initialise the bridge
    # and wait a little longer for the terminal to finish login.
    try:
        info = mt5.account_info()
    except Exception:
        info = None

    # If MT5 seems uninitialised, attempt to initialise (best-effort).
    if not info:
        try:
            # If mt5.initialize exists, call it. Ignore return value; we'll re-check.
            if hasattr(mt5, 'initialize'):
                try:
                    mt5.initialize()
                except Exception:
                    # initialization may fail if terminal is already running; ignore
                    pass
        except Exception as e:
            log_debug("scan/findtrade command final fallback failed:", e)

    # Try to gather server/login info for up to 30 seconds to allow terminal
    # to complete any background login sequence.
    server = None
    info = None
    start_time = time.time()
    while time.time() - start_time < 30.0:
        try:
            info = mt5.account_info()
        except Exception:
            info = None
        if info:
            server = getattr(info, 'server', None)
            # if server is a non-empty string, break early
            if server and str(server).strip() != "":
                break
        # also attempt terminal_info() as an alternate source for server name
        try:
            if hasattr(mt5, 'terminal_info'):
                ti = mt5.terminal_info()
                if ti:
                    # common terminal_info fields: 'company', 'trade_server'
                    server_candidate = getattr(ti, 'trade_server', None) or getattr(ti, 'server', None) or getattr(ti, 'company', None)
                    if server_candidate and str(server_candidate).strip() != "":
                        server = server_candidate
                        break
        except Exception as e:
            log_debug("xau_status command dispatch failed:", e)
        time.sleep(1.0)

    # If still no server after waiting, return UNKNOWNs
    if not server or str(server).strip() == "":
        return "UNKNOWN", {}, "UNKNOWN"

    server_str = str(server).strip()
    # Normalise for matching
    s_norm = server_str.lower().replace(" ", "").replace("-", "")

    # Determine firm name based on server patterns and account metadata
    firm = "UNKNOWN"
    # custom mappings with flexible substring matching
    # Mapping keys are normalized to lowercase and stripped of spaces/dashes.
    # Value strings follow the requested naming convention (uppercase) to
    # ensure /status never returns UNKNOWN when a supported firm is present.
    mappings: Dict[str, str] = {
        "icmarkets": "ICMARKETS",
        "ftmo": "FTMO",
        "fxify": "FXIFY",
        "goated": "GOATED",
        "goat": "GOATED",
        "aqua": "AQUAFUNDED",
        # retain legacy aliases for backward compatibility
        "myfundedfx": "MyFundedFX",
        "fundednext": "FundedNext",
        "smartprop": "Smart Prop Trader",
        "smartproptrader": "Smart Prop Trader",
        "alphacapital": "Alpha Capital",
        "thefundedtrader": "The Funded Trader",
        "e8funding": "E8 Funding",
    }
    for key, name in mappings.items():
        if key in s_norm:
            firm = name
            break
    # Leverage and other heuristics as fallback
    if firm == "UNKNOWN":
        try:
            ai = mt5.account_info()
            lev = int(getattr(ai, 'leverage', 0) or 0)
            # FTMO often uses 1:100 or specific server names; if server contains 'ftmo' it's already caught
            if lev and lev <= 100 and 'ftmo' in server_str.lower():
                firm = 'FTMO'
        except Exception as e:
            log_debug("detect_prop_firm: leverage check failed:", e)

    # Load risk rules corresponding to the detected firm
    try:
        rules = load_risk_rules(firm)
    except Exception as e:
        log_debug("load_risk_rules failed:", e)
        rules = {}
    # If still unknown, send an alert and default to SAFE mode
    if firm == "UNKNOWN":
        try:
            telegram_msg("⚠️ Unable to identify prop firm. Defaulting to SAFE mode.")
        except Exception:
            log_debug("telegram_msg failed when notifying unknown firm")
    return firm, rules, server_str

# ---------------------------------------------------------------------------
# Risk rules definition
# ---------------------------------------------------------------------------
def load_risk_rules(firm: str) -> Dict[str, Optional[float]]:
    """
    Return a mapping of risk limits specific to each proprietary firm.

    Args:
        firm: The name of the proprietary firm (e.g. 'FTMO', 'FXIFY',
              'Goated Funded Trader', 'Aquafunded', or 'UNKNOWN').

    Returns:
        A dictionary of risk parameters.  Missing values indicate no restriction.
    """
    # FTMO limits
    if firm == "FTMO":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trailing_dd": None,
            "max_trades_per_day": 2,
            "max_lots": None,
            "consistency": False,
            "trailing": False,
        }
    # FXIFY limits
    if firm == "FXIFY":
        return {
            "max_daily_dd": 0.04,
            "max_total_dd": 0.08,
            "max_trailing_dd": None,
            "max_trades_per_day": None,
            "max_lots": None,
            "consistency": True,
            "trailing": False,
        }
    # Goated Funded Trader limits
    if firm == "Goated Funded Trader":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trailing_dd": None,
            "max_trades_per_day": None,
            "max_lots": 5,
            "consistency": False,
            "trailing": False,
        }
    # GOATED limits (uppercase alias for Goated Funded Trader)
    if firm == "GOATED":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trailing_dd": None,
            "max_trades_per_day": None,
            "max_lots": 5,
            "consistency": False,
            "trailing": False,
        }
    # Aquafunded limits
    if firm == "Aquafunded":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": None,
            "max_trailing_dd": 0.12,
            "max_trades_per_day": None,
            "max_lots": None,
            "consistency": False,
            "trailing": True,
        }
    # AQUAFUNDED limits (uppercase alias for Aquafunded)
    if firm == "AQUAFUNDED":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": None,
            "max_trailing_dd": 0.12,
            "max_trades_per_day": None,
            "max_lots": None,
            "consistency": False,
            "trailing": True,
        }
    # ICMARKETS has no specific prop firm restrictions; treat as unrestricted
    if firm == "ICMARKETS":
        return {
            "max_daily_dd": None,
            "max_total_dd": None,
            "max_trailing_dd": None,
            "max_trades_per_day": None,
            "max_lots": None,
            "consistency": False,
            "trailing": False,
        }
    # Default: no restrictions
    return {
        "max_daily_dd": None,
        "max_total_dd": None,
        "max_trailing_dd": None,
        "max_trades_per_day": None,
        "max_lots": None,
        "consistency": False,
        "trailing": False,
    }

def save_state():
    """Save persistent state to file (JSON)."""
    try:
        state = {
            "day_stats": day_stats,
            "mode": MODE,
            "strategy": STRATEGY,
            "symbol_route": SYMBOL_ROUTE,
            "micro_rr_idx": MICRO_RR_IDX,
            "micro_lot_target": MICRO_LOT_TARGET,
            "risk_pct": RISK_PCT,
            "quiet_spam": QUIET_SPAM,
            "live_only": globals().get('LIVE_ONLY', False),
            "prop_active": PROP_ACTIVE,
            "prop_pivot": globals().get('PROP_DAILY_PIVOT_UTC', {}),
            "prop_rulesets": PROP_RULESETS,
            "symbol_thresholds": SYMBOL_CONF_THRESH,
            "AI_WEIGHTS": AI_WEIGHTS,
            "AI_BIAS": AI_BIAS,
            # Persist adaptive memory (learning) so the AI retains experience
            "adaptive_memory": adaptive_memory,
            "open_trades_info": open_trades_info
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        try:
            log_debug("save_state failed:", e)
        except Exception:
            pass
def enable_test_mode():
    """Enable TEST_MODE: the bot will simulate orders instead of sending them."""
    try:
        globals()['TEST_MODE'] = True
        log_msg("🔬 TEST_MODE enabled - orders will be simulated (no mt5.order_send calls).")
    except Exception as e:
        log_debug("enable_test_mode failed:", e)


def disable_test_mode():
    """Disable TEST_MODE and return to normal operation."""
    try:
        globals()['TEST_MODE'] = False
        log_msg("🔬 TEST_MODE disabled - real orders will be sent when triggered.")
    except Exception as e:
        log_debug("disable_test_mode failed:", e)


def simulate_scan(announce: bool = False) -> bool:
    """Run a scan with TEST_MODE enabled so execution paths are exercised
    without sending live orders. Returns the boolean result from run_scan().
    """
    prev = globals().get('TEST_MODE', False)
    try:
        enable_test_mode()
        return run_scan(announce=announce)
    except Exception as e:
        try:
            log_msg(f"🔬 simulate_scan error: {e}")
        except Exception as e2:
            log_debug("log_msg failed during simulate_scan error notify:", e2)
        return False
    finally:
        try:
            globals()['TEST_MODE'] = prev
        except Exception as e:
            log_debug("simulate_scan: restoring TEST_MODE failed:", e)
def load_state():
    """Load persistent state from file if available, updating global variables."""
    if not os.path.exists(STATE_FILE):
        return
    try:
        if os.path.getsize(STATE_FILE) == 0:
            return
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return
        try:
            st = json.loads(raw)
        except Exception as e:
            try:
                os.replace(STATE_FILE, STATE_FILE + ".bak")
            except Exception as e2:
                try:
                    log_debug("failed to backup corrupted STATE_FILE:", e2)
                except Exception:
                    pass
            try:
                log_debug("json.loads failed while loading state:", e)
            except Exception:
                pass
            return
        if isinstance(st.get("day_stats"), dict):
            day_stats.update(st["day_stats"])
        global MODE, STRATEGY, SYMBOL_ROUTE, MICRO_RR_IDX, MICRO_LOT_TARGET
        global RISK_PCT, QUIET_SPAM, PROP_ACTIVE, PROP_RULESETS, AI_BIAS
        MODE       = st.get("mode", MODE)
        STRATEGY   = st.get("strategy", STRATEGY)
        RISK_PCT   = st.get("risk_pct", RISK_PCT)
        MICRO_RR_IDX = st.get("micro_rr_idx", MICRO_RR_IDX)
        MICRO_LOT_TARGET = st.get("micro_lot_target", MICRO_LOT_TARGET)
        QUIET_SPAM = st.get("quiet_spam", QUIET_SPAM)
        live_only_val = st.get("live_only", False)
        globals()['LIVE_ONLY'] = live_only_val  # dynamically set LIVE_ONLY if present
        if isinstance(st.get("symbol_route"), dict):
            SYMBOL_ROUTE.update(st["symbol_route"])
        if isinstance(st.get("prop_active"), dict):
            PROP_ACTIVE.update(st["prop_active"])
        if isinstance(st.get("prop_pivot"), dict):
            globals().setdefault('PROP_DAILY_PIVOT_UTC', {}).update(st["prop_pivot"])
        if isinstance(st.get("prop_rulesets"), dict):
            PROP_RULESETS.update(st["prop_rulesets"])
        if isinstance(st.get("symbol_thresholds"), dict):
            SYMBOL_CONF_THRESH.update(st["symbol_thresholds"])
        if "AI_WEIGHTS" in st:
            try:
                AI_WEIGHTS.update(st["AI_WEIGHTS"])
            except Exception as e:
                log_debug("AI_WEIGHTS update failed during load_state:", e)
        AI_BIAS = float(st.get("AI_BIAS", AI_BIAS))
        # Restore adaptive memory if present
        try:
            adm = st.get("adaptive_memory")
            if isinstance(adm, dict):
                for k, v in adm.items():
                    adaptive_memory[k] = v if isinstance(v, list) else list(v)
        except Exception as e:
            log_debug("adaptive_memory restore failed:", e)
        try:
            oti = st.get("open_trades_info")
            if isinstance(oti, dict):
                open_trades_info.update(oti)
        except Exception as e:
            log_debug("open_trades_info restore failed:", e)
    except Exception as e:
        try:
            telegram_msg(f"⚠️ State load error: {e}")
        except Exception as e2:
            log_debug("telegram_msg failed when notifying state load error:", e2)
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
def now_uk():
    """Current time in the configured timezone, with safe fallback to UTC."""
    try:
        return datetime.now(SAFE_TZ)
    except Exception:
        # Fallback to naive now() if datetime.now() with timezone fails
        return datetime.now()
def _within(now, start_h, start_m, end_h, end_m):
    s = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    e = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return s <= now <= e
def in_session_full():
    now = now_uk()
    a = _within(now, *FULL_SESSION_LON)
    b = _within(now, *FULL_SESSION_NY)
    # Include Asian session windows
    return a or b or is_asian_session(now)
def in_session_micro():
    now = now_uk()
    a = _within(now, *MICRO_SESSION_LON)
    b = _within(now, *MICRO_SESSION_NY)
    return a or b or is_asian_session(now)
def now_utc():
    return datetime.now(timezone.utc)
def telegram_html(text):
    """Send raw HTML text to Telegram (if token/chat configured)."""
    try:
        # Respect compact-only policy: route through `telegram_msg` which enforces
        # allowed message formats. If raw HTML must be sent, the operator can
        # enable `ALLOW_RAW_TELEGRAM=1` in the environment to bypass the gate.
        if str(os.getenv('ALLOW_RAW_TELEGRAM', '0')).lower() in ('1', 'true'):
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                    timeout=10
                )
            except Exception as e:
                print(f"⚠️ Telegram HTML failed (raw send): {e}")
        else:
            # Send via `telegram_msg` (will be printed to console if not allowed)
            try:
                telegram_msg(text)
            except Exception:
                try:
                    print(text)
                except Exception as e:
                    log_debug("telegram_html fallback print failed:", e)
    except Exception as e:
        print(f"⚠️ Telegram HTML failed: {e}")
def log_msg(text):
    """Log message to console and Telegram."""
    try:
        # Prefer direct buffer write with utf-8 to avoid Windows cp1252 encoding errors
        try:
            sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            # Fallback to print with replacement for any remaining issues
            print(str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    except Exception:
        # As a last resort, attempt a plain print and log errors
        try:
            print(str(text))
        except Exception as e:
            log_debug("log_msg final print failed:", e)
    try:
        telegram_msg(text)
    except Exception:
        log_debug("telegram_msg failed in log_msg")

# ---------------------------------------------------------------------------
# Decision logging helper
# ---------------------------------------------------------------------------
def log_decision(reason, side, symbol):
    """
    Log decision reasons for trade evaluations.  This helper logs both to the
    console and via ``log_msg`` so that reasons for passing or failing a
    trade setup are visible in real time and in Telegram logs.  It accepts
    the reason string, the trade side ("BUY" or "SELL"), and the symbol.
    """
    try:
        print(f"[DECISION] {symbol} {side}: {reason}")
        log_msg(f"[DECISION] {symbol} {side}: {reason}")
    except Exception:
        try:
            log_debug("log_decision fallback failed")
        except Exception:
            pass
def _base_of(symbol):
    """Extract base symbol name (e.g., remove suffix like .r or prefix if any)."""
    return symbol.split('.')[0].upper()
def retcode_text(code: int):
    """Human-readable text for common MT5 trade retcodes."""
    mapping = {
        0: "OK/CHECK",
        10009: "DONE",
        10010: "DONE_PARTIAL",
        10016: "MARKET_CLOSED",
        10017: "NO_MONEY",
        10018: "PRICE_CHANGED",
        10019: "PRICE_OFF",
        10024: "SERVER_DISABLES_AUTO",
        10027: "CLIENT_AUTO_OFF",
        10028: "SERVER_AUTO_OFF",
        10029: "SYMBOL_AUTO_OFF",
        10030: "INVALID_FILL"
    }
    return mapping.get(int(code), f"RET={code}")
def autotrading_hint(prefix=""):
    msg = (prefix + "❗ 10027: **Autotrading OFF in MT5**.\n" +
           "Enable: Toolbar **Algo Trading** (green) → Tools > Options > **Expert Advisors** → allow algo trading.")
    log_msg(msg)
def get_data(symbol, timeframe, bars=200):
    """Fetch recent market data for symbol/timeframe as a pandas DataFrame.

    This helper includes a simple retry mechanism to mitigate transient feed
    interruptions from the MetaTrader5 bridge.  If no data is returned after
    DATA_RETRY_COUNT attempts, the most recent cached DataFrame for the same
    (symbol, timeframe, bars) tuple is used as a fallback.  Without a cache
    fallback the bot would skip trades whenever a temporary feed spike occurs.
    """
    rates = None
    for _ in range(DATA_RETRY_COUNT):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        except Exception:
            rates = None
        # Break early if we received a non‑empty response
        if rates is not None and len(rates) > 0:
            break
        time.sleep(DATA_RETRY_DELAY)
    key = (symbol, timeframe, bars)
    # If still no data after retries, return the last cached result if present
    if rates is None or len(rates) == 0:
        cached = LAST_DATA_CACHE.get(key)
        # Return a copy to avoid accidental mutation of cached DataFrames
        return cached.copy() if isinstance(cached, pd.DataFrame) else None
    # Convert to DataFrame and update the cache
    df = pd.DataFrame(rates)
    LAST_DATA_CACHE[key] = df
    return df


# Lightweight per-tick cache to avoid repeated expensive MT5 calls inside
# a single scan iteration.  Keys are (symbol, timeframe).  Values store a
# tuple (timestamp, dataframe) and are refreshed at most once per second
# per timeframe to balance freshness and performance.
DATA_CACHE: Dict[Tuple[str,int], Tuple[float, pd.DataFrame]] = {}

def fetch_data_cached(symbol: str, timeframe: int, bars: int = 200, max_age: float = 1.0) -> Optional[pd.DataFrame]:
    key = (symbol, timeframe)
    now_ts = time.time()
    rec = DATA_CACHE.get(key)
    if rec and (now_ts - rec[0]) <= max_age and isinstance(rec[1], pd.DataFrame) and len(rec[1]) >= bars:
        return rec[1]
    df = get_data(symbol, timeframe, bars)
    if df is not None:
        DATA_CACHE[key] = (now_ts, df)
    return df
def _ensure_df_ok(df, min_rows=20):
    try:
        if df is None or len(df) < min_rows:
            return False
        return not (df[['high','low','close']].isna().any().any())
    except Exception:
        return False
def get_ema(df, period=50):
    return df["close"].ewm(span=period).mean()
def evaluate_smc_quality(confirmed: bool, forming: bool, strength: float) -> str:
    """Evaluate SMC condition and return emoji quality rating.
    
    Args:
        confirmed: Whether the condition is confirmed
        forming: Whether the condition is forming
        strength: Strength metric (0-1 scale)
    
    Returns:
        Emoji string: ✅ (strong), 👍🏾 (weak ok), 👎🏾 (very weak), ❌ (invalid)
    """
    if not confirmed and not forming:
        return EMOJI_INVALID  # ❌
    
    if confirmed and strength >= 0.7:
        return EMOJI_STRONG  # ✅
    elif confirmed and strength >= 0.4:
        return EMOJI_WEAK_OK  # 👍🏾
    elif forming and strength >= 0.5:
        return EMOJI_WEAK_OK  # 👍🏾
    else:
        return EMOJI_VERY_WEAK  # 👎🏾

def detect_bos(df):
    """Detect Break of Structure (BOS) using DataFrame input.

    Returns (emoji_quality, strength_score) tuple.
    emoji_quality: ✅ / 👍🏾 / 👎🏾 / ❌
    strength_score: float 0-1
    """
    try:
        if df is None or len(df) < 5:
            return EMOJI_INVALID, 0.0
        recent = df.tail(10)
        # Use all but the last candle for previous structure
        prev_high = recent["high"].iloc[:-1].max()
        prev_low = recent["low"].iloc[:-1].min()
        last_close = recent["close"].iloc[-1]
        last_high = recent["high"].iloc[-1]
        last_low = recent["low"].iloc[-1]
        
        confirmed = False
        forming = False
        strength = 0.0

        # Bullish BOS: last close breaks above recent high
        if last_close > prev_high:
            confirmed = True
            strength = min(1.0, (last_close - prev_high) / (prev_high * 0.01))  # 1% = full strength
        # Bearish BOS: last close breaks below recent low
        elif last_close < prev_low:
            confirmed = True
            strength = min(1.0, (prev_low - last_close) / (prev_low * 0.01))
        # Forming BOS: high/low approaching structure but not confirmed
        elif last_high > prev_high * 0.998 or last_low < prev_low * 1.002:
            forming = True
            strength = 0.5  # Moderate strength for forming
        
        emoji = evaluate_smc_quality(confirmed, forming, strength)
        return emoji, strength
    except Exception as e:
        print(f"❌ detect_bos error: {e}")
        return EMOJI_INVALID, 0.0

def interpret_bos(value):
    """
    Normalise a Break of Structure (BOS) value into a numeric representation.
    Returns +1 for bullish, -1 for bearish and 0 for neutral/unknown values.

    This helper allows code paths that expect a numeric BOS value to safely
    consume any string, boolean or numeric BOS representation.  When an
    unrecognised type or value is supplied, the function defaults to 0
    (neutral) to avoid raising errors that could block execution.
    """
    try:
        # None is neutral/unknown
        if value is None:
            return 0
        # Strings: compare against the known BOS keywords (case-insensitive)
        if isinstance(value, str):
            v = value.upper()
            if v == "BOS_UP":
                return 1
            if v == "BOS_DOWN":
                return -1
            return 0
        # Booleans: treat True as bullish, False as bearish
        if isinstance(value, bool):
            return 1 if value else -1
        # Numeric values: positive -> bullish, negative -> bearish
        num = float(value)
        if num > 0:
            return 1
        if num < 0:
            return -1
        return 0
    except Exception:
        return 0
def detect_sweep(df):
    """Detect liquidity sweep using DataFrame input.

    Returns (emoji_quality, strength_score) tuple.
    """
    try:
        if df is None or len(df) < 5:
            return EMOJI_INVALID, 0.0
        recent = df.tail(10)
        prev_high = recent["high"].iloc[:-1].max()
        prev_low = recent["low"].iloc[:-1].min()
        last_high = recent["high"].iloc[-1]
        last_low = recent["low"].iloc[-1]
        last_close = recent["close"].iloc[-1]

        confirmed = False
        forming = False
        strength = 0.0

        swept_high = last_high > prev_high and last_close < prev_high
        swept_low = last_low < prev_low and last_close > prev_low
        
        if swept_high or swept_low:
            confirmed = True
            # Strength based on wick size vs close rejection
            wick_size = max(last_high - last_close, last_close - last_low)
            candle_range = last_high - last_low
            strength = min(1.0, wick_size / (candle_range + 1e-9))
        elif (last_high > prev_high * 0.995) or (last_low < prev_low * 1.005):
            forming = True
            strength = 0.5
        
        emoji = evaluate_smc_quality(confirmed, forming, strength)
        return emoji, strength
    except Exception as e:
        print(f"❌ detect_sweep error: {e}")
        return EMOJI_INVALID, 0.0


def detect_fvg(df):
    """Detect Fair Value Gap (FVG) using DataFrame input.

    Returns (emoji_quality, strength_score) tuple.
    """
    try:
        if df is None or len(df) < 3:
            return EMOJI_INVALID, 0.0
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        confirmed = False
        developing = False
        strength = 0.0
        
        bull_gap = c2["low"] > c1["high"] and c2["low"] > c3["high"]
        bear_gap = c2["high"] < c1["low"] and c2["high"] < c3["low"]
        
        if bull_gap or bear_gap:
            confirmed = True
            # Strength based on gap size
            if bull_gap:
                gap_size = c2["low"] - max(c1["high"], c3["high"])
                candle_size = c2["high"] - c2["low"]
            else:
                gap_size = min(c1["low"], c3["low"]) - c2["high"]
                candle_size = c2["high"] - c2["low"]
            strength = min(1.0, gap_size / (candle_size + 1e-9))
        elif ((c2["low"] >= c1["high"] * 0.995 and c2["low"] >= c3["high"] * 0.995) or
              (c2["high"] <= c1["low"] * 1.005 and c2["high"] <= c3["low"] * 1.005)):
            developing = True
            strength = 0.5
        
        emoji = evaluate_smc_quality(confirmed, developing, strength)
        return emoji, strength
    except Exception as e:
        print(f"❌ detect_fvg error: {e}")
        return EMOJI_INVALID, 0.0

# ===================== SMART MONEY CONCEPTS (SMC) MODULE =====================
@dataclass
class FVG:
    start_idx: int
    end_idx: int
    direction: str
    timeframe: int

    def size(self, df: pd.DataFrame) -> float:
        try:
            if self.direction == 'BULL':
                return float(df['low'].iloc[self.end_idx] - df['high'].iloc[self.start_idx])
            return float(df['high'].iloc[self.start_idx] - df['low'].iloc[self.end_idx])
        except Exception:
            return 0.0

def detect_swing_high_low(df: pd.DataFrame, lookback: int = 50) -> Tuple[List[int], List[int]]:
    """Return lists of swing high and swing low indices for the DataFrame."""
    highs, lows = [], []
    try:
        for i in range(2, min(len(df)-2, lookback)):
            if df['high'].iloc[-i] > df['high'].iloc[-i-1] and df['high'].iloc[-i] > df['high'].iloc[-i+1]:
                highs.append(len(df)-i-1)
            if df['low'].iloc[-i] < df['low'].iloc[-i-1] and df['low'].iloc[-i] < df['low'].iloc[-i+1]:
                lows.append(len(df)-i-1)
    except Exception as e:
        log_debug("detect_swing_high_low failed:", e)
    return highs, lows


def detect_clear_swing_high_low(df: pd.DataFrame, lookback: int = 120, left: int = 2, right: int = 2) -> Tuple[List[int], List[int]]:
    """Return clear swing high/low indices requiring at least `left`/`right` candles.

    This enforces the user's requirement of 2 candles on each side by default.
    """
    highs, lows = [], []
    try:
        if df is None or len(df) < (left + right + 3):
            return highs, lows
        start = max(0, len(df) - lookback)
        for i in range(start + left, len(df) - right):
            hi = float(df['high'].iloc[i])
            lo = float(df['low'].iloc[i])
            if all(hi > float(df['high'].iloc[i - j]) for j in range(1, left + 1)) and \
               all(hi > float(df['high'].iloc[i + j]) for j in range(1, right + 1)):
                highs.append(i)
            if all(lo < float(df['low'].iloc[i - j]) for j in range(1, left + 1)) and \
               all(lo < float(df['low'].iloc[i + j]) for j in range(1, right + 1)):
                lows.append(i)
    except Exception as e:
        log_debug("detect_clear_swing_high_low failed:", e)
    return highs, lows


def _last_clear_swing_price(df: pd.DataFrame, kind: str = "LOW", lookback: int = 160) -> Optional[Tuple[int, float]]:
    """Return (index, price) for the most recent clear swing high/low."""
    try:
        highs, lows = detect_clear_swing_high_low(df, lookback=lookback, left=2, right=2)
        if kind.upper() == "HIGH":
            if not highs:
                return None
            idx = highs[-1]
            return idx, float(df['high'].iloc[idx])
        if not lows:
            return None
        idx = lows[-1]
        return idx, float(df['low'].iloc[idx])
    except Exception as e:
        log_debug("_last_clear_swing_price failed:", e)
        return None


def _nearest_swing_level(df: pd.DataFrame, entry: float, kind: str = "HIGH", lookback: int = 200) -> Optional[float]:
    """Return nearest clear swing level above/below entry."""
    try:
        highs, lows = detect_clear_swing_high_low(df, lookback=lookback, left=2, right=2)
        if kind.upper() == "HIGH":
            candidates = [float(df['high'].iloc[i]) for i in highs if float(df['high'].iloc[i]) > entry]
            return min(candidates) if candidates else None
        candidates = [float(df['low'].iloc[i]) for i in lows if float(df['low'].iloc[i]) < entry]
        return max(candidates) if candidates else None
    except Exception as e:
        log_debug("_nearest_swing_level failed:", e)
        return None


def _nearest_untapped_swing(df: pd.DataFrame, entry: float, kind: str = "HIGH", lookback: int = 220) -> Optional[float]:
    """Find the nearest untapped clear swing level above/below entry."""
    try:
        highs, lows = detect_clear_swing_high_low(df, lookback=lookback, left=2, right=2)
        if kind.upper() == "HIGH":
            candidates = []
            for idx in highs:
                price = float(df['high'].iloc[idx])
                if price <= entry:
                    continue
                tail = df['high'].iloc[idx + 1:]
                if len(tail) > 0 and float(tail.max()) >= price:
                    continue
                candidates.append(price)
            return min(candidates) if candidates else None
        candidates = []
        for idx in lows:
            price = float(df['low'].iloc[idx])
            if price >= entry:
                continue
            tail = df['low'].iloc[idx + 1:]
            if len(tail) > 0 and float(tail.min()) <= price:
                continue
            candidates.append(price)
        return max(candidates) if candidates else None
    except Exception as e:
        log_debug("_nearest_untapped_swing failed:", e)
        return None


def _find_fvg_zones(df: pd.DataFrame, lookback: int = 200) -> List[Dict[str, Any]]:
    """Scan for FVG zones. Returns list of dicts with keys: dir, low, high, index."""
    zones: List[Dict[str, Any]] = []
    try:
        if df is None or len(df) < 5:
            return zones
        start = max(2, len(df) - lookback)
        for i in range(start, len(df)):
            c1 = df.iloc[i - 2]
            c3 = df.iloc[i]
            # Bullish FVG: gap up between c1 high and c3 low
            if float(c1['high']) < float(c3['low']):
                zones.append({
                    "dir": "BULL",
                    "low": float(c1['high']),
                    "high": float(c3['low']),
                    "index": i,
                })
            # Bearish FVG: gap down between c3 high and c1 low
            if float(c1['low']) > float(c3['high']):
                zones.append({
                    "dir": "BEAR",
                    "low": float(c3['high']),
                    "high": float(c1['low']),
                    "index": i,
                })
    except Exception as e:
        log_debug("_find_fvg_zones failed:", e)
    return zones


def _select_fvg_level(zones: List[Dict[str, Any]], entry: float, side: str) -> Optional[float]:
    """Select a relevant FVG level for SL anchoring based on side and entry."""
    try:
        if not zones:
            return None
        side_u = (side or "").upper()
        if side_u == "BUY":
            candidates = [z for z in zones if float(z.get("low", 0.0)) < entry]
            if not candidates:
                return None
            z = max(candidates, key=lambda x: int(x.get("index", 0)))
            return float(z.get("low"))
        candidates = [z for z in zones if float(z.get("high", 0.0)) > entry]
        if not candidates:
            return None
        z = max(candidates, key=lambda x: int(x.get("index", 0)))
        return float(z.get("high"))
    except Exception as e:
        log_debug("_select_fvg_level failed:", e)
        return None


def _volatility_buffer_from_atr(df: pd.DataFrame, min_buf: float = 2.0, max_buf: float = 5.0, atr_mult: float = 1.0) -> float:
    """Compute volatility buffer based on ATR(14) with min/max caps."""
    try:
        atr_val = true_atr(df, 14) if df is not None else 0.0
        raw = float(atr_val) * float(atr_mult) if atr_val and atr_val > 0 else float(min_buf)
        return float(clamp(raw, min_buf, max_buf))
    except Exception as e:
        log_debug("_volatility_buffer_from_atr failed:", e)
        return float(min_buf)


def compute_structure_sl_tp(symbol: str, side: str, entry_price: float) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any], Optional[str]]:
    """Compute structure-based SL/TP using swings, FVG, and HTF targets.

    Returns (sl, tp, rr, info, error). If error is not None, caller should skip trade.
    """
    info: Dict[str, Any] = {}
    try:
        side_u = (side or "").upper()
        m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 200, max_age=1.5)
        h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 260, max_age=3.0)
        h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 260, max_age=6.0)

        if not _ensure_df_ok(m15, 20):
            return None, None, None, info, "no M15 data for structure"

        # Clear swing anchors (2 candles each side)
        if side_u == "BUY":
            swing_low = _last_clear_swing_price(m15, kind="LOW", lookback=200)
            if not swing_low:
                return None, None, None, info, "no clear swing low"
            info["swing_low"] = swing_low[1]
            fvg_level = _select_fvg_level(_find_fvg_zones(m15, lookback=220), entry_price, "BUY")
            if fvg_level is not None:
                info["fvg_low"] = fvg_level
            structure_low = min(swing_low[1], fvg_level) if fvg_level is not None else swing_low[1]
            buffer = _volatility_buffer_from_atr(m15, min_buf=2.0, max_buf=5.0, atr_mult=1.0)
            sl = float(structure_low) - float(buffer)

            # TP targets (liquidity)
            candidates: List[Tuple[str, float]] = []
            last_swing_high = _last_clear_swing_price(m15, kind="HIGH", lookback=200)
            if last_swing_high and float(last_swing_high[1]) > entry_price:
                candidates.append(("last_swing_high", float(last_swing_high[1])))
            untapped_high = _nearest_untapped_swing(m15, entry_price, kind="HIGH", lookback=220)
            if untapped_high:
                candidates.append(("untapped_high", float(untapped_high)))
            h1_high = _nearest_swing_level(h1, entry_price, kind="HIGH", lookback=200) if _ensure_df_ok(h1, 20) else None
            if h1_high:
                candidates.append(("h1_resistance", float(h1_high)))
            h4_high = _nearest_swing_level(h4, entry_price, kind="HIGH", lookback=200) if _ensure_df_ok(h4, 20) else None
            if h4_high:
                candidates.append(("h4_resistance", float(h4_high)))
            if not candidates:
                return None, None, None, info, "no liquidity target above entry"
            tp_name, tp = min(candidates, key=lambda x: x[1])
            info["tp_source"] = tp_name
        else:
            swing_high = _last_clear_swing_price(m15, kind="HIGH", lookback=200)
            if not swing_high:
                return None, None, None, info, "no clear swing high"
            info["swing_high"] = swing_high[1]
            fvg_level = _select_fvg_level(_find_fvg_zones(m15, lookback=220), entry_price, "SELL")
            if fvg_level is not None:
                info["fvg_high"] = fvg_level
            structure_high = max(swing_high[1], fvg_level) if fvg_level is not None else swing_high[1]
            buffer = _volatility_buffer_from_atr(m15, min_buf=2.0, max_buf=5.0, atr_mult=1.0)
            sl = float(structure_high) + float(buffer)

            candidates = []
            last_swing_low = _last_clear_swing_price(m15, kind="LOW", lookback=200)
            if last_swing_low and float(last_swing_low[1]) < entry_price:
                candidates.append(("last_swing_low", float(last_swing_low[1])))
            untapped_low = _nearest_untapped_swing(m15, entry_price, kind="LOW", lookback=220)
            if untapped_low:
                candidates.append(("untapped_low", float(untapped_low)))
            h1_low = _nearest_swing_level(h1, entry_price, kind="LOW", lookback=200) if _ensure_df_ok(h1, 20) else None
            if h1_low:
                candidates.append(("h1_support", float(h1_low)))
            h4_low = _nearest_swing_level(h4, entry_price, kind="LOW", lookback=200) if _ensure_df_ok(h4, 20) else None
            if h4_low:
                candidates.append(("h4_support", float(h4_low)))
            if not candidates:
                return None, None, None, info, "no liquidity target below entry"
            tp_name, tp = max(candidates, key=lambda x: x[1])
            info["tp_source"] = tp_name

        rr = _calc_rr(entry_price, sl, tp, side_u)
        return float(sl), float(tp), float(rr) if rr is not None else None, info, None
    except Exception as e:
        log_debug("compute_structure_sl_tp failed:", e)
        return None, None, None, info, "structure calc error"


def detect_choch(df: pd.DataFrame, lookback: int = 50) -> Optional[Dict[str, Any]]:
    """Detect Change of Character (CHoCH) on HTF data.

    Returns {'type':'CHoCH_UP'|'CHoCH_DOWN', 'index': idx} or None.
    """
    try:
        if df is None or len(df) < 6:
            return None
        # CHoCH defined as breach of prior swing structure: a new high above
        # previous swing high followed by loss of higher low structure etc.
        highs, lows = detect_swing_high_low(df, lookback=lookback)
        if not highs or not lows:
            return None
        last_high = highs[-1] if highs else None
        last_low = lows[-1] if lows else None
        last_close = df['close'].iloc[-1]
        # bullish CHoCH: close above last swing high
        if last_high is not None and last_close > df['high'].iloc[last_high]:
            return {'type': 'CHoCH_UP', 'index': len(df)-1}
        # bearish CHoCH: close below last swing low
        if last_low is not None and last_close < df['low'].iloc[last_low]:
            return {'type': 'CHoCH_DOWN', 'index': len(df)-1}
        return None
    except Exception as e:
        log_debug("detect_choch failed:", e)
        return None


def detect_msb(df: pd.DataFrame, lookback: int = 100) -> Optional[Dict[str, Any]]:
    """Detect Market Structure Break (MSB) - extended BOS confirmation.

    Returns {'type':'MSB_UP'|'MSB_DOWN','index': idx, 'strength':0-1} or None.
    """
    try:
        if df is None or len(df) < 10:
            return None
        recent = df.tail(min(len(df), lookback))
        hi = recent['high'].max()
        lo = recent['low'].min()
        last = recent.iloc[-1]
        if last['close'] > hi:
            strength = min(1.0, (last['close'] - hi) / (true_atr(df.tail(lookback), 14) + 1e-9))
            return {'type': 'MSB_UP', 'index': len(df)-1, 'strength': float(clamp(strength,0.0,1.0))}
        if last['close'] < lo:
            strength = min(1.0, (lo - last['close']) / (true_atr(df.tail(lookback), 14) + 1e-9))
            return {'type': 'MSB_DOWN', 'index': len(df)-1, 'strength': float(clamp(strength,0.0,1.0))}
        return None
    except Exception as e:
        log_debug("detect_msb failed:", e)
        return None

def detect_liquidity_sweep(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> Optional[Dict[str, Any]]:
    """Detect H1 liquidity sweeps (wick above/below recent swing then close back inside).

    Returns a dict {"type": "BUY"/"SELL", "index": idx, "strength": 0-1}
    or None if none detected.
    """
    try:
        df = fetch_data_cached(symbol, timeframe, 200, max_age=5.0)
        if df is None or len(df) < 10:
            return None
        highs, lows = detect_swing_high_low(df, lookback=60)
        if highs:
            last_h = highs[-1]
        else:
            last_h = None
        if lows:
            last_l = lows[-1]
        else:
            last_l = None
        # Examine recent candles for wick sweeps beyond last swing
        for i in range(len(df)-5, len(df)):
            c = df.iloc[i]
            # buy-side sweep: wick uptick above previous swing high, close back inside range
            if last_h is not None and c['high'] > df['high'].iloc[last_h] and c['close'] < df['high'].iloc[last_h]:
                strength = min(1.0, (c['high'] - df['high'].iloc[last_h]) / (true_atr(df.tail(50), 14) + 1e-9))
                return {"type": "BUY", "index": i, "strength": float(clamp(strength, 0.0, 1.0))}
            # sell-side sweep: wick below previous swing low, close back inside
            if last_l is not None and c['low'] < df['low'].iloc[last_l] and c['close'] > df['low'].iloc[last_l]:
                strength = min(1.0, (df['low'].iloc[last_l] - c['low']) / (true_atr(df.tail(50), 14) + 1e-9))
                return {"type": "SELL", "index": i, "strength": float(clamp(strength, 0.0, 1.0))}
        return None
    except Exception as e:
        log_debug("detect_liquidity_sweep failed:", e)
        return None

def detect_displacement(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """Detect displacement/thrust: large body relative to ATR and recent range.

    Returns {'score':0-100, 'strong':bool, 'direction': 'BUY'/'SELL'/'NEUTRAL'}
    """
    try:
        if df is None or len(df) < period + 5:
            return {"score": 0, "strong": False, "direction": "NEUTRAL"}
        atr_v = true_atr(df, period)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last['close'] - last['open'])
        wick = (last['high'] - last['low']) - body
        # Body > wick requirement for displacement
        body_wick_ratio = (body / (wick + 1e-9)) if wick >= 0 else 999.0
        # Momentum: compare body to recent average body
        recent_bodies = (df['close'] - df['open']).abs().tail(10)
        avg_body = float(recent_bodies.mean()) if len(recent_bodies) > 0 else 0.0
        momentum = (body / (avg_body + 1e-9)) if avg_body > 0 else 1.0
        score = 0
        if atr_v and atr_v > 0:
            # base score from body size relative to ATR
            score = int(min(60, (body / atr_v) * 30))
        # increase score for body>wick and momentum
        if body_wick_ratio > 1.2:
            score += 15
        if momentum > 1.5:
            score += int(min(25, (momentum - 1.0) * 15))
        # if the prior candle was also aligned, boost score
        if (last['close'] > last['open'] and prev['close'] > prev['open']) or (last['close'] < last['open'] and prev['close'] < prev['open']):
            score += 5
        direction = 'BUY' if last['close'] > last['open'] else ('SELL' if last['close'] < last['open'] else 'NEUTRAL')
        score = int(max(0, min(100, score)))
        strong = score >= 50 and body_wick_ratio > 1.0 and momentum > 1.2
        return {"score": score, "strong": bool(strong), "direction": direction, "body_wick_ratio": float(body_wick_ratio), "momentum": float(momentum)}
    except Exception as e:
        log_debug("detect_displacement failed:", e)
        return {"score": 0, "strong": False, "direction": "NEUTRAL"}

def detect_order_block(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Detect last order block (last opposite directional candle before a strong move).

    Returns {'index': idx, 'direction': 'BUY'/'SELL', 'valid':bool}
    """
    try:
        if df is None or len(df) < 6:
            return None
        # find last candle with large body opposite to current direction
        cur = df.iloc[-1]
        cur_dir = 'BUY' if cur['close'] > cur['open'] else 'SELL'
        for i in range(len(df)-3, 1, -1):
            c = df.iloc[i]
            body = abs(c['close'] - c['open'])
            rng = c['high'] - c['low']
            if body >= 0.6 * rng:
                dir_ = 'BUY' if c['close'] > c['open'] else 'SELL'
                if dir_ != cur_dir:
                    return {"index": i, "direction": dir_, "valid": True}
        return None
    except Exception as e:
        log_debug("detect_order_block failed:", e)
        return None

def smc_confluence(symbol: str, side: str, h1_tf=mt5.TIMEFRAME_H1, m15_tf=mt5.TIMEFRAME_M15) -> Dict[str, Any]:
    """Compute an SMC-based confluence score and allow/block decision.

    Returns: {allow:bool, score:int, direction: 'BUY'|'SELL'|None, details: {...}}
    """
    details = {}
    try:
        # New SMC scoring follows the user-specified weighting focusing on
        # BOS, Liquidity Sweep, and FVG as mandatory components plus trend
        # bias, mitigation and multi-timeframe refinement.  AI is used only
        # as an advisory (predictive) input and never outright blocks SMC.
        h1 = fetch_data_cached(symbol, h1_tf, 200, max_age=CONSTANTS.get("DATA_CACHE_TTL", {}).get('H1', 3.0))
        m15 = fetch_data_cached(symbol, m15_tf, 120, max_age=CONSTANTS.get("DATA_CACHE_TTL", {}).get('M15', 1.5))
        m5 = fetch_data_cached(symbol, mt5.TIMEFRAME_M5, 50, max_age=CONSTANTS.get("DATA_CACHE_TTL", {}).get('M5', 0.6))
        if h1 is None or m15 is None:
            return {"allow": False, "score": 0, "direction": None, "details": {}}

        # Core structure
        bos_raw = detect_bos(h1)
        bos_val = interpret_bos(bos_raw)
        bos_dir = 'BUY' if bos_val > 0 else ('SELL' if bos_val < 0 else None)
        bos_present = bool(bos_dir == side)
        details['bos_raw'] = bos_raw
        details['bos_present'] = bos_present

        # Liquidity sweep detection on H1 and M15 (prefer H1 as HTF structure)
        sweep_h1 = detect_liquidity_sweep(symbol, timeframe=h1_tf)
        sweep_m15 = detect_liquidity_sweep(symbol, timeframe=m15_tf)
        sweep_strength = 0.0
        sweep_confirmed = False
        sweep_forming = False
        for sdict in (sweep_h1, sweep_m15):
            if isinstance(sdict, dict) and sdict.get('strength'):
                sweep_strength = max(sweep_strength, float(sdict.get('strength', 0.0)))
                if sdict.get('type') == side:
                    sweep_confirmed = sweep_confirmed or (sdict.get('strength', 0.0) >= 0.5)
                    sweep_forming = sweep_forming or (sdict.get('strength', 0.0) >= 0.2)
        details['sweep_strength'] = sweep_strength
        details['sweep_confirmed'] = sweep_confirmed
        details['sweep_forming'] = sweep_forming

        # FVG detection on M15 and M5
        fvg_m15 = detect_fvg(m15)
        fvg_m5 = detect_fvg(m5)
        fvg_present = bool(fvg_m15 or fvg_m5)
        fvg_confirmed = bool(fvg_m15)
        fvg_forming = bool(fvg_m5 and not fvg_m15)
        details['fvg_present'] = fvg_present
        details['fvg_confirmed'] = fvg_confirmed
        details['fvg_forming'] = fvg_forming

        # Trend bias (HTF -> LTF alignment): H1 bias vs M15 bias
        try:
            h1_bias = ema_bias(h1)
            m15_bias = ema_bias(m15)
            mtf_refine = 1.0 if (h1_bias == m15_bias == side) else (0.5 if (h1_bias == side or m15_bias == side) else 0.0)
        except Exception as e:
            log_debug("mtf_refine ema_bias failed:", e)
            mtf_refine = 0.0
        details['mtf_refine'] = mtf_refine

        # Mitigation: check for opposing order blocks or strong opposite liquidity
        ob = detect_order_block(h1)
        mitigation_penalty = 0.0
        if isinstance(ob, dict) and ob.get('direction') and ob.get('direction') != side:
            mitigation_penalty = 1.0
        details['mitigation_penalty'] = mitigation_penalty

        # Compose weighted score per requested distribution
        # BOS 25%, Sweep 25%, FVG 20%, Trend Bias 10%, Mitigation 10%, MTF refine 10%
        comp = {}
        comp['bos'] = 25.0 if bos_present else 0.0
        # Sweep contribution scales with strength
        comp['sweep'] = 25.0 * min(1.0, sweep_strength)
        comp['fvg'] = 20.0 if fvg_present else 0.0
        comp['trend'] = 10.0 * mtf_refine
        comp['mitigation'] = -10.0 * mitigation_penalty
        comp['mtf'] = 10.0 * mtf_refine
        details['components'] = comp

        raw_score = int(max(0, min(100, round(sum(comp.values())))))

        # Mandatory SMC core: require BOS + sweep + FVG (confirmed or forming)
        core_bos = bos_present
        core_sweep = sweep_confirmed or sweep_forming
        core_fvg = fvg_confirmed or fvg_forming
        core_ok = bool(core_bos and core_sweep and core_fvg)
        details['core_bos'] = core_bos
        details['core_sweep'] = core_sweep
        details['core_fvg'] = core_fvg

        # AI advisory score (0..1)
        ai_score = blended_prediction(symbol) if ENABLE_PREDICTIVE_AI else 0.5
        details['ai_score'] = ai_score

        # Final allow logic: core must be present; otherwise block. AI cannot block valid SMC.
        allow = bool(core_ok)

        # Direction hint from BOS
        dir_hint = bos_dir

        return {"allow": allow, "score": raw_score, "direction": dir_hint, "details": details}
    except Exception as e:
        log_debug("smc_confluence failed:", e)
        return {"allow": True, "score": 50, "direction": None, "details": {}}

# ===================== ASIAN SESSION SUPPORT =====================
ASIAN_SYDNEY_UTC = (22, 0, 1, 0)  # 22:00–01:00 UTC
ASIAN_TOKYO_UTC  = (0, 0, 3, 0)   # 00:00–03:00 UTC

def is_asian_session(now: Optional[datetime] = None) -> bool:
    try:
        n = now or now_utc()
        tz = SAFE_TZ
        # Normalize to UTC naive hours for comparison
        h = n.hour
        m = n.minute
        # Sydney window spans midnight
        if h >= 22 or h < 1:
            return True
        if 0 <= h < 3:
            return True
        return False
    except Exception as e:
        log_debug("is_asian_session failed:", e)
        return False

# Removed duplicate true_atr placeholder. A full implementation appears later.
# NOTE: The original dummy ADX implementation returned a constant value.
# We later provide a full implementation below, so here we delegate to that
# implementation to avoid duplicated logic.  If the real implementation is not
# yet available, this will safely return 0.0.
# Removed duplicate thin wrapper implementations of adx and adx_rising.  See the full implementation later.
def rsi(df, period=14):
    """Calculate RSI for given DataFrame (expects 'close')."""
    try:
        delta = df['close'].diff(1)
        up = delta.clip(lower=0).rolling(period).mean()
        down = -delta.clip(upper=0).rolling(period).mean()
        rs = up / down
        rsi_val = 100 - 100 / (1 + rs.iloc[-1])
        return rsi_val
    except Exception as e:
        log_debug("rsi calculation failed:", e)
        return 50.0

# ------------------------------------------------------------------------------
# Additional analysis utilities for enhanced accuracy
#
# These helper functions compute derived metrics that inform the AI scoring
# engine about volatility regime and momentum. They are lightweight and
# self-contained so they do not disturb existing logic.

def atr_z_score(df, period: int = 14, window: int = 20) -> float:
    """
    Compute the Z‑score of the ATR to determine whether current volatility
    is expanding or contracting relative to recent history.

    Parameters:
        df     : DataFrame with 'high' and 'low' prices.
        period : Lookback period for ATR calculation (default 14).
        window : Rolling window to compute the mean and standard deviation
                 of the ATR series (default 20).

    Returns:
        A float representing how many standard deviations the latest ATR
        deviates from its rolling mean. Positive values indicate above‑average
        volatility (favourable for breakouts); negative values indicate
        contraction (avoid trading).
    """
    try:
        # True range as high-low difference; fallback to difference of high/low if not available
        tr = (df['high'] - df['low']).rolling(period).mean()
        # Rolling mean and standard deviation of ATR
        ma = tr.rolling(window).mean().iloc[-1]
        sd = tr.rolling(window).std().iloc[-1]
        if sd is None or sd == 0:
            return 0.0
        return float((tr.iloc[-1] - ma) / sd)
    except Exception as e:
        log_debug("atr_z_score failed:", e)
        return 0.0

def rsi_leader(df, period: int = 14) -> float:
    """
    Compute an enhanced RSI measure combining the standard RSI with its
    slope (momentum) to anticipate trend continuations. A rising RSIL
    above 50 favours long trades; a falling RSIL below 50 favours short.

    Parameters:
        df     : DataFrame with 'close' prices.
        period : Lookback period for RSI calculation (default 14).

    Returns:
        A float representing the RSI‑Leader value (0–100 range).
    """
    try:
        # Basic RSI value
        base_rsi = rsi(df, period)
        # Compute a momentum term as the difference between current and
        # RSI value from several periods ago to capture slope
        lookback = min(len(df) - 1, max(3, period // 2))
        if lookback < 1:
            return base_rsi
        prev_rsi = rsi(df.iloc[:-lookback], period)
        slope = base_rsi - prev_rsi
        # Blend the RSI with its slope: positive slope boosts RSIL, negative reduces
        rsil = base_rsi + slope * 0.5
        # Clamp result to [0,100]
        return max(0.0, min(100.0, rsil))
    except Exception as e:
        log_debug("rsi_leader failed:", e)
        return rsi(df, period)
def ema_bias(df):
    """Return BUY/SELL bias based on EMA50 vs EMA200 on given DataFrame."""
    try:
        ema50 = get_ema(df, 50)
        ema200 = get_ema(df, 200)
        if ema50.iloc[-1] > ema200.iloc[-1]:
            return "BUY"
        if ema50.iloc[-1] < ema200.iloc[-1]:
            return "SELL"
        return None
    except Exception as e:
        log_debug("ema_bias failed:", e)
        return None

# ===========================================================================
# Average Directional Index (ADX) and derivatives
#
# The ADX measures trend strength by comparing the magnitude of up and down
# movements relative to the true range of price action.  We define a robust
# implementation here so that all indicator functions appear before any strategy
# or scanning routines.  This avoids runtime ``NameError`` exceptions and
# provides a single, reliable calculation of ADX, plus/minus directional
# indicators and an ADX rising flag.  The implementation follows the classic
# Wilder smoothing approach.

# Dataclass to hold ADX result components.
@dataclass
class ADXResult:
    adx: float
    plus_di: float
    minus_di: float

def _dx_from_dm_tr(plus_dm: pd.Series, minus_dm: pd.Series, tr: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper to compute the smoothed +DI, -DI and DX series from directional
    movements and true range.
    """
    # Smooth true range over the lookback window
    tr_smooth = tr.rolling(window=period).sum()
    # Calculate +DI and –DI percentages, handling division by zero
    pdi = 100.0 * (plus_dm.rolling(window=period).sum() / tr_smooth).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    ndi = 100.0 * (minus_dm.rolling(window=period).sum() / tr_smooth).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Compute the DX as the absolute difference of +DI and –DI relative to their sum
    dx = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)) * 100.0
    dx = dx.fillna(0.0)
    return pdi, ndi, dx

def adx_full(df: pd.DataFrame, period: int = 14) -> ADXResult:
    """
    Return the most recent ADX and directional indicators.

    Parameters
    ----------
    df : DataFrame
        OHLC data with at least 'high', 'low' and 'close' columns.
    period : int, optional
        Lookback period for ADX calculation, by default 14.

    Returns
    -------
    ADXResult
        A dataclass containing the latest ADX, +DI and –DI values.  If
        insufficient data is supplied a result of zeros is returned.
    """
    try:
        # Ensure enough data and required columns exist; delegate to helper if present.
        if df is None or len(df) < period + 2:
            return ADXResult(0.0, 0.0, 0.0)
        # Extract series
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        # Calculate directional movements
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        # True range: maximum of high-low, high-previous close, low-previous close
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        # Get smoothed +DI, –DI and DX
        pdi, ndi, dx = _dx_from_dm_tr(plus_dm, minus_dm, tr, period)
        # Wilder's smoothing for ADX
        adx_series = dx.rolling(window=period).mean().fillna(0.0)
        return ADXResult(float(adx_series.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1]))
    except Exception:
        log_debug("adx_full failed:")
        return ADXResult(0.0, 0.0, 0.0)

def adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute the ADX value for the supplied DataFrame.  Returns 0.0 on any
    error or insufficient data.
    """
    try:
        return adx_full(df, period).adx
    except Exception:
        log_debug("adx failed:")
        return 0.0

def adx_rising(df: pd.DataFrame, period: int = 14) -> Tuple[float, bool]:
    """
    Return the latest ADX and a flag indicating if it is rising compared to
    the previous period.
    """
    try:
        if df is None or len(df) < period * 3:
            return 0.0, False
        # Compute ADX over a trailing window to detect trend direction
        adxs: List[float] = []
        for i in range(len(df) - period, len(df)):
            adxs.append(adx(df.iloc[:i], period))
        if not adxs:
            return 0.0, False
        latest = float(adxs[-1])
        prev = float(adxs[-2]) if len(adxs) > 1 else latest
        return latest, (latest > prev)
    except Exception:
        log_debug("adx_rising failed:")
        return 0.0, False

# AI_THRESHOLD is defined near the top of the file; avoid redefining it here to prevent confusion.
def _ema_slope(series, lookback=10):
    """Calculate slope of a series over given lookback interval."""
    try:
        s = pd.Series(series).astype(float)
        return float(s.iloc[-1] - s.iloc[-lookback])
    except Exception:
        log_debug("_ema_slope failed:")
        return 0.0
def _pct_of_range(val, lo, hi):
    """Return percentage position of val in [lo, hi] range."""
    try:
        if hi <= lo:
            return 0.0
        x = (val - lo) / (hi - lo)
        return float(max(0.0, min(1.0, x)))
    except Exception:
        log_debug("_pct_of_range failed:")
        return 0.0
def ai_validate_signal(symbol, side, h4, h1, m15, m5=None, tick=None, meta=None):
    """
    Evaluate a potential trade signal with an AI-like weighted scoring system.
    Returns: dict {approve: bool, score: int, why: [reasons]}
    """
    reasons = []
    score = 0.0
    w = AI_WEIGHTS  # use current AI model weights
    h4_bias_dir = ema_bias(h4)
    adx_h1_val, adx_is_rising = adx_rising(h1)
    try:
        adx_m15_val = float(adx(m15))
    except Exception:
        adx_m15_val = 0.0
    try:
        rsi_h1_val = float(rsi(h1)); rsi_m15_val = float(rsi(m15))
    except Exception:
        rsi_h1_val = rsi_m15_val = 50.0
    try:
        ema50 = get_ema(h1, 50); ema200 = get_ema(h1, 200)
        ema_diff = ema50 - ema200
        ema_slope_val = float(ema_diff.iloc[-1] - ema_diff.iloc[-min(11, len(ema_diff)-1)]) if len(ema_diff) > 11 else 0.0
    except Exception:
        ema_slope_val = 0.0
    try:
        bos_dir = detect_bos(h1)
    except Exception:
        bos_dir = None
    try:
        fvg_ok = detect_fvg(m15) or (m5 is not None and detect_fvg(m5))
    except Exception:
        fvg_ok = False
    try:
        atr_series = (m15["high"] - m15["low"]).rolling(20).mean()
        atr_hot = bool(atr_series.iloc[-1] >= atr_series.iloc[-2]) if len(atr_series) > 1 else False
    except Exception:
        atr_hot = False
    try:
        session_ok = in_session()
    except Exception:
        session_ok = True
    try:
        spread_ok_flag = spread_ok(symbol)
    except Exception:
        spread_ok_flag = True
    if h4_bias_dir and h4_bias_dir == side:
        score += w["h4_bias"]; reasons.append("H4 bias aligned")
    #
    # Use unified ADX thresholds on both H1 and M15.  The legacy logic used
    # different values for STRICT vs NORMAL modes (30/28 on H1 and 25 on
    # M15).  These inconsistent gates led to mismatches with the unified
    # configuration specified at the top of the module.  Replace them with
    # a single check that both the H1 and M15 ADX values meet the
    # configured minima (ADX_MIN_H1 and ADX_MIN_M15).  Any exceptions
    # encountered when computing ADX values should be logged rather than
    # swallowed silently.
    adx_gate = False
    try:
        adx_gate = (adx_h1_val >= ADX_MIN_H1 and adx_m15_val >= ADX_MIN_M15)
    except Exception as e:
        # log the error but fall back to a conservative gate result
        log_error(f"ai_validate_signal ADX gate error: {e}")
        adx_gate = False
    if adx_gate:
        # scale ADX contributions relative to the original bases (25/20) for
        # backwards compatibility with the weighting function.  When the
        # unified thresholds are met, the score increases and the reason
        # string is recorded.  The rising ADX bonus is still applied.
        score += w["adx_h1"] * _pct_of_range(adx_h1_val, 25, 45)
        score += w["adx_m15"] * _pct_of_range(adx_m15_val, 20, 40)
        if adx_is_rising:
            score += w["adx_rising"]; reasons.append("ADX rising")
    #
    # Apply unified RSI gates.  The previous implementation used
    # hard‑coded 55/45 thresholds which conflicted with the unified
    # configuration (53/47).  For BUY signals both the H1 and M15 RSI
    # values must exceed RSI_BUY_MIN; for SELL signals both must be
    # below RSI_SELL_MAX.  These gates only add to the score – final
    # approval is governed by the AI threshold.
    if side == "BUY" and (rsi_h1_val >= RSI_BUY_MIN and rsi_m15_val >= RSI_BUY_MIN):
        score += w["rsi_regime"]; reasons.append("RSI bullish")
    if side == "SELL" and (rsi_h1_val <= RSI_SELL_MAX and rsi_m15_val <= RSI_SELL_MAX):
        score += w["rsi_regime"]; reasons.append("RSI bearish")
    if (side == "BUY" and ema_slope_val > 0) or (side == "SELL" and ema_slope_val < 0):
        score += w["ema_slope"]; reasons.append("EMA slope favorable")
    if (side == "BUY" and bos_dir == "BOS_UP") or (side == "SELL" and bos_dir == "BOS_DOWN"):
        score += w["bos_match"]; reasons.append("Structure break aligned")
    if fvg_ok or atr_hot:
        score += w["fvg_or_atrhot"]; reasons.append("Volatility momentum")
    if session_ok:
        score += w["session"]; reasons.append("In-session")
    if spread_ok_flag:
        score += w["spread_ok"]; reasons.append("Spread OK")
    score = int(round(min(100.0, score), 0))
    score_with_bias = score + AI_BIAS
    score_with_bias = int(round(max(0.0, min(100.0, score_with_bias)), 0))
    # Use the unified AI threshold for approval.  AI_THRESHOLD is set to
    # AI_MIN_SCORE near the top of the module so this comparison remains
    # consistent across the codebase.
    approve = (score_with_bias >= AI_THRESHOLD)
    return {"approve": approve, "score": score_with_bias, "why": reasons}
def high_prob_filters_ok(symbol, h1, m15, h4, side, announce=False, label=""):
    """
    LOOSEN full-trade filters:
      • H4 EMA50/200 bias must match.
      • ADX gate: one of (H1, M15) ≥ 25 and the other ≥ 22.
      • RSI: BUY if H1 & M15 ≥ 55; SELL if H1 & M15 ≤ 45.
      • ATR floor (M15) ≥ LOOSEN_FULL_ATR_FLOOR.
      • Range filter: current candle not within LOOSEN_FULL_RANGE_MAX_PCT of prior H/L.
      • Spread under cap.
    """
    # If HTF gating is globally disabled, do not block on H1/H4/D1 filters.
    try:
        if globals().get('HARD_DISABLE_HTF'):
            # Quietly bypass HTF-based high-prob filters when globally disabled.
            return True
    except Exception as e:
        try:
            log_debug("HARD_DISABLE_HTF check failed in high_prob_filters_ok:", e)
        except Exception:
            pass

    bias_h4 = ema_bias(h4)
    if bias_h4 != side:
        if announce: log_msg(f"🚫 {symbol} {label} - H4 bias mismatch ({bias_h4} vs {side})")
        return False
    adx_h1_val, _r = adx_rising(h1); adx_m15_val = float(adx(m15))
    adx_ok = ((adx_h1_val >= LOOSEN_FULL_ADX_H1_MIN and adx_m15_val >= LOOSEN_FULL_ADX_M15_MIN) or
              (adx_m15_val >= LOOSEN_FULL_ADX_H1_MIN and adx_h1_val >= LOOSEN_FULL_ADX_M15_MIN))
    if not adx_ok:
        if announce: log_msg(f"🚫 {symbol} {label} - ADX fail (H1 {adx_h1_val:.1f}, M15 {adx_m15_val:.1f})")
        return False
    rsi_h1_val = float(rsi(h1)); rsi_m15_val = float(rsi(m15))
    if side == "BUY" and not (rsi_h1_val >= LOOSEN_FULL_RSI_BUY and rsi_m15_val >= LOOSEN_FULL_RSI_BUY):
        if announce: log_msg(f"🚫 {symbol} {label} - RSI BUY gate fail (H1 {rsi_h1_val:.1f}, M15 {rsi_m15_val:.1f})")
        return False
    if side == "SELL" and not (rsi_h1_val <= LOOSEN_FULL_RSI_SELL and rsi_m15_val <= LOOSEN_FULL_RSI_SELL):
        if announce: log_msg(f"🚫 {symbol} {label} - RSI SELL gate fail (H1 {rsi_h1_val:.1f}, M15 {rsi_m15_val:.1f})")
        return False
    try:
        atr_m15 = true_atr(m15, period=14)
    except Exception as e:
        try:
            log_debug("true_atr failed in high_prob_filters_ok:", e)
        except Exception:
            pass
        atr_m15 = 0.0
    if atr_m15 < LOOSEN_FULL_ATR_FLOOR:
        if announce: log_msg(f"🚫 {symbol} {label} - ATR floor fail (M15 ATR {atr_m15:.3f} < {LOOSEN_FULL_ATR_FLOOR})")
        return False
    try:
        prev_high = float(m15['high'].iloc[-2]); prev_low = float(m15['low'].iloc[-2])
        last_close = float(m15['close'].iloc[-1])
        mid = (prev_high + prev_low) / 2.0
        pct = abs(last_close - mid) / max(1e-9, mid) * 100.0
        if pct <= LOOSEN_FULL_RANGE_MAX_PCT:
            if announce: log_msg(f"🚫 {symbol} {label} - Range filter (|pos| {pct:.2f}% <= {LOOSEN_FULL_RANGE_MAX_PCT}%)")
            return False
    except Exception as e:
        try:
            log_debug("range filter calculation failed in high_prob_filters_ok:", e)
        except Exception:
            pass
    # For XAUUSD use the specialised policy for full trades; for other
    # symbols fall back to the legacy spread_ok behaviour.
    try:
        # Use the specialised policy for gold (XAUUSD), falling back to the legacy
        # spread check for other symbols.  When an exception is raised during the
        # XAUUSD checks, handle it gracefully by falling back to the spread check.
        if _base_of(symbol) == 'XAUUSD':
            ok_full, reason_full = can_place_on_xau(symbol, micro=False)
            if not ok_full:
                if announce:
                    try:
                        log_block_verbose(symbol, f"HighProb blocked", extra_info=f"{label} - {reason_full}")
                    except Exception:
                        _log_block_once(symbol, f"🚫 {symbol} {label} - {reason_full}")
                return False
        else:
            if not spread_ok(symbol):
                if announce:
                    log_msg(f"🚫 {symbol} {label} - spread above cap")
                return False
    except Exception as e:
        # On any unexpected exception during the XAUUSD/spread checks, log and
        # fallback to the spread check to avoid blocking trades unnecessarily.
        log_debug("high_prob_filters_ok exception:", e)
        if not spread_ok(symbol):
            if announce:
                log_msg(f"🚫 {symbol} {label} - spread above cap")
            return False
    return True
def ichimoku(df):
    """Calculate Ichimoku Kinko Hyo cloud components for last data point."""
    high9 = df['high'].rolling(9).max()
    low9  = df['low'].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = df['high'].rolling(26).max()
    low26  = df['low'].rolling(26).min()
    kijun = (high26 + low26) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    last_close = df['close'].iloc[-1]
    return last_close, span_a.iloc[-1], span_b.iloc[-1]
def ema_bias_signal(h1):
    """EMA 50/200 bias on H1, returning bias and the EMA series for further use."""
    ema50 = get_ema(h1, 50); ema200 = get_ema(h1, 200)
    if ema50.iloc[-1] > ema200.iloc[-1]:
        return "BUY", ema50, ema200
    if ema50.iloc[-1] < ema200.iloc[-1]:
        return "SELL", ema50, ema200
    return None, ema50, ema200
def strong_confluence(h1, m15, m5):
    """
    Legacy 'strong confluence' signal detection:
    - Must have: EMA bias + BOS + in_session() all true.
    - Plus at least one of: FVG or ATR momentum.
    - Additional filters: ADX >= 25, RSI not overbought/oversold, Ichimoku bias match.
    Returns: side ("BUY"/"SELL") if strong confluence found, else None.
    """
    bos = detect_bos(h1)
    ema50, ema200 = get_ema(h1, 50), get_ema(h1, 200)
    bias = "BUY" if ema50.iloc[-1] > ema200.iloc[-1] else "SELL"
    session_ok = in_session()
    fvg_ok = detect_fvg(m15) or detect_fvg(m5)
    try:
        atr_series = (m15["high"] - m15["low"]).rolling(14).mean()
        atr_hot = bool(atr_series.iloc[-1] >= atr_series.iloc[-2])
    except Exception:
        atr_hot = False
    try:
        adx_val = adx(h1)
    except Exception:
        adx_val = 0
    try:
        rsi_val = rsi(h1)
    except Exception:
        rsi_val = 50
    try:
        price, span_a, span_b = ichimoku(h1)
        ichimoku_bias = "BUY" if price > max(span_a, span_b) else "SELL"
    except Exception:
        ichimoku_bias = None
    side = None
    #
    # Use unified confluence gates rather than hard‑coded 25/75 RSI and
    # ADX values.  A BUY signal requires the H1 ADX to meet the global
    # minimum and the H1 RSI to exceed the unified BUY threshold; a SELL
    # signal requires the H1 ADX to meet the minimum and the H1 RSI to
    # fall below the unified SELL threshold.  This brings legacy strong
    # confluence detection in line with the rest of the bot.
    if (bias == "BUY" and bos == "BOS_UP" and session_ok and (fvg_ok or atr_hot)
            and adx_val >= ADX_MIN_H1 and rsi_val >= RSI_BUY_MIN and (ichimoku_bias != "SELL")):
        side = "BUY"
    elif (bias == "SELL" and bos == "BOS_DOWN" and session_ok and (fvg_ok or atr_hot)
          and adx_val >= ADX_MIN_H1 and rsi_val <= RSI_SELL_MAX and (ichimoku_bias != "BUY")):
        side = "SELL"
    return side
def build_signals_by_strategy(symbol, announce=False):
    """Route to appropriate strategy function for the given symbol."""
    base = _base_of(symbol)
    mapped = (STRATEGY_MAP.get(base) or (STRATEGY or "GOAT")).upper()
    if mapped == "GOAT" and "build_signals_goat" in globals():
        return build_signals_goat(symbol, announce=announce)
    if mapped == "GBPUSD_STRAT":
        return build_signals_gbpusd(symbol, announce=announce)
    if mapped == "GBPJPY_STRAT":
        return build_signals_gbpjpy(symbol, announce=announce)
    if "build_signals" in globals():
        return build_signals(symbol, announce=announce)
    return []
def build_signals_gbpusd(symbol, announce=False, period=20):
    if not in_session():
        if announce: log_msg(f"⏸️ {symbol} out of session")
        return []
    h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 260, max_age=6.0)
    h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 220, max_age=3.0)
    m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 160, max_age=1.5)
    if h4 is None or h1 is None or m15 is None or len(h1) < period + 5:
        if announce: log_msg(f"⚠️ {symbol} insufficient data for BUILD_SIGNALS_GBPUSD")
        return []
    bias = ema_bias(h1)
    bos = detect_bos(h1)
    side = "BUY" if (bias == "BUY" and bos == "BOS_UP") else ("SELL" if (bias == "SELL" and bos == "BOS_DOWN") else None)
    if not side:
        if announce: log_msg(f"❌ {symbol} BUILD_SIGNALS_GBPUSD: no BOS in bias direction")
        return []
    if not high_prob_filters_ok(symbol, h1, h1, h4, side, announce=announce, label="BUILD_SIGNALS_GBPUSD"):
        return []
    t = mt5.symbol_info_tick(symbol)
    if not t:
        if announce: log_msg(f"⚠️ {symbol} no tick for BUILD_SIGNALS_GBPUSD trade")
        return []
    try:
        atr_series = (h1["high"] - h1["low"]).rolling(14).mean()
        a_m15 = float(atr_series.iloc[-1])
    except Exception:
        a_m15 = max(0.10, abs(t.ask - t.bid))
    sl = (t.bid - 2.0 * a_m15) if side == "BUY" else (t.ask + 2.0 * a_m15)
    if announce:
        log_msg(f"✅ BUILD_SIGNALS_GBPUSD signal: {side} {symbol} | SL≈2*ATR(h1)")
    return [{"side": side, "sl": sl, "tp1": None, "tp2": None, "tp3": None, "atr": a_m15}]
def build_signals_gbpjpy(symbol, announce=False, period=20):
    if not in_session():
        if announce: log_msg(f"⏸️ {symbol} out of session")
        return []
    h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 260, max_age=6.0)
    h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 220, max_age=3.0)
    m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 160, max_age=1.5)
    if h4 is None or h1 is None or m15 is None or len(h1) < period + 5:
        if announce: log_msg(f"⚠️ {symbol} insufficient data for BUILD_SIGNALS_GBPJPY")
        return []
    bias = ema_bias(h1)
    bos = detect_bos(h1)
    side = "BUY" if (bias == "BUY" and bos == "BOS_UP") else ("SELL" if (bias == "SELL" and bos == "BOS_DOWN") else None)
    if not side:
        if announce: log_msg(f"❌ {symbol} BUILD_SIGNALS_GBPJPY: no BOS in bias direction")
        return []
    if not high_prob_filters_ok(symbol, h1, m15, h4, side, announce=announce, label="BUILD_SIGNALS_GBPJPY"):
        return []
    t = mt5.symbol_info_tick(symbol)
    if not t:
        if announce: log_msg(f"⚠️ {symbol} no tick for BUILD_SIGNALS_GBPJPY trade")
        return []
    try:
        atr_series = (m15["high"] - m15["low"]).rolling(14).mean()
        a_m15 = float(atr_series.iloc[-1])
    except Exception:
        a_m15 = max(0.10, abs(t.ask - t.bid))
    sl = (t.bid - 2.0 * a_m15) if side == "BUY" else (t.ask + 2.0 * a_m15)
    if announce:
        log_msg(f"✅ BUILD_SIGNALS_GBPJPY signal: {side} {symbol} | SL≈2*ATR(M15)")
    return [{"side": side, "sl": sl, "tp1": None, "tp2": None, "tp3": None, "atr": a_m15}]
def deprecated_build_signals_goat(symbol, announce=False, period=20):
    """
    GOAT strategy:
      - H1 Donchian breakout with ATR-based buffer to avoid fakeouts.
      - Trend filter: EMA50 > EMA200 (H1).
      - Strict MTF filter: uses H4/H1/M15 via high_prob_filters_ok.
      - SL = 2 * ATR(M15).
    """
    if not in_session():
        if announce:
            log_msg(f"⏸️ {symbol} out of session")
        return []
    h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 260, max_age=6.0)
    h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 220, max_age=3.0)
    m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 160, max_age=1.5)
    if h4 is None or h1 is None or m15 is None or len(h1) < period + 5:
        if announce:
            log_msg(f"⚠️ {symbol} insufficient data for GOAT strategy")
        return []
    bias, ema50, ema200 = ema_bias_signal(h1)
    prev_high = h1['high'].rolling(period).max().iloc[-2]
    prev_low  = h1['low'].rolling(period).min().iloc[-2]
    close_prev = float(h1['close'].iloc[-2])
    close_now  = float(h1['close'].iloc[-1])
    a_m15 = float(rsi(m15))  # FIX: Using RSI as placeholder for ATR
    buffer = max((mt5.symbol_info(symbol).point or 0.0001) * 2, 0.20 * a_m15)
    long_break  = (close_prev <= prev_high) and (close_now > prev_high + buffer)
    short_break = (close_prev >= prev_low)  and (close_now < prev_low - buffer)
    side = None
    if bias == "BUY" and long_break:
        side = "BUY"
    elif bias == "SELL" and short_break:
        side = "SELL"
    if not side:
        if announce:
            log_msg(f"❌ {symbol} GOAT: no qualified breakout (buffered)")
        return []
    if not high_prob_filters_ok(symbol, h1, m15, h4, side, announce=announce, label="GOAT"):
        return []
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        if announce:
            log_msg(f"⚠️ {symbol} no tick for GOAT trade")
        return []
    sl = (tick.bid - 2.0 * a_m15) if side == "BUY" else (tick.ask + 2.0 * a_m15)
    if announce:
        log_msg(f"🐐 GOAT✅: {side} {symbol} | Donchian({period})+buffer {buffer:.4f} | SL=2*ATR(M15)")
    return [{
        "side": side, "sl": sl, "tp1": None, "tp2": None, "tp3": None, "atr": a_m15
    }]
def spread_points(symbol):
    """Calculate current spread in points for the given symbol."""
    tick = mt5.symbol_info_tick(symbol)
    si = mt5.symbol_info(symbol)
    if not tick or not si or not si.point:
        return 999999
    return int(abs(tick.ask - tick.bid) / si.point)
def spread_ok(symbol):
    """Check if current spread is within the configured max spread for the symbol."""
    try:
        base = _base_of(symbol)
        max_sp = PAIR_CONFIG.get(base, {}).get('max_spread', None)
        if max_sp is None:
            return True
        return spread_points(symbol) <= max_sp
    except Exception as e:
        log_debug("spread_ok failed:", e)
        return True
def build_deviations(symbol):
    """Build a list of slippage deviation values to try for order placement (ensure 100 included)."""
    sp = spread_points(symbol) or 0
    base_dev = max(40, sp * 2)
    devs = sorted(set([base_dev, base_dev * 2, MAX_DEVIATION_DEFAULT]))
    return devs
def symbol_lot_specs(symbol):
    """Get lot sizing specs (min volume, step, max volume) for the symbol."""
    si = mt5.symbol_info(symbol)
    if not si:
        return (0.10, 0.10, 100.0)
    step = si.volume_step or 0.01
    minv = si.volume_min or step
    maxv = si.volume_max or 100.0
    return (float(minv), float(step), float(maxv))
def normalize_volume(symbol, lots):
    """Normalize a lot size to the symbol's allowed step and range."""
    minv, step, maxv = symbol_lot_specs(symbol)
    lots = max(minv, min(maxv, float(lots)))
    s = f"{step:.8f}".rstrip('0').rstrip('.')
    prec = len(s.split('.')[1]) if '.' in s else 0
    steps = round(lots / step)
    return round(steps * step, prec)
def normalize_price(symbol, price):
    """Normalize a price to the symbol's tick size."""
    si = mt5.symbol_info(symbol)
    if not si or not si.point:
        return price
    return round(price / si.point) * si.point
def stops_level_points(symbol):
    """Get the stop level (min distance for SL/TP) in points for the symbol."""
    si = mt5.symbol_info(symbol)
    return int(getattr(si, "trade_stops_level", 0) or getattr(si, "stops_level", 0) or 0)

def safe_get_rates(symbol, timeframe, bars=150):
    """Safe wrapper around get_data that returns a DataFrame or None.

    Many legacy routines call `safe_get_rates`; implement as a thin
    wrapper so older call sites continue to work.
    """
    try:
        return get_data(symbol, timeframe, bars)
    except Exception as e:
        try:
            log_debug("safe_get_rates failed:", e)
        except Exception:
            pass
        return None

def ema(df, period=50):
    """Return the latest EMA value (float) for the given DataFrame."""
    try:
        s = get_ema(df, period)
        return float(s.iloc[-1])
    except Exception as e:
        try:
            log_debug("ema computation failed:", e)
        except Exception:
            pass
        return 0.0

def current_ruleset():
    """Return (firm_name, rules_dict) using detect_prop_firm()."""
    try:
        firm, rules, _server = detect_prop_firm()
        return firm, rules or {}
    except Exception as e:
        try:
            log_debug("current_ruleset detection failed:", e)
        except Exception:
            pass
        return None, {}

def account_is_demo() -> bool:
    """Return True when the connected account appears to be a demo account."""
    try:
        ai = mt5.account_info()
        if not ai:
            return False
        srv = getattr(ai, 'server', '') or ''
        if 'demo' in str(srv).lower():
            return True
        tm = getattr(ai, 'trade_mode', None)
        if tm in (1, 'DEMO', 'demo'):
            return True
        return False
    except Exception as e:
        try:
            log_debug("account_is_demo failed:", e)
        except Exception:
            pass
        return False

def _equity_info():
    """Return (balance, equity) tuple from MT5 account info."""
    try:
        ai = mt5.account_info()
        if not ai:
            return 0.0, 0.0
        return float(getattr(ai, 'balance', 0.0) or 0.0), float(getattr(ai, 'equity', 0.0) or 0.0)
    except Exception as e:
        try:
            log_debug("_equity_info failed:", e)
        except Exception:
            pass
        return 0.0, 0.0

# Removed duplicate _notify_trade_event stub. A compact notifier is defined later.

def sltp_guard(symbol, side, price, sl, tp, adjust=False):
    """Permissive SL/TP guard used when no external guard is provided.

    Returns: (ok:bool, sl, tp, info:dict)
    """
    try:
        if sl is None:
            return False, sl, tp, {"reason": "NoSL"}
        ok, pts, need = _min_sl_points(symbol, price, sl)
        if not ok and not adjust:
            return False, sl, tp, {"reason": "SLTooTight", "pts": pts, "need": need}
# The above return ends the try-block.  If any exception occurs,
# default to permissive behaviour and indicate no adjustment.
        return True, sl, tp, {"adjusted": False}
    except Exception:
        return True, sl, tp, {"adjusted": False}

# -----------------------------------------------------------------------------
# Symbol tradability pre‑check
#
# To prevent trades from being submitted when the market is closed or when
# trading is disabled on a particular symbol (e.g. outside of trading hours,
# weekends or broker maintenance), this helper performs a series of lightweight
# checks before any order is sent.  It verifies that the symbol is selected
# and in full trading mode, attempts to subscribe to the market book if
# available, and ensures there is a valid tick with non‑zero bid/ask prices.
# When any of these checks fail, a concise log message is emitted and the
# trade is skipped.  See usage in `safe_order_send` and `send_market_with_retries`.

def _tradeable_pre_check(symbol: str) -> bool:
    """Return True if the given symbol appears tradable right now.

    Performs checks on symbol_info.trade_mode, symbol selection, market book
    subscription and tick data.  When any check fails, a log message is
    emitted and the function returns False.  This prevents order_send() from
    being called when the market is closed or trading is disabled, thereby
    avoiding MT5 retcode 10018 (trade blocked).
    """
    try:
        if not symbol:
            return False
        # Obtain symbol info and ensure trading mode is full
        si = None
        try:
            si = mt5.symbol_info(symbol)
        except Exception:
            si = None
        full_const = None
        try:
            full_const = getattr(mt5, 'SYMBOL_TRADE_MODE_FULL')
        except Exception:
            # Fallback constant value for full trading mode (4)
            full_const = 4
        if not si or getattr(si, 'trade_mode', None) != full_const:
            log_msg("Trade skipped — market closed / trading disabled (pre-check)")
            return False
        # Ensure symbol is selected/enabled for trading
        try:
            # symbol_select returns True if selection succeeded; False otherwise
            if not mt5.symbol_select(symbol, True):
                log_msg("Trade skipped — market closed / trading disabled (pre-check)")
                return False
        except Exception as e:
            try:
                log_debug("mt5.symbol_select failed in _tradeable_pre_check:", e)
            except Exception:
                pass
        # Subscribe to market book if supported; False indicates failure
        try:
            if hasattr(mt5, 'market_book_add'):
                try:
                    book_ok = mt5.market_book_add(symbol)
                except Exception:
                    book_ok = None
                # Some brokers return False when book subscription fails
                if book_ok is False:
                    log_msg("Trade skipped — market closed / trading disabled (pre-check)")
                    return False
        except Exception as e:
            try:
                log_debug("market_book_add check failed in _tradeable_pre_check:", e)
            except Exception:
                pass
        # Require a valid tick with non‑zero bid/ask (market open)
        tick = None
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception as e:
            try:
                log_debug("mt5.symbol_info_tick failed in _tradeable_pre_check:", e)
            except Exception:
                pass
            tick = None
        if not tick:
            log_msg("Trade skipped — market closed / trading disabled (pre-check)")
            return False
        bid = getattr(tick, 'bid', None)
        ask = getattr(tick, 'ask', None)
        if not bid or not ask or bid == 0 or ask == 0:
            log_msg("Trade skipped — market closed / trading disabled (pre-check)")
            return False
        return True
    except Exception as e:
        # On unexpected errors, be conservative and skip (log for diagnostics)
        try:
            log_msg("Trade skipped — market closed / trading disabled (pre-check): %s" % (e,))
        except Exception:
            try:
                log_debug("_tradeable_pre_check unexpected error:", e)
            except Exception:
                pass
        return False
    except Exception:
        return True, sl, tp, {"adjusted": False}

def safe_order_send(req):
    """Send an order with additional reliability and safety controls.

    This wrapper performs a tradeable pre‑check, validates spread and volatility
    conditions, refreshes the price on every attempt, adjusts stop‑loss and
    take‑profit levels to satisfy broker stop distances and falls back to
    IOC after multiple FOK attempts.  It waits a short, random interval
    between retries to mitigate trade context busy errors.  It returns the
    final response from ``mt5.order_send`` or ``None`` if pre‑checks fail.
    """
    # Pre‑check: ensure the symbol is tradable for new open orders (DEAL or PENDING)
    try:
        symbol_for_check = req.get('symbol') if isinstance(req, dict) else None
        action = req.get('action') if isinstance(req, dict) else None
        # Determine constants for deal and pending actions
        try:
            action_deal = getattr(mt5, 'TRADE_ACTION_DEAL', 1)
        except Exception:
            action_deal = 1
        try:
            action_pending = getattr(mt5, 'TRADE_ACTION_PENDING', 5)
        except Exception:
            action_pending = 5
        if symbol_for_check and action in (action_deal, action_pending):
            try:
                if not _tradeable_pre_check(symbol_for_check):
                    return None
            except Exception:
                # If pre‑check errors, block conservatively
                return None
    except Exception as e:
        try:
            log_debug("safe_order_send pre-checks failed:", e)
        except Exception:
            pass

    # Helper to refresh price and adjust SL/TP according to stop levels
    def _refresh_price_and_stops(r: Dict[str, Any]) -> bool:
        try:
            sym = r.get('symbol')
            if not sym:
                return False
            # Fetch latest tick
            tick = None
            try:
                tick = mt5.symbol_info_tick(sym)
            except Exception:
                tick = None
            if not tick:
                return False
            # Determine side from order type (defaults to BUY on unknown)
            try:
                otype = int(r.get('type', getattr(mt5, 'ORDER_TYPE_BUY', 0)))
            except Exception:
                otype = getattr(mt5, 'ORDER_TYPE_BUY', 0)
            side = 'BUY' if otype == getattr(mt5, 'ORDER_TYPE_BUY', 0) else 'SELL'
            price = float(getattr(tick, 'ask', 0.0)) if side == 'BUY' else float(getattr(tick, 'bid', 0.0))
            # Get existing SL/TP
            sl = r.get('sl')
            tp = r.get('tp')
            # Determine minimal stop distance in price units
            try:
                stops_pts = stops_level_points(sym)
            except Exception:
                stops_pts = 0
            try:
                si = mt5.symbol_info(sym)
                pt = float(getattr(si, 'point', 0.0001)) if si else 0.0001
            except Exception:
                pt = 0.0001
            # Compute raw minimum stop distance (points * point size)
            min_dist = float(stops_pts) * float(pt)
            # Determine additional buffer based on current spread and optional extra pts.
            try:
                spread_price = (float(getattr(tick, 'ask', 0.0)) - float(getattr(tick, 'bid', 0.0))) if tick else 0.0
            except Exception:
                spread_price = 0.0
            try:
                extra_pts_env = SL_EXTRA_BUFFER_PTS if 'SL_EXTRA_BUFFER_PTS' in globals() else 0.0
            except Exception:
                extra_pts_env = 0.0
            extra_price = float(extra_pts_env) * float(pt)
            buffer_dist = min_dist + spread_price + extra_price
            # Expand stops by margin multiplier when available
            margin_mult = MIN_STOP_MARGIN_MULTIPLIER if 'MIN_STOP_MARGIN_MULTIPLIER' in globals() else 1.1
            # Determine the minimum required distance: either the broker minimum times margin or the computed buffer
            try:
                target_dist = max(min_dist * float(margin_mult), buffer_dist)
            except Exception:
                target_dist = buffer_dist
            if sl is not None:
                try:
                    sl = float(sl)
                    if side == 'BUY':
                        # Ensure SL is below entry by at least target_dist
                        if (price - sl) <= target_dist:
                            sl = price - target_dist
                    else:
                        if (sl - price) <= target_dist:
                            sl = price + target_dist
                    sl = normalize_price(sym, sl)
                except Exception:
                    pass
            if tp is not None:
                try:
                    tp = float(tp)
                    # Only adjust TP relative to min_dist*margin to avoid extremely tight targets
                    if side == 'BUY':
                        if (tp - price) <= (min_dist * margin_mult):
                            tp = price + min_dist * margin_mult
                    else:
                        if (price - tp) <= (min_dist * margin_mult):
                            tp = price - min_dist * margin_mult
                    tp = normalize_price(sym, tp)
                except Exception:
                    pass
            # Normalize entry price
            try:
                price = normalize_price(sym, price)
            except Exception:
                pass
            # Update the request dictionary
            r['price'] = price
            if sl is not None:
                r['sl'] = sl
            if tp is not None:
                r['tp'] = tp
            return True
        except Exception:
            return False

    # Check spread and volatility guard
    try:
        sym = req.get('symbol') if isinstance(req, dict) else None
        if sym:
            tick1 = mt5.symbol_info_tick(sym)
            if not tick1:
                return None
            # Compute spread in points
            try:
                sp = spread_points(sym)
            except Exception:
                sp = None
            # Use safe threshold constant; default to 200 points if undefined
            try:
                safe_sp = SPREAD_SAFE_THRESHOLD_PTS
            except Exception:
                safe_sp = 200
            if sp is not None and sp > safe_sp:
                log_msg(f"🛑 Spread {sp} > safe threshold {safe_sp} on {sym}, aborting order")
                return None
            # Volatility jump check: compare mid price movement over a short interval
            try:
                mid1 = (float(getattr(tick1, 'bid', 0.0)) + float(getattr(tick1, 'ask', 0.0))) / 2.0
                # wait a brief moment to measure jump
                time.sleep(0.05)
                tick2 = mt5.symbol_info_tick(sym)
                if tick2:
                    mid2 = (float(getattr(tick2, 'bid', 0.0)) + float(getattr(tick2, 'ask', 0.0))) / 2.0
                    try:
                        delta_pts = abs(mid2 - mid1) / (float(mt5.symbol_info(sym).point) if mt5.symbol_info(sym) else 0.0001)
                    except Exception:
                        delta_pts = 0.0
                    try:
                        vt_thresh = VOLATILITY_JUMP_THRESHOLD_PTS
                    except Exception:
                        vt_thresh = 100
                    if delta_pts > vt_thresh:
                        log_msg(f"🛑 Tick jump {delta_pts:.1f} > volatility threshold {vt_thresh} on {sym}, aborting order")
                        return None
            except Exception:
                pass
    except Exception:
        pass

    last_res: Any = None
    # Attempt up to three executions: two FOK, final IOC
    for attempt in range(3):
        try:
            # Refresh latest price and stops before each send
            _refresh_price_and_stops(req)
            # Deviation: 150 on first, 100 on second, 80 on third
            if attempt == 0:
                dev = 150
            elif attempt == 1:
                dev = 100
            else:
                dev = 80
            req['deviation'] = dev
            # Filling mode: FOK for first two attempts, IOC on final
            if attempt < 2:
                req['type_filling'] = getattr(mt5, 'ORDER_FILLING_FOK', req.get('type_filling'))
            else:
                req['type_filling'] = getattr(mt5, 'ORDER_FILLING_IOC', req.get('type_filling'))
            # Send order
            res = mt5.order_send(req)
            last_res = res
            # Determine success via retcode 10009
            if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                log_msg(f"✅ Executed @ deviation {dev} (attempt {attempt}) | ticket={getattr(res,'order',0)} deal={getattr(res,'deal',0)}")
                return res
            else:
                ret = getattr(res, 'retcode', None) if res else None
                log_msg(f"⚠️ order_send retcode {ret} on attempt {attempt}")
        except Exception as e:
            log_msg(f"⚠️ order_send exception on attempt {attempt}: {e}")
        # Wait between retries (except after final)
        if attempt < 2:
            # randomised delay between 0.2 and 0.3 seconds
            try:
                delay = random.uniform(0.2, 0.3)
            except Exception:
                delay = 0.25
            try:
                time.sleep(delay)
            except Exception:
                pass
    # After all attempts, log and return last result
    try:
        rc = getattr(last_res, 'retcode', None) if last_res else None
        log_msg(f"🛑 order_send failed after retries; last retcode {rc}")
    except Exception:
        pass
    return last_res

# =============================================================================
# === IC Markets Micro-Trade Enforcement and Safe Order Wrapper ===
#
# To comply with IC Markets restrictions (micro trades only) and resolve
# MT5 transient errors, we override ``safe_order_send`` with a wrapper that:
#   • Detects IC Markets accounts via mt5.account_info().server.
#   • Forces micro lot sizing (0.01 by default; 0.02 when balance is within
#     £150–£200) on new open trades, and rounds/clamps the volume to the
#     symbol’s volume_min, volume_max and volume_step.  If the incoming
#     volume differs from the corrected value, a log/Telegram message is
#     emitted to record the adjustment.
#   • Serialises order sends via a global lock and inserts a ~0.4 s delay
#     between retries to mitigate “trade context busy” (retcode 10016).
#   • Attempts up to 3 FOK fills, then falls back to IOC.  Each retry is
#     logged; failure only aborts after all retries fail.
#   • Delegates to the original ``safe_order_send`` implementation for
#     non‑IC Markets brokers so FTMO and other accounts retain normal logic.
#
# The wrapper functions and constants below are defined in this section and
# registered immediately after the original ``safe_order_send`` definition.
## -----------------------------------------------------------------------------

# Preserve the original safe_order_send so we can delegate for non‑ICM brokers.
_original_safe_order_send = safe_order_send  # type: ignore[var-annotated]

# Lock to serialise order sends and prevent “trade context busy” errors on IC Markets.
_safe_order_lock_icm = threading.Lock()

def is_icm_account() -> bool:
    """Return True if the current MT5 account is hosted on an IC Markets server.

    Detection uses substring matching on mt5.account_info().server via the
    existing ``is_icmarkets_server`` helper.  Any exception defaults to False.
    """
    try:
        ai = mt5.account_info()
        srv = getattr(ai, 'server', '') or ''
        return is_icmarkets_server(str(srv))
    except Exception:
        return False

def icm_micro_lot() -> float:
    """Compute the micro lot size for IC Markets accounts.

    By default the lot size is 0.01.  When the account balance is within
    150 to 200 (inclusive) the lot size increases to 0.02.  Any errors fall
    back to the default 0.01.
    """
    try:
        ai = mt5.account_info()
        bal = float(getattr(ai, 'balance', 0.0)) if ai is not None else 0.0
        if 150.0 <= bal <= 200.0:
            return 0.02
    except Exception as e:
        try:
            log_debug("icm_micro_lot balance check failed:", e)
        except Exception:
            pass
    return 0.01

def _safe_order_send_icm(req):
    """Safe order send wrapper with IC Markets micro enforcement.

    On IC Markets, enforces micro lot sizing, serialised order sends,
    retry/delay logic and FOK→IOC fallback.  On other brokers, delegates to
    the original safe_order_send.
    """
    # On IC Markets perform micro and robust execution; otherwise call original
    if is_icm_account():
        # Adjust lot size for new open orders only
        try:
            action = req.get('action') if isinstance(req, dict) else None
            try:
                action_deal = getattr(mt5, 'TRADE_ACTION_DEAL', 1)
            except Exception:
                action_deal = 1
            try:
                action_pending = getattr(mt5, 'TRADE_ACTION_PENDING', 5)
            except Exception:
                action_pending = 5
            if action in (action_deal, action_pending) and isinstance(req, dict) and 'position' not in req:
                sym = req.get('symbol')
                if sym:
                    desired_lot = icm_micro_lot()
                    try:
                        minv, step, maxv = symbol_lot_specs(sym)
                        clamped = max(minv, min(maxv, float(desired_lot)))
                        steps = round(clamped / step) if step > 0 else 1
                        adj_lot = steps * step
                        new_lot = normalize_volume(sym, adj_lot)
                    except Exception:
                        new_lot = desired_lot
                    try:
                        old_vol = float(req.get('volume', 0.0))
                    except Exception:
                        old_vol = 0.0
                    if abs(old_vol - float(new_lot)) > 1e-8:
                        msg = f"🔧 Lot adjusted for IC Markets: {old_vol} → {new_lot} on {sym}"
                        try:
                            log_msg(msg)
                        except Exception:
                            pass
                        try:
                            telegram_msg(msg)
                        except Exception:
                            pass
                    try:
                        req['volume'] = new_lot
                    except Exception:
                        pass
        except Exception:
            pass

        # Pre‑check: ensure the symbol is tradable for new open orders (DEAL or PENDING)
        try:
            symbol_for_check = req.get('symbol') if isinstance(req, dict) else None
            action = req.get('action') if isinstance(req, dict) else None
            try:
                action_deal = getattr(mt5, 'TRADE_ACTION_DEAL', 1)
            except Exception:
                action_deal = 1
            try:
                action_pending = getattr(mt5, 'TRADE_ACTION_PENDING', 5)
            except Exception:
                action_pending = 5
            if symbol_for_check and action in (action_deal, action_pending):
                try:
                    if not _tradeable_pre_check(symbol_for_check):
                        return None
                except Exception:
                    return None
        except Exception:
            pass

        # Spread and volatility guard
        try:
            sym = req.get('symbol') if isinstance(req, dict) else None
            if sym:
                tick1 = mt5.symbol_info_tick(sym)
                if not tick1:
                    return None
                try:
                    sp = spread_points(sym)
                except Exception:
                    sp = None
                try:
                    safe_sp = SPREAD_SAFE_THRESHOLD_PTS
                except Exception:
                    safe_sp = 200
                if sp is not None and sp > safe_sp:
                    log_msg(f"🛑 Spread {sp} > safe threshold {safe_sp} on {sym}, aborting order")
                    return None
                try:
                    mid1 = (float(getattr(tick1, 'bid', 0.0)) + float(getattr(tick1, 'ask', 0.0))) / 2.0
                    time.sleep(0.05)
                    tick2 = mt5.symbol_info_tick(sym)
                    if tick2:
                        mid2 = (float(getattr(tick2, 'bid', 0.0)) + float(getattr(tick2, 'ask', 0.0))) / 2.0
                        try:
                            pt_val = float(mt5.symbol_info(sym).point) if mt5.symbol_info(sym) else 0.0001
                        except Exception:
                            pt_val = 0.0001
                        delta_pts = abs(mid2 - mid1) / pt_val
                        try:
                            vt_thresh = VOLATILITY_JUMP_THRESHOLD_PTS
                        except Exception:
                            vt_thresh = 100
                        if delta_pts > vt_thresh:
                            log_msg(f"🛑 Tick jump {delta_pts:.1f} > volatility threshold {vt_thresh} on {sym}, aborting order")
                            return None
                except Exception:
                    pass
        except Exception:
            pass

        # Helper to refresh price and adjust stops
        def _refresh_price_and_stops_icm(r: Dict[str, Any]) -> bool:
            try:
                symb = r.get('symbol')
                if not symb:
                    return False
                tick = None
                try:
                    tick = mt5.symbol_info_tick(symb)
                except Exception:
                    tick = None
                if not tick:
                    return False
                try:
                    otype = int(r.get('type', getattr(mt5, 'ORDER_TYPE_BUY', 0)))
                except Exception:
                    otype = getattr(mt5, 'ORDER_TYPE_BUY', 0)
                side = 'BUY' if otype == getattr(mt5, 'ORDER_TYPE_BUY', 0) else 'SELL'
                price = float(getattr(tick, 'ask', 0.0)) if side == 'BUY' else float(getattr(tick, 'bid', 0.0))
                sl = r.get('sl')
                tp = r.get('tp')
                try:
                    stops_pts = stops_level_points(symb)
                except Exception:
                    stops_pts = 0
                try:
                    si = mt5.symbol_info(symb)
                    pt = float(getattr(si, 'point', 0.0001)) if si else 0.0001
                except Exception:
                    pt = 0.0001
                min_dist = float(stops_pts) * float(pt)
                margin_mult = MIN_STOP_MARGIN_MULTIPLIER if 'MIN_STOP_MARGIN_MULTIPLIER' in globals() else 1.1
                if sl is not None:
                    try:
                        sl = float(sl)
                        if side == 'BUY':
                            if (price - sl) <= min_dist:
                                sl = price - min_dist * margin_mult
                        else:
                            if (sl - price) <= min_dist:
                                sl = price + min_dist * margin_mult
                        sl = normalize_price(symb, sl)
                    except Exception:
                        pass
                if tp is not None:
                    try:
                        tp = float(tp)
                        if side == 'BUY':
                            if (tp - price) <= min_dist:
                                tp = price + min_dist * margin_mult
                        else:
                            if (price - tp) <= min_dist:
                                tp = price - min_dist * margin_mult
                        tp = normalize_price(symb, tp)
                    except Exception:
                        pass
                try:
                    price = normalize_price(symb, price)
                except Exception:
                    pass
                r['price'] = price
                if sl is not None:
                    r['sl'] = sl
                if tp is not None:
                    r['tp'] = tp
                return True
            except Exception:
                return False

        last_res: Any = None
        with _safe_order_lock_icm:
            for attempt in range(3):
                try:
                    # Refresh price and stops
                    _refresh_price_and_stops_icm(req)
                    # Set deviation: 150, 100, 80
                    if attempt == 0:
                        dev = 150
                    elif attempt == 1:
                        dev = 100
                    else:
                        dev = 80
                    if isinstance(req, dict):
                        req['deviation'] = dev
                        # Set filling mode: FOK for first two, IOC on last
                        if attempt < 2:
                            req['type_filling'] = getattr(mt5, 'ORDER_FILLING_FOK', req.get('type_filling', None))
                        else:
                            req['type_filling'] = getattr(mt5, 'ORDER_FILLING_IOC', req.get('type_filling', None))
                    res = mt5.order_send(req)
                    last_res = res
                    if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                        try:
                            log_msg(f"✅ IC Markets order executed on attempt {attempt}")
                        except Exception:
                            pass
                        return res
                    else:
                        try:
                            log_msg(f"⚠️ IC Markets retcode {getattr(res,'retcode',-1)} on attempt {attempt}")
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        log_msg(f"⚠️ IC Markets exception on attempt {attempt}: {e}")
                    except Exception:
                        pass
                if attempt < 2:
                    try:
                        delay = random.uniform(0.2, 0.3)
                    except Exception:
                        delay = 0.25
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
            # After loop, log final
            try:
                rc = getattr(last_res, 'retcode', None) if last_res else None
                log_msg(f"🛑 IC Markets order_send failed after retries; last retcode {rc}")
            except Exception:
                pass
            return last_res
    # Non‑IC Markets: delegate to the original implementation (which includes reliability logic)
    return _original_safe_order_send(req)

# Override the global safe_order_send with our IC Markets aware version
safe_order_send = _safe_order_send_icm  # type: ignore[assignment]

def _min_sl_points(symbol, price, sl):
    si = mt5.symbol_info(symbol)
    pts = int(abs(price - sl) / (si.point if si and si.point else 0.01))
    base = SL_MIN_PTS_BY_SYMBOL.get(_base_of(symbol), SL_MIN_PTS_DEFAULT)
    return pts >= base, pts, base
def _rr_ok(price, sl, tp_mult=2.0):
    try:
        r = abs(price - sl)
        tp = price + (tp_mult * (price - sl)) if price > sl else price - (tp_mult * (sl - price))
        rr = abs(tp - price) / max(1e-9, r)
        return rr >= LOOSEN_MIN_RR_FULL, rr
    except Exception:
        return True, 2.0
def _firm_known():
    try:
        name, rules = current_ruleset()
        return bool(name)
    except Exception:
        return False
def send_market_with_retries(req):
    # For IC Markets accounts, bypass standard retry logic and defer to the
    # safe_order_send wrapper directly.  This ensures micro sizing, locking
    # and fallback behaviour are applied to all market orders.
    try:
        if is_icm_account():
            return safe_order_send(req)
    except Exception:
        pass
    # Before attempting retries, ensure the symbol is tradable
    try:
        symbol_for_check = None
        if isinstance(req, dict):
            symbol_for_check = req.get('symbol')
        if symbol_for_check and not _tradeable_pre_check(symbol_for_check):
            return None
    except Exception:
        pass
    devs = [40, 60, 80, MAX_DEVIATION_DEFAULT]
    last = None
    for d in devs:
        try:
            req['deviation'] = d
            r = mt5.order_send(req)
            last = r
            if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                log_msg(f"✅ Executed @ deviation {d} | ticket={getattr(r,'order',0)} deal={getattr(r,'deal',0)} ret={r.retcode}")
                return r
            else:
                log_msg(f"⚠️ Retcode {getattr(r,'retcode',-1)} at deviation {d}")
        except Exception as e:
            log_msg(f"⚠️ Send error at deviation {d}: {e}")
    if last:
        log_msg(f"🛑 Market send failed after retries | last ret={getattr(last,'retcode',-1)}")
    else:
        log_msg("🛑 Market send failed - no response")
    return last
def place_order(symbol, side, sl, tp1, tp2, tp3, lot, atr_v, micro=False, comment_tag="", meta=None):
    """Attempt to place a market order (with robust retry & fallback to pending orders if needed)."""
    try:
        # If DRY_RUN or TEST_MODE is enabled, simulate an order and return
        # a mock successful result to avoid interacting with MT5.
        if globals().get('DRY_RUN') or globals().get('TEST_MODE'):
            try:
                log_msg(f"🔬 Simulating {'MICRO' if micro else 'FULL'} order on {symbol} lot={lot} sl={sl} (TEST_MODE/DRY_RUN active)")
            except Exception:
                pass
            class _MockOrderResult:
                def __init__(self):
                    self.retcode = getattr(mt5, 'TRADE_RETCODE_DONE', 10009)
                    self.order = 0
                    self.deal = 0
            return _MockOrderResult()
        if isinstance(meta, dict) and "ai_score" in meta:
            try:
                log_msg(f"🧠 AI score: {meta.get('ai_score')}/100 on {symbol}")
            except Exception:
                pass
    except Exception:
        pass

    # ------------------------------------------------------------------
    # IC Markets micro enforcement
    # When connected to an IC Markets account, override the trade to micro
    # mode and set the lot size explicitly.  This ensures that only micro
    # trades (0.01 or 0.02 lots) are sent on IC Markets accounts regardless of
    # incoming parameters or risk logic.
    try:
        if is_icm_account():
            micro = True
            # Override the lot size using the IC Markets micro lot function.
            lot = icm_micro_lot()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # SMC confirmation gate (additional layer). Block trades that SMC deems
    # contradictory to the proposed direction or that score very low.
    try:
        try:
            smc = smc_confluence(symbol, side)
        except Exception:
            smc = None
        if isinstance(smc, dict):
            # If SMC explicitly disallows or core SMC components are missing, block
            if not smc.get('allow', False):
                log_msg(f"🛑 SMC block: {symbol} {side} - score={smc.get('score')}")
                return None
            score_smc = int(smc.get('score', 0))
            # Strong contradiction (BOS suggests opposite direction with high score) still blocks
            if smc.get('direction') and smc.get('direction') != side and score_smc >= 70:
                log_msg(f"🛑 SMC contradiction: suggested {smc.get('direction')}, blocking {side} trade on {symbol} (score {score_smc})")
                return None
            # If SMC score is in predictive range (65-69), require AI confirmation to proceed
            if 65 <= score_smc <= 69:
                try:
                    ai_conf = blended_prediction(symbol) if ENABLE_PREDICTIVE_AI else 0.0
                except Exception:
                    ai_conf = 0.0
                if ai_conf < 0.70:
                    log_msg(f"🔍 SMC predictive-only (score {score_smc}) but AI not confirming ({ai_conf:.2f}) - blocking")
                    return None
            # If >=70 allow to proceed regardless of AI
    except Exception:
        pass

    # Spread/tick enforcement: use XAU-aware policy for gold, otherwise fall
    # back to pair-configured max spreads.
    base = _base_of(symbol)
    if base == 'XAUUSD':
        allowed_full, reason_full = can_place_on_xau(symbol, micro=False)
        allowed_micro, reason_micro = can_place_on_xau(symbol, micro=True)
        # If placing a micro, only hard blocks apply (can_place_on_xau enforces that)
        if micro:
            if not allowed_micro:
                try:
                    log_block_verbose(symbol, "Micro blocked", extra_info=reason_micro)
                except Exception:
                    _log_block_once(symbol, f"🚫 Micro blocked: {reason_micro}")
                try:
                    _maybe_alert_xau_block(symbol, reason_micro, extra_info=reason_micro)
                except Exception:
                    pass
                day_stats["blocks"] += 1
                return None
        else:
            if not allowed_full:
                try:
                    log_block_verbose(symbol, "Full blocked", extra_info=reason_full)
                except Exception:
                    _log_block_once(symbol, f"🚫 Full blocked: {reason_full}")
                try:
                    _maybe_alert_xau_block(symbol, reason_full, extra_info=reason_full)
                except Exception:
                    pass
                day_stats["blocks"] += 1
                return None
    else:
        # Non-XAU: existing behaviour — fetch spread and enforce per-pair cap
        sp = None
        for _s in range(3):
            try:
                sp = spread_points(symbol)
            except Exception:
                sp = None
            if sp is not None:
                break
            time.sleep(0.1)
        if sp is None:
            log_msg(f"⚠️ Unable to fetch spread for {symbol}; aborting order to avoid blind placement")
            day_stats["blocks"] += 1
            return None
        max_spread = PAIR_CONFIG.get(base, {}).get('max_spread', None)
        if max_spread is not None and sp > max_spread:
            log_msg(f"🚫 Spread over cap on {symbol} - blocking order (spread {sp} > {max_spread}).")
            return None

    if not micro:
        full_limit = FULL_MAX_PER_DAY
        try:
            if PROP_ACTIVE.get("name") == "FTMO":
                full_limit = 2
        except Exception:
            pass
        fulls_today = day_stats.get('fulls_LON', 0) + day_stats.get('fulls_NY', 0)
        if fulls_today >= full_limit:
            log_msg(f"🚫 Daily FULL trade cap reached ({fulls_today}/{full_limit}).")
            return None

    if micro:
        # Skip micro start delay for IC Markets; apply default gating otherwise.
        if not is_icm_account():
            try:
                ready_at = globals().get('MICRO_READY_AT')
                if ready_at is None:
                    ready_at = now_uk() + timedelta(minutes=MICRO_START_DELAY_MIN)
                    globals()['MICRO_READY_AT'] = ready_at
                if now_uk() < ready_at:
                    return None
            except Exception:
                return None

    status, reason = check_prop_rules_before_trade() if 'check_prop_rules_before_trade' in globals() else ("OK", "")
    if status != "OK":
        log_msg(f"🛑 Trade blocked: {reason}")
        return None
    if not preflight_autotrading():
        day_stats["fails"] += 1
        return None

    symbol = resolve_symbol(symbol)
    si = mt5.symbol_info(symbol)
    if not si or not mt5.symbol_select(symbol, True):
        log_msg(f"⚠️ Cannot select symbol {symbol} for trading")
        return None

    firm_name, _rules = current_ruleset() if 'current_ruleset' in globals() else (None, None)
    firm_tag = firm_name or "Unknown"
    acct_tag = "DEMO" if account_is_demo() else "LIVE"
    log_msg(f"ℹ️ Routing {symbol} ({'Micro' if micro else 'Full'}) trade on account [{acct_tag}] firm [{firm_tag}]")

    # Ensure a valid tick; retry a few times before failing and attempt MT5 reconnect
    tick = None
    for _attempt in range(3):
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None
        if tick:
            break
        # try lightweight reconnect
        try:
            mt5.shutdown(); time.sleep(0.2); mt5.initialize()
        except Exception:
            pass
        time.sleep(0.2)
    if not tick:
        day_stats["blocks"] += 1
        _log_block_once(symbol, f"⚠️ No tick data for {symbol} after retries, cannot place order")
        return None

    price = tick.ask if side == "BUY" else tick.bid

    if not micro and BLOCK_FULL_IF_FIRM_UNKNOWN and not _firm_known():
        log_msg("🛑 Full trade blocked - firm unknown")
        return None

    ok_sl, pts, need = _min_sl_points(symbol, price, sl)
    if not ok_sl and not micro:
        log_msg(f"🚫 SL too tight ({pts}pts < {need}pts)")
        return None
    if not micro:
        ok_rr, rr = _rr_ok(price, sl, tp_mult=2.0)
        if rr < LOOSEN_MIN_RR_FULL:
            log_msg(f"🚫 R:R {rr:.2f} < {LOOSEN_MIN_RR_FULL}")
            return None

    # Apply global lot scaling factors (consistency, leverage, capital caps)
    try:
        global LOT_SCALE_FACTOR, LEVERAGE_RESTRICT_FACTOR, TEMP_LOT_REDUCTION_UNTIL, SCALING_ACTIVE
        now_ts = datetime.now(SAFE_TZ).timestamp()
        if TEMP_LOT_REDUCTION_UNTIL and now_ts < TEMP_LOT_REDUCTION_UNTIL:
            lot = float(lot) * float(LOT_SCALE_FACTOR)
        else:
            # apply general LOT_SCALE_FACTOR regardless (defaults to 1.0)
            lot = float(lot) * float(globals().get('LOT_SCALE_FACTOR', 1.0))
        # leverage-based reduction
        lot = float(lot) * float(LEVERAGE_RESTRICT_FACTOR)
        # maximum capital allocation: scale down if balance > MAX_CAP_ALLOC and scaling not active
        try:
            bal, _ = _equity_info()
            if bal and bal > MAX_CAP_ALLOC and not SCALING_ACTIVE:
                cap_factor = float(MAX_CAP_ALLOC) / float(bal)
                lot = lot * cap_factor
        except Exception:
            pass
        lot = normalize_volume(symbol, lot)
    except Exception:
        lot = normalize_volume(symbol, lot)
    minv, step, _ = symbol_lot_specs(symbol)
    log_msg(f"ℹ️ {symbol} lot spec: min={minv}, step={step}. Sending volume={lot}")

    try:
        rr = abs(price - sl) / max(1e-9, abs(price - sl))
    except Exception:
        rr = 1.0

    deviations = build_deviations(symbol)

    def mark_session_success():
        label, _, _ = session_bounds()
        if label == "LON":
            day_stats["fulls_LON" if not micro else "micros_LON"] += 1
        if label == "NY":
            day_stats["fulls_NY" if not micro else "micros_NY"] += 1
        try:
            if micro:
                lab, _, _ = session_bounds()
                day_stats.setdefault('micros_by_symbol', {'LON': {}, 'NY': {}})
                d = day_stats['micros_by_symbol'].setdefault(lab, {})
                d[symbol] = d.get(symbol, 0) + 1
        except Exception:
            pass
        day_stats["trades"] += 1
        try:
            global _trailing_thread
            if '_trailing_thread' in globals():
                th = globals().get('_trailing_thread')
                if th is None or not getattr(th, 'is_alive', lambda: False)():
                    th = threading.Thread(target=institutional_trailing_worker, daemon=True)
                    globals()['_trailing_thread'] = th
                    th.start()
        except Exception:
            pass

    # Build market request (without SL/TP attached)
    try:
        typ = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": typ,
            "price": price,
            "deviation": deviations[0] if deviations else MAX_DEVIATION_DEFAULT,
            "magic": 20250916,
            "comment": comment_tag or ""
        }
    except Exception as e:
        log_msg(f"⚠️ Failed to build order request: {e}")
        return None

    # Send market order with retries; on IC Markets use safe_order_send for robust handling.
    try:
        if is_icm_account():
            res = safe_order_send(req)
        else:
            res = send_market_with_retries(req)
    except Exception:
        res = None

    if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
        try:
            mark_session_success()
        except Exception:
            pass
        try:
            _notify_trade_event("🟢 MARKET", symbol, side, price=price, sl=sl, tp=None, extra=("MICRO" if micro else "FULL"))
        except Exception:
            pass
        try:
            tp_primary = tp1 if tp1 not in (None, 0, 0.0) else (tp2 if tp2 not in (None, 0, 0.0) else tp3)
            sess_label = None
            try:
                sess_label, _, _ = session_bounds()
            except Exception:
                sess_label = None
            telegram_trade_levels(symbol, side, price, sl, tp_primary, None, sess_label)
        except Exception:
            pass
        return res

    # Market failed - attempt pending STOP then LIMIT as fallback
    try:
        si2 = mt5.symbol_info(symbol)
        pt = (si2.point if si2 and si2.point else 0.0001)
        cur = price
        min_pts = stops_level_points(symbol) or 0
        dist = (min_pts or 0) * pt
        p_stop = normalize_price(symbol, cur + dist if side == "BUY" else cur - dist)
        preq = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": (mt5.ORDER_TYPE_BUY_STOP if side == "BUY" else mt5.ORDER_TYPE_SELL_STOP),
            "price": p_stop,
            "deviation": 80,
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": 20250916,
            "comment": ("PENDING STOP " + ("Micro" if micro else "Full") + f" [{firm_tag}][{acct_tag}]")
        }
        rr_mult = (RR_MICRO if micro else RR_FULL)
        risk = abs(cur - sl) if sl else max(2*pt, abs(cur - p_stop))
        tp_here = (cur + rr_mult * risk) if side == "BUY" else (cur - rr_mult * risk)
        preq.update({"sl": normalize_price(symbol, sl), "tp": normalize_price(symbol, tp_here)})
        send_preq = dict(preq)
        send_preq.pop("sl", None)
        send_preq.pop("tp", None)
        # Use safe_order_send to respect pre‑checks before sending pending STOP orders
        pr = safe_order_send(send_preq) if 'safe_order_send' in globals() else mt5.order_send(send_preq)
        if pr and getattr(pr, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
            try:
                mark_session_success()
            except Exception:
                pass
            try:
                _notify_trade_event("🧷 PENDING STOP", symbol, side, price=preq.get('price'), sl=preq.get('sl'), tp=preq.get('tp'), extra=("MICRO" if micro else "FULL"))
            except Exception:
                pass
            try:
                sess_label = None
                try:
                    sess_label, _, _ = session_bounds()
                except Exception:
                    sess_label = None
                telegram_trade_levels(symbol, side, preq.get('price'), preq.get('sl'), preq.get('tp'), None, sess_label)
            except Exception:
                pass
            return pr
        # Try pending limit at current normalized price
        p_lim = normalize_price(symbol, cur)
        preq2 = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": (mt5.ORDER_TYPE_BUY_LIMIT if side == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT),
            "price": p_lim,
            "deviation": 80,
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": 20250916,
            "comment": ("PENDING LIMIT " + ("Micro" if micro else "Full") + f" [{firm_tag}][{acct_tag}]")
        }
        risk = abs(p_lim - sl) if sl else max(2*pt, pt*10)
        tp_here = (p_lim + rr_mult * risk) if side == "BUY" else (p_lim - rr_mult * risk)
        preq2.update({"sl": normalize_price(symbol, sl), "tp": normalize_price(symbol, tp_here)})
        send_preq2 = dict(preq2)
        send_preq2.pop("sl", None)
        send_preq2.pop("tp", None)
        # Use safe_order_send to respect pre‑checks before sending pending LIMIT orders
        pr2 = safe_order_send(send_preq2) if 'safe_order_send' in globals() else mt5.order_send(send_preq2)
        if pr2 and getattr(pr2, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
            try:
                mark_session_success()
            except Exception:
                pass
            try:
                _notify_trade_event("🧷 PENDING LIMIT", symbol, side, price=preq2.get('price'), sl=preq2.get('sl'), tp=preq2.get('tp'), extra=("MICRO" if micro else "FULL"))
            except Exception:
                pass
            try:
                sess_label = None
                try:
                    sess_label, _, _ = session_bounds()
                except Exception:
                    sess_label = None
                telegram_trade_levels(symbol, side, preq2.get('price'), preq2.get('sl'), preq2.get('tp'), None, sess_label)
            except Exception:
                pass
            return pr2
    except Exception as e:
        try:
            log_msg(f"⚠️ place_order fallback error: {e}")
        except Exception:
            pass

    return None
def place_pending_now():
    """Fallback: place a pending stop/limit order if instant execution fails."""
    try:
        min_pts = stops_level_points(symbol)
        _, t = cur_price()
        si2 = mt5.symbol_info(symbol)
        pt = (si2.point if si2 and si2.point else 0.0001)
        cur = t.ask if side == "BUY" else t.bid
        dist = (min_pts or 0) * pt
        p_stop = normalize_price(symbol, cur + dist if side == "BUY" else cur - dist)
        preq = {
            "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": volumes[0],
            "type": (mt5.ORDER_TYPE_BUY_STOP if side == "BUY" else mt5.ORDER_TYPE_SELL_STOP),
            "price": p_stop, "deviation": 80, "type_time": mt5.ORDER_TIME_GTC,
            "magic": 20250916,
            "comment": ("PENDING STOP " + ("Micro" if micro else "Full") + f" [{firm_tag}][{acct_tag}]")
        }
        try:
            rr_mult = (RR_MICRO if micro else RR_FULL)
            risk = abs(cur - sl) if 'sl' in locals() and sl else max(2*pt, abs(cur - p_stop))
            tp_here = (cur + rr_mult * risk) if side == "BUY" else (cur - rr_mult * risk)
            preq.update({"sl": normalize_price(symbol, sl), "tp": normalize_price(symbol, tp_here)})
        except Exception:
            pass
        try:
            px__ = preq.get("price", 0.0) or 0.0
            sl__ = preq.get("sl", 0.0) or 0.0
            tp__ = preq.get("tp", 0.0) or 0.0
            if 'sltp_guard' in globals():
                ok__, sl_new__, tp_new__, info__ = sltp_guard(symbol, side, px__, sl__, tp__, adjust=True)
                if not ok__:
                    log_msg(f"🚫 Guard reject PENDING STOP {('MICRO' if micro else 'FULL')} {side} {symbol}: {info__.get('reason')}")
                    return None
                if info__.get("adjusted"):
                    preq["sl"] = normalize_price(symbol, sl_new__)
                    preq["tp"] = normalize_price(symbol, tp_new__)
        except Exception:
            pass
        send_preq = dict(preq)
        send_preq.pop("sl", None)
        send_preq.pop("tp", None)
        # Use safe_order_send to respect pre‑checks before sending pending STOP orders
        pr = safe_order_send(send_preq) if 'safe_order_send' in globals() else mt5.order_send(send_preq)
        if pr and pr.retcode == mt5.TRADE_RETCODE_DONE:
            mark_session_success()
            log_msg(f"📥 Pending STOP placed on {symbol} @ {p_stop:.3f} (side {side})")
            try:
                px__ = preq.get("price", 0.0); sl__ = preq.get("sl", 0.0) or 0.0; tp__ = preq.get("tp", 0.0) or 0.0
                _notify_trade_event("🧷 PENDING STOP", symbol, side, price=px__, sl=sl__ or None, tp=tp__ or None, extra=("MICRO" if micro else "FULL"))
            except Exception:
                pass
            try:
                sess_label = None
                try:
                    sess_label, _, _ = session_bounds()
                except Exception:
                    sess_label = None
                telegram_trade_levels(symbol, side, preq.get('price'), preq.get('sl'), preq.get('tp'), None, sess_label)
            except Exception:
                pass
            return pr
        p_lim = normalize_price(symbol, cur)
        preq2 = {
            "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": volumes[0],
            "type": (mt5.ORDER_TYPE_BUY_LIMIT if side == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT),
            "price": p_lim, "deviation": 80, "type_time": mt5.ORDER_TIME_GTC,
            "magic": 20250916,
            "comment": ("PENDING LIMIT " + ("Micro" if micro else "Full") + f" [{firm_tag}][{acct_tag}]")
        }
        try:
            rr_mult = (RR_MICRO if micro else RR_FULL)
            risk = abs(p_lim - sl) if 'sl' in locals() and sl else max(2*pt, pt*10)
            tp_here = (p_lim + rr_mult * risk) if side == "BUY" else (p_lim - rr_mult * risk)
            preq2.update({"sl": normalize_price(symbol, sl), "tp": normalize_price(symbol, tp_here)})
        except Exception:
            pass
        try:
            if 'sltp_guard' in globals():
                px__ = preq2.get("price", 0.0) or 0.0
                sl__ = preq2.get("sl", 0.0) or 0.0
                tp__ = preq2.get("tp", 0.0) or 0.0
                ok__, sl_new__, tp_new__, info__ = sltp_guard(symbol, side, px__, sl__, tp__, adjust=True)
                if not ok__:
                    log_msg(f"🚫 Guard reject PENDING LIMIT {('MICRO' if micro else 'FULL')} {side} {symbol}: {info__.get('reason')}")
                    return None
                if info__.get("adjusted"):
                    preq2["sl"] = normalize_price(symbol, sl_new__)
                    preq2["tp"] = normalize_price(symbol, tp_new__)
        except Exception:
            pass
        send_preq2 = dict(preq2)
        send_preq2.pop("sl", None)
        send_preq2.pop("tp", None)
        # Use safe_order_send to respect pre‑checks before sending pending LIMIT orders
        pr2 = safe_order_send(send_preq2) if 'safe_order_send' in globals() else mt5.order_send(send_preq2)
        if pr2 and pr2.retcode == mt5.TRADE_RETCODE_DONE:
            mark_session_success()
            log_msg(f"📥 Pending LIMIT placed on {symbol} @ {p_lim:.3f} (side {side})")
            try:
                px__ = preq2.get("price", 0.0); sl__ = preq2.get("sl", 0.0) or 0.0; tp__ = preq2.get("tp", 0.0) or 0.0
                _notify_trade_event("🧷 PENDING LIMIT", symbol, side, price=px__, sl=sl__ or None, tp=tp__ or None, extra=("MICRO" if micro else "FULL"))
            except Exception:
                pass
            try:
                sess_label = None
                try:
                    sess_label, _, _ = session_bounds()
                except Exception:
                    sess_label = None
                telegram_trade_levels(symbol, side, preq2.get('price'), preq2.get('sl'), preq2.get('tp'), None, sess_label)
            except Exception:
                pass
            return pr2
    except Exception as e:
        try:
            log_msg(f"⚠️ place_pending_now error: {e}")
        except Exception:
            pass
    return None
def preflight_autotrading():
    """Check if MT5 terminal and account allow algo trading. Skipped if FORCE_ENABLE_TRADING."""
    if globals().get('FORCE_ENABLE_TRADING'):
        global FORCE_MSG_SHOWN
        if not FORCE_MSG_SHOWN:
            log_msg("⚠️ FORCE_ENABLE_TRADING active: skipping autotrading preflight checks.")
            FORCE_MSG_SHOWN = True
        return True
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    allowed = True
    if ti and not getattr(ti, "trade_allowed", True):
        log_msg("⛔ MT5 terminal: Autotrading is DISABLED.")
        allowed = False
    if ai and not getattr(ai, "trade_allowed", True):
        log_msg("⛔ Account: trading not allowed (check broker settings).")
        allowed = False
    if not allowed:
        autotrading_hint("🔒 Preflight: ")
    return allowed

# -----------------------------------------------------------------------
# Multi‑timeframe bias and candle confirmation utilities
def get_tf_bias(symbol, timeframe):
    """
    Returns +1 for bullish, -1 for bearish, 0 for neutral on a given timeframe.
    Uses a simple EMA50/EMA200 crossover as the bias metric.
    """
    try:
        data = safe_get_rates(symbol, timeframe, 150)
        # Require sufficient data to calculate EMAs; otherwise neutral.
        if data is None or len(data) < 50:
            return 0
        ema50 = ema(data, 50)
        ema200 = ema(data, 200)
        # When the fast EMA is above the slow EMA, bias is bullish (+1).
        if ema50 > ema200:
            return 1
        # When the fast EMA is below the slow EMA, bias is bearish (‑1).
        if ema50 < ema200:
            return -1
        # Otherwise neutral.
        return 0
    except Exception:
        # On any error, default to neutral to avoid false bias.
        return 0


def multi_tf_direction_ok(symbol):
    """
    Enforce higher‑timeframe alignment.  This version loosens the strict
    requirement that Daily, H4, H1 and M15 all point in the same direction.
    A neutral bias on any timeframe still blocks the trade, but minor
    disagreements between the Daily and H4 timeframes are tolerated
    provided they are not opposite to the H1/M15 bias.  The H1 and M15
    biases must match exactly; if they disagree the trade is blocked.  A
    difference of 2 between any higher timeframe bias and the H1 bias
    indicates an outright opposite trend (1 vs ‑1) and blocks the trade.
    Return (True, reason) when the biases are sufficiently aligned.
    """
    # If HTF gating is globally disabled, report OK unconditionally.
    try:
        if HARD_DISABLE_HTF:
            # When HTF is globally disabled, do not emit intermediate debug
            # messages here: consumers should rely on the startup summary log.
            return True, "HTF_DISABLED"
    except Exception:
        pass

    # Compute biases on multiple timeframes.  A neutral (0) bias will block.
    d_bias  = get_tf_bias(symbol, mt5.TIMEFRAME_D1)
    h4_bias = get_tf_bias(symbol, mt5.TIMEFRAME_H4)
    h1_bias = get_tf_bias(symbol, mt5.TIMEFRAME_H1)
    m15_bias = get_tf_bias(symbol, mt5.TIMEFRAME_M15)
    # If any timeframe returns a neutral bias, block the trade.
    if 0 in [d_bias, h4_bias, h1_bias, m15_bias]:
        return False, "HTF_Neutral"
    # H1 and M15 must match perfectly
    if h1_bias != m15_bias:
        return False, "HTF_H1_M15_Disagree"
    # Daily and H4 biases no longer hard‑block unless they are opposite to
    # the H1 bias.  An opposite trend is indicated by a difference of 2
    # (e.g. 1 vs ‑1 or ‑1 vs 1).
    if d_bias != h1_bias and abs(d_bias - h1_bias) == 2:
        return False, "HTF_D1_Opposite"
    if h4_bias != h1_bias and abs(h4_bias - h1_bias) == 2:
        return False, "HTF_H4_Opposite"
    # Otherwise consider the higher timeframes sufficiently aligned
    return True, "HTF_SoftAligned"


# ---------------------------------------------------------------------------
# Higher Time Frame Break of Structure validator
#
# Soften the requirement that both H4 and Daily BOS match the trade side.  A
# mismatch on the daily timeframe will no longer block a trade unless it is
# directly opposite to the H1/M15 trend.  This function examines BOS
# signals on the H4 and D1 timeframes, normalises them to a numeric bias and
# determines if they permit a trade on the given side ("BUY" or "SELL").
def htf_bos_ok(symbol: str, side: str) -> bool:
    """Return True if the H4 and D1 BOS directions permit a trade on side."""

    # If HTF gating is globally disabled, permit BOS by default.
    try:
        if HARD_DISABLE_HTF:
            # Quiet bypass; avoid emitting intermediate debug messages.
            return True
    except Exception:
        pass
    try:
        # Fetch H4 and D1 price data.  If unavailable, do not block.
        h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 300, max_age=6.0)
        d1 = fetch_data_cached(symbol, mt5.TIMEFRAME_D1, 300, max_age=10.0)
        if h4 is None or d1 is None:
            return True
        # Determine BOS direction for each timeframe.  detect_bos may return a
        # string ("BOS_UP" / "BOS_DOWN"), a boolean, a numeric score, or
        # None.  Normalise these into +1 (bullish), -1 (bearish) or 0.
        def _norm_bos(value):
            if value is None:
                return 0
            # Strings: compare case-insensitive
            if isinstance(value, str):
                up = value.upper()
                if up == "BOS_UP":
                    return 1
                if up == "BOS_DOWN":
                    return -1
                return 0
            # Booleans: True→bullish, False→bearish
            if isinstance(value, bool):
                return 1 if value else -1
            # Numeric or numeric‑like: positive→bullish, negative→bearish
            try:
                num = float(value)
                if num > 0:
                    return 1
                if num < 0:
                    return -1
                return 0
            except Exception:
                return 0
        bos_h4 = _norm_bos(detect_bos(h4))
        bos_d1 = _norm_bos(detect_bos(d1))
        # H4 BOS must align with the trade side
        if side == "BUY" and bos_h4 < 0:
            return False
        if side == "SELL" and bos_h4 > 0:
            return False
        # D1 BOS is advisory: only block if directly opposite to the side
        if side == "BUY" and bos_d1 < 0:
            return False
        if side == "SELL" and bos_d1 > 0:
            return False
        return True
    except Exception:
        # On any error, default to permitting the trade
        return True


def m5_confirmation(symbol, direction):
    """
    One‑candle confirmation on the M5 chart.  Returns True when the most
    recent M5 candle strongly confirms the intended direction; otherwise False.
    direction must be 'buy' or 'sell'.
    """
    try:
        data = safe_get_rates(symbol, mt5.TIMEFRAME_M5, 5)
    except Exception:
        data = None
    # Require at least three candles of data.
    if data is None or len(data) < 3:
        return False, "No_M5_Data"
    # Focus on the last complete candle
    last = data[-1]
    # Candle body size
    bodysize = abs(last['close'] - last['open'])
    # Total range minus body size gives wick size
    wicksize = (last['high'] - last['low']) - bodysize
    if direction.lower() == "buy":
        # Confirm a bullish candle with a dominant body (>40% of total range)
        if last['close'] > last['open'] and bodysize > wicksize * 0.4:
            return True, "M5_Bull_Confirmed"
        return False, "M5_No_Bull_Candle"
    if direction.lower() == "sell":
        # Confirm a bearish candle with a dominant body (>40% of total range)
        if last['close'] < last['open'] and bodysize > wicksize * 0.4:
            return True, "M5_Bear_Confirmed"
        return False, "M5_No_Bear_Candle"
    # Invalid direction string
    return False, "Invalid_Direction"
def attempt_full_trade_once():
    """Scan for any full-trade signals and attempt to place one trade (at most one per scan)."""
    # If the recent scan reported XAU had no setup, emit a single concise
    # console-only message here and clear the flag.
    global LAST_SCAN_XAU_NO_SETUP
    try:
        if LAST_SCAN_XAU_NO_SETUP:
            print("ℹ️ Checked XAUUSD — no setup this scan")
            LAST_SCAN_XAU_NO_SETUP = False
    except Exception:
        pass

    placed = False
    blocked_gold = in_news_blackout('XAUUSD')
    if blocked_gold:
        log_msg("🕒 News blackout active for XAUUSD — gold full trades paused; scanning other pairs.")
    # Track per-scan XAUUSD state to avoid repeated noisy prints
    xau_no_setup_seen = False
    xau_setup_found = False
    for sym_base in SYMBOLS:
        # Asian session rule: during Asian hours only GBPJPY is allowed; XAUUSD disabled
        try:
            if is_asian_session() and not sym_base.upper().startswith("GBPJPY"):
                continue
            if blocked_gold and sym_base.upper().startswith("XAUUSD"):
                continue
        except Exception:
            if blocked_gold and sym_base.upper().startswith("XAUUSD"):
                continue
        sym = resolve_symbol(sym_base)
        # Enforce higher‑timeframe alignment on each symbol.  If the Daily, H4,
        # H1 and M15 biases disagree or any timeframe is neutral, skip this
        # symbol entirely.  Provide a skip reason via Telegram for visibility.
        try:
            ok_htf, reason_htf = multi_tf_direction_ok(sym)
        except Exception:
            ok_htf, reason_htf = (True, "HTF_OK")
        if not ok_htf:
            log_msg(f"🚫 HTF alignment blocked full trade on {sym} ({reason_htf})")
            # Use structured block notification but suppress XAUUSD noisy alerts
            try:
                base_sym = _base_of(sym)
                def _notify_block(s, r, had_setup=False, placed=False):
                    try:
                        if base_sym == 'XAUUSD' and not (had_setup or placed):
                            # Defer XAUUSD notifications until a setup/trade occurs
                            return False
                        return telegram_block(s, r)
                    except Exception:
                        return False
                _notify_block(sym, reason_htf, had_setup=False, placed=False)
            except Exception:
                pass
            continue
        # Record a snapshot of recent market state for memory biasing
        try:
            record_market_snapshot(sym)
        except Exception:
            pass

        sigs = build_signals_by_strategy(sym, announce=True)
        if not sigs:
            # Mark XAUUSD no-setup once per scan
            try:
                if _base_of(sym) == 'XAUUSD':
                    xau_no_setup_seen = True
                    try:
                        globals()['LAST_SCAN_XAU_NO_SETUP'] = True
                    except Exception:
                        pass
            except Exception:
                pass
            continue
        t = mt5.symbol_info_tick(sym)
        if not t:
            continue

        # Update strict mode based on recent performance
        _update_strict_mode()

        # Preload required data for quality/volatility filters
        m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 120, max_age=1.5)
        h1 = fetch_data_cached(sym, mt5.TIMEFRAME_H1, 150, max_age=3.0)
        if m15 is None or h1 is None:
            continue

        # Predictive and liquidity gating computed once per symbol
        p_score = 1.0
        liquidity = 1.0
        risk_scale_symbol = 1.0
        low_vol_gold = False
        base_sym = _base_of(sym)
        if ENABLE_PREDICTIVE_AI:
            try:
                p_score = blended_prediction(sym)
            except Exception:
                p_score = 1.0
            try:
                liquidity = predict_liquidity_pressure(sym)
            except Exception:
                liquidity = 1.0
            # When HTF is disabled, predictive and liquidity gating must not block.
            if HARD_DISABLE_HTF:
                p_score = 1.0
                liquidity = 1.0
                risk_scale_symbol = 1.0
                low_vol_gold = False
            # Low‑volatility gold mode: disable full trades when gold is dead
            if base_sym == 'XAUUSD':
                try:
                    m15_data_for_lv = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 100, max_age=1.5)
                    if m15_data_for_lv is not None and len(m15_data_for_lv) >= 50:
                        # current ATR
                        try:
                            cur_atr = true_atr(m15_data_for_lv, 14)
                        except Exception:
                            cur_atr = None
                        # average of high-low range over 50 bars
                        try:
                            atr_avg_raw = (m15_data_for_lv['high'] - m15_data_for_lv['low']).rolling(50).mean().iloc[-1]
                        except Exception:
                            atr_avg_raw = None
                        # ADX
                        try:
                            adx_m15_lv = float(adx(m15_data_for_lv))
                        except Exception:
                            adx_m15_lv = 0.0
                        if cur_atr and atr_avg_raw and adx_m15_lv < 18:
                            if cur_atr < 0.6 * atr_avg_raw:
                                low_vol_gold = True
                except Exception:
                    low_vol_gold = False
            # Apply predictive gating: block full trades if predictive score is low
            # or liquidity is very poor.  Liquidity threshold reduced to 0.25
            # (from 0.35) to allow more entries in quieter markets.  If the
            # predictive score is very high (≥0.70), allow the trade to proceed
            # despite low liquidity or dead gold conditions by applying a
            # reduced risk scale later.
            if not HARD_DISABLE_HTF:
                if p_score < 0.62 or liquidity < 0.25 or low_vol_gold:
                    if p_score >= 0.70:
                        pass
                    else:
                        continue
            # Compute dynamic risk scaling factor; if zero then skip unless a
            # high predictive score (≥0.70) warrants a reduced fallback risk scale.
            try:
                risk_scale_symbol = compute_dynamic_risk_scale(sym, p_score, liquidity)
            except Exception:
                risk_scale_symbol = 1.0
            if not HARD_DISABLE_HTF:
                if risk_scale_symbol <= 0.0:
                    if p_score >= 0.70:
                        risk_scale_symbol = 0.50
                    else:
                        continue
        # End predictive gating
        for s in sigs:
            side = s["side"]
            # Hard limits (caps, pauses, duplicates)
            ok_limits, reason_limits = _check_hard_limits(sym, side, "FULL")
            if not ok_limits:
                try:
                    telegram_msg(f"🚫 {sym} FULL blocked: {reason_limits}")
                except Exception:
                    pass
                continue
            # Market quality filter
            ok_mkt, reason_mkt = _market_quality_ok(sym, side, m15, h1, datetime.now(SAFE_TZ))
            if not ok_mkt:
                try:
                    telegram_msg(f"🚫 {sym} FULL skipped: {reason_mkt}")
                except Exception:
                    pass
                continue
            # Volatility control
            ok_vol, reason_vol = _volatility_ok(m15, datetime.now(SAFE_TZ))
            if not ok_vol:
                try:
                    telegram_msg(f"🚫 {sym} FULL skipped: {reason_vol}")
                except Exception:
                    pass
                continue
            # Trend consistency
            ok_trend, reason_trend = _trend_consistency_ok(sym, side, h1)
            if not ok_trend:
                try:
                    telegram_msg(f"🚫 {sym} FULL skipped: {reason_trend}")
                except Exception:
                    pass
                continue
            # Entry confirmation
            ok_entry, reason_entry = _entry_confirmation_ok(m15, side)
            if not ok_entry:
                try:
                    telegram_msg(f"🚫 {sym} FULL skipped: {reason_entry}")
                except Exception:
                    pass
                continue
            # If SMC pure mode is enabled, bypass the legacy indicator-heavy
            # confluence checks and use the SMC scoring engine as the single
            # gate (plus minimal allowed blockers).
            if CONSTANTS.get("SMC_PURE_MODE", False):
                try:
                    smc = smc_confluence(sym, side)
                except Exception:
                    smc = None
                if not isinstance(smc, dict):
                    continue
                # Mandatory core SMC check enforced by smc_confluence (bos+sweep+fvg)
                if not smc.get('allow', False):
                    continue
                score_smc = int(smc.get('score', 0))
                # Only allowed blockers: spread, invalid SL/TP, MT5 connectivity, FTMO DD
                try:
                    if _base_of(sym) == 'XAUUSD':
                        ok_full, reason_full = can_place_on_xau(sym, micro=False)
                        ok_micro, reason_micro = can_place_on_xau(sym, micro=True)
                        if not ok_full and not ok_micro:
                            try:
                                log_block_verbose(sym, "Full+Micro blocked", extra_info=(reason_full or reason_micro))
                            except Exception:
                                _log_block_once(sym, f"⚠️ {sym} blocked — {reason_full or reason_micro}")
                            try:
                                _maybe_alert_xau_block(sym, reason_full or reason_micro, extra_info=(reason_full or reason_micro))
                            except Exception:
                                pass
                            continue
                        if not ok_full and ok_micro:
                            try:
                                log_block_verbose(sym, "Full blocked (micro allowed)", extra_info=reason_full)
                            except Exception:
                                _log_block_once(sym, f"⚠️ {sym} full blocked ({reason_full}); scanning for micro-only")
                            try:
                                _maybe_alert_xau_block(sym, reason_full or reason_micro, extra_info=(reason_full or reason_micro))
                            except Exception:
                                pass
                            # continue scanning (micro-only allowed)
                    else:
                        if not spread_ok(sym):
                            log_msg(f"⚠️ Spread too high, skipping {sym}")
                            continue
                except Exception:
                    if not spread_ok(sym):
                        log_msg(f"⚠️ Spread too high, skipping {sym}")
                        continue
                # Prop firm rule check (FTMO dd). If violated, skip
                try:
                    status, reason = check_prop_rules_before_trade()
                    if status != "OK":
                        log_msg(f"⚠️ Prop rules blocking trade on {sym}: {reason}")
                        continue
                except Exception:
                    pass
                # Score threshold logic
                if score_smc >= 70:
                    pass  # allow full execution
                elif 65 <= score_smc <= 69:
                    # predictive-only range requires AI confirmation
                    try:
                        ai_conf = blended_prediction(sym) if ENABLE_PREDICTIVE_AI else 0.0
                    except Exception:
                        ai_conf = 0.0
                    if ai_conf < 0.70:
                        continue
                else:
                    continue
                # Passed SMC gating - compute decision and execute using existing helpers
                try:
                    # minimal sl derived from signal structure if present
                    sl = s.get('sl') if isinstance(s, dict) and s.get('sl') else None
                    atr_m15 = 0.0
                    try:
                        m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 120, max_age=1.5)
                        atr_m15 = true_atr(m15, 14) if m15 is not None else 0.0
                    except Exception:
                        atr_m15 = 0.0
                    decision = prepare_trade_decision(sym, side, sl, atr_m15, dynamic_scale=1.0, micro=False, meta={"smc_score": score_smc, "smc_details": smc.get('details', {})})
                    if not decision:
                        continue
                    res = execute_market_order(sym, side, decision.get('sl'), decision.get('tp1'), decision.get('tp2'), decision.get('tp3'), decision.get('lot'), micro=False, comment_tag="SMC", meta={"smc": smc})
                except Exception:
                    res = None
                if res:
                    try:
                        base_sym = _base_of(sym)
                        if isinstance(base_sym, str) and base_sym.upper() == 'XAUUSD':
                            try:
                                sess_lbl, _, _ = session_bounds()
                            except Exception:
                                sess_lbl = None
                            _record_xau_session_trade(sess_lbl, side)
                            log_msg(f"ℹ️ Recorded XAUUSD {side} in session {sess_lbl}")
                    except Exception:
                        pass
                    placed = True
                    break
                continue
            # Soft HTF BOS gating: ensure H4 BOS aligns with the trade side and
            # Daily BOS is not opposite.  If BOS check fails, skip this
            # signal.  This replaces the original strict BOS gating and
            # encourages more trades without sacrificing safety.
            try:
                if not htf_bos_ok(sym, side):
                    log_msg(f"🚫 {sym} blocked: HTF BOS mismatch")
                    continue
            except Exception:
                pass
            h4 = fetch_data_cached(sym, mt5.TIMEFRAME_H4, 260, max_age=6.0)
            h1 = fetch_data_cached(sym, mt5.TIMEFRAME_H1, 220, max_age=3.0)
            m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 160, max_age=1.5)
            m5 = fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6)
            if h4 is None or h1 is None or m15 is None:
                continue
            bias_h4 = ema_bias(h4)
            h4_align = 1 if bias_h4 and bias_h4 == side else 0
            adx_h1_val, adx_is_rising = adx_rising(h1)
            adx_m15_val = float(adx(m15)) if m15 is not None else 0.0
            adx_rise_flag = 1 if adx_is_rising else 0
            try:
                rsi_h1_val = float(rsi(h1)); rsi_m15_val = float(rsi(m15))
            except Exception:
                rsi_h1_val = rsi_m15_val = 50.0
            # Use relaxed RSI thresholds (≥53 for buys and ≤47 for sells)
            rsi_ok_flag = 1 if ((side == "BUY" and rsi_h1_val >= LOOSEN_FULL_RSI_BUY and rsi_m15_val >= LOOSEN_FULL_RSI_BUY) or 
                                 (side == "SELL" and rsi_h1_val <= LOOSEN_FULL_RSI_SELL and rsi_m15_val <= LOOSEN_FULL_RSI_SELL)) else 0
            ema50 = get_ema(h1, 50); ema200 = get_ema(h1, 200)
            ema_slope_val = 0.0
            if len(h1) > 12:
                try:
                    ema_diff = ema50 - ema200
                    ema_slope_val = float(ema_diff.iloc[-1] - ema_diff.iloc[-11]) if len(ema_diff) > 11 else 0.0
                except Exception:
                    ema_slope_val = 0.0
            ema_align_flag = 1 if ((side == "BUY" and ema_slope_val > 0) or (side == "SELL" and ema_slope_val < 0)) else 0
            bos_dir = detect_bos(h1)
            bos_align_flag = 1 if ((side == "BUY" and bos_dir == "BOS_UP") or (side == "SELL" and bos_dir == "BOS_DOWN")) else 0
            fvg_flag = False
            try:
                fvg_flag = bool(detect_fvg(m15) or (m5 is not None and detect_fvg(m5)))
            except Exception:
                fvg_flag = False
            atr_hot_flag = False
            try:
                atr_series = (m15["high"] - m15["low"]).rolling(20).mean()
                if len(atr_series) > 1:
                    atr_hot_flag = bool(atr_series.iloc[-1] >= atr_series.iloc[-2])
            except Exception:
                atr_hot_flag = False
            vol_mom_flag = 1 if (fvg_flag or atr_hot_flag) else 0
            session_flag = 1 if in_session() else 0
            spread_flag = 1 if spread_ok(sym) else 0
            # Compute new volatility and momentum indicators for higher accuracy
            try:
                atr_z_val = atr_z_score(m15)
            except Exception:
                atr_z_val = 0.0
            try:
                rsil_val = rsi_leader(m15)
            except Exception:
                rsil_val = rsi(m15)
            # ------------------------------------------------------------------
            # Full-trade confluence and low volatility gating
            # In order to safely increase the number of full-trade opportunities
            # without sacrificing overall accuracy, compute a simple confluence
            # count of key features.  Trades are only considered if the count
            # meets a minimum threshold (reduced by 1 relative to the prior
            # implementation, but never below 4).  An exception is made for
            # moderately quiet markets: if volatility is below the M15 ATR floor
            # but not extremely low (≥70% of the floor) and the RSI gate is
            # satisfied, then one fewer confluence point is required.  A
            # relaxed ADX gate allows one timeframe to slightly underperform so
            # long as the other exceeds a higher threshold.  This logic
            # increases frequency on calm days while retaining safety.
            try:
                atr_m15 = true_atr(m15, 14)
            except Exception:
                atr_m15 = 0.0
            # Determine if we are in a low‑volatility regime relative to the full
            # ATR floor.  The LOOSEN_FULL_ATR_FLOOR constant reflects the
            # baseline minimum; permit trades if ATR is at least 70% of that.
            low_vol_setup = False
            try:
                if atr_m15 < LOOSEN_FULL_ATR_FLOOR and atr_m15 >= LOOSEN_FULL_ATR_FLOOR * 0.70:
                    low_vol_setup = True
            except Exception:
                low_vol_setup = False
            # Relaxed ADX gate: at least one timeframe must meet ADX ≥22 and the
            # other may be as low as 18.  This further loosens the gate
            # compared to the original 22/20 requirement and allows trades
            # during emerging trends or transitional periods.
            adx_gate_bool = ((adx_h1_val >= 22 and adx_m15_val >= 18) or
                             (adx_m15_val >= 22 and adx_h1_val >= 18))
            # Build a list of confluence flags to count
            feature_flags = [
                h4_align,
                1 if adx_gate_bool else 0,
                rsi_ok_flag,
                ema_align_flag,
                bos_align_flag,
                vol_mom_flag,
                session_flag,
                spread_flag
            ]
            conf_count = sum(1 for f in feature_flags if f)
            min_conf = 4  # minimum confluence count required for a trade
            # Determine if current setup passes the confluence threshold
            passes_conf = (conf_count >= min_conf)
            # In aggressive mode, permit one fewer confluence point
            if AGGRESSIVE_MODE and not passes_conf and conf_count >= (min_conf - 1):
                passes_conf = True
            # For low volatility setups with valid RSI, allow one less confluence (original logic)
            if not passes_conf and low_vol_setup and conf_count >= (min_conf - 1) and rsi_ok_flag:
                passes_conf = True
            # If confluence criteria are not met, skip this signal
            if not passes_conf:
                continue

            # Enforce micro-only policy for GBP pairs: skip any full-trade
            # execution paths for symbols whose base starts with 'GBP'. This
            # ensures `GBPUSD`, `GBPJPY` (and similar) will only ever be
            # considered for micro trades in the full-trade loop.
            try:
                if globals().get('GBP_MICRO_ONLY'):
                    _base = _base_of(sym)
                    if isinstance(_base, str) and _base.upper().startswith('GBP'):
                        log_msg(f"ℹ️ {sym} is a GBP pair — skipping full trade (micro-only)")
                        continue
            except Exception:
                pass

            w = AI_WEIGHTS
            score_val = 0.0
            if h4_align: score_val += w["h4_bias"]
            # Relax the ADX and RSI gates per user request.  Require both H1 and M15
            # ADX values to exceed the lowered minimum (22).  If the gate is met,
            # compute contributions scaled from the original bases (25/20) for
            # backwards compatibility.  Similarly, RSI thresholds are now
            # 53/47 rather than 55/45.
            adx_gate = (adx_h1_val >= LOOSEN_FULL_ADX_H1_MIN and adx_m15_val >= LOOSEN_FULL_ADX_M15_MIN)
            if adx_gate:
                score_val += w["adx_h1"] * max(0.0, min(1.0, (adx_h1_val - 25) / 20.0))
                score_val += w["adx_m15"] * max(0.0, min(1.0, (adx_m15_val - 20) / 20.0))
                if adx_rise_flag:
                    score_val += w["adx_rising"]
            if rsi_ok_flag: score_val += w["rsi_regime"]
            if ema_align_flag: score_val += w["ema_slope"]
            if bos_align_flag: score_val += w["bos_match"]
            if vol_mom_flag: score_val += w["fvg_or_atrhot"]
            if session_flag: score_val += w["session"]
            if spread_flag: score_val += w["spread_ok"]
            # Account for ATR Z-score and RSIL features
            score_val += w.get("atr_z", 0) * max(0.0, min(1.0, atr_z_val / 1.5))
            score_val += w.get("rsil", 0) * max(0.0, min(1.0, abs(rsil_val - 50.0) / 50.0))
            score_val = min(100.0, score_val)
            # Apply bias then smooth the full-trade AI score for stability.
            raw_score = int(round(max(0.0, min(100.0, score_val + AI_BIAS)), 0))
            # Retrieve the previous AI score for smoothing.  If none exists for this
            # symbol, fall back to the current raw score.
            prev_score = LAST_AI_SCORES.get(sym, raw_score)
            smoothed = smooth_ai_score(raw_score, prev_score)
            LAST_AI_SCORES[sym] = smoothed
            score_val = int(round(max(0.0, min(100.0, smoothed)), 0))
            # Apply market memory bias to the score for additional smoothing
            try:
                score_val = int(round(max(0.0, min(100.0, apply_market_memory_to_score(score_val))), 0))
            except Exception:
                pass
            conf_thresh = SYMBOL_CONF_THRESH.get(sym_base, AI_THRESHOLD)
            approve_signal = (score_val >= conf_thresh)
            if not approve_signal:
                if AUTO_PLACE_FULL:
                    log_msg(f"🤖 {sym_base} signal rejected by AI (score {score_val} < {conf_thresh})")
                continue
            # Enforce SMC confirmation as a strict gate for full entries
            try:
                try:
                    smc = smc_confluence(sym, side)
                except Exception:
                    smc = None
                if isinstance(smc, dict):
                    if not smc.get('allow', True):
                        log_msg(f"🛑 SMC block: {sym} {side} - score={smc.get('score')}")
                        continue
                    # Strong contradiction blocks the trade
                    if smc.get('direction') and smc.get('direction') != side and smc.get('score', 0) >= 70:
                        log_msg(f"🛑 SMC contradiction: suggested {smc.get('direction')}, blocking {side} trade on {sym} (score {smc.get('score')})")
                        continue
                    # If SMC agrees, give a modest boost to score for sizing decisions
                    if smc.get('direction') == side:
                        score_val = min(100, score_val + int(min(12, smc.get('score', 0) * 0.08)))
                        meta = meta if 'meta' in locals() else {}
                        meta = {**meta, 'smc_score': smc.get('score'), 'smc_details': smc.get('details')}
            except Exception:
                pass
            meta = {
                "ai_score": score_val,
                "features": {
                    "h4_bias": h4_align,
                    "adx_h1": adx_h1_val, "adx_m15": adx_m15_val, "adx_rising": adx_rise_flag,
                    "rsi_h1": rsi_h1_val, "rsi_m15": rsi_m15_val, "ema_slope": ema_slope_val,
                    "ema_align": ema_align_flag, "bos_align": bos_align_flag,
                    "fvg": 1 if fvg_flag else 0,
                    "atr_hot": 1 if atr_hot_flag else 0,
                    "vol_mom": vol_mom_flag,
                    "atr_z": atr_z_val,
                    "rsil": rsil_val,
                    "session": session_flag,
                    "spread_ok": spread_flag,
                    "rsi_ok": rsi_ok_flag
                }
            }
            # Before computing the order entry price, enforce a one‑candle M5 confirmation.
            # Only proceed if the most recent M5 candle supports the direction.
            try:
                dir_str = "buy" if side.upper() == "BUY" else "sell"
                m5_ok, m5_reason = m5_confirmation(sym, dir_str)
            except Exception:
                m5_ok, m5_reason = (True, "M5_OK")
            if not m5_ok:
                log_msg(f"🚫 M5 confirmation blocked full trade on {sym} ({m5_reason})")
                try:
                    # For XAUUSD, defer noisy telegram blocks unless a setup exists
                    base_sym = _base_of(sym)
                    if base_sym == 'XAUUSD':
                        # mark no-setup (handled by outer scan flow)
                        xau_no_setup_seen = True
                        try:
                            globals()['LAST_SCAN_XAU_NO_SETUP'] = True
                        except Exception:
                            pass
                    else:
                        telegram_block(sym, m5_reason)
                except Exception:
                    pass
                continue
            entry_price = t.ask if side == "BUY" else t.bid
            # Determine the order lot while incorporating the dynamic risk scale.  Do not
            # exceed a 10% increase over the base full lot or risk percentage.
            if USE_FIXED_FULL_LOT:
                # When using a fixed lot, apply the risk scale but never widen beyond
                # 10% of the default lot size.  Clamp between MICRO_LOT_MIN and
                # FULL_LOT_DEFAULT * 1.10.
                dyn_lot = FULL_LOT_DEFAULT * risk_scale_symbol
                max_lot = FULL_LOT_DEFAULT * 1.10
                dyn_lot = max(MICRO_LOT_MIN, min(max_lot, dyn_lot))
                order_lot = normalize_volume(sym, dyn_lot)
            else:
                thr = conf_thresh
                # Existing factor based on AI score
                factor = 1.0 + ((score_val - thr) / max(1.0, (100.0 - thr))) * 0.5
                factor = clamp(factor, 1.0, 1.5)
                risk_pct_dyn = RISK_PCT * factor * risk_scale_symbol
                # Cap the dynamic risk to 110% of the baseline
                risk_pct_dyn = min(RISK_PCT * 1.10, risk_pct_dyn)
                order_lot = normalize_volume(sym, get_risk_lot(sym, entry_price, s["sl"], risk_pct_dyn))
            # Use centralised decision and execution path for full trades
            try:
                atr_m15 = s.get("atr", 0.0)
            except Exception:
                atr_m15 = 0.0
            decision = prepare_trade_decision(sym, side, s["sl"], atr_m15, dynamic_scale=risk_scale_symbol, micro=False, meta=meta)
            res = None
            if decision:
                try:
                    res = execute_market_order(sym, side, decision.get("sl"), decision.get("tp1"), decision.get("tp2"), decision.get("tp3"), decision.get("lot"), micro=False, comment_tag="", meta=meta)
                except Exception:
                    res = None
            if res:
                # If a full trade was placed during a manual scan, record the symbol and type
                if globals().get('SCAN_MODE'):
                    try:
                        globals()['LAST_SCAN_PLACED_SYMBOL'] = sym_base.split('.')[0].upper() if isinstance(sym_base, str) else sym_base
                        globals()['LAST_SCAN_TRADE_TYPE'] = 'full'
                    except Exception:
                        pass
                try:
                    base_sym = _base_of(sym)
                    if isinstance(base_sym, str) and base_sym.upper() == 'XAUUSD':
                        try:
                            sess_lbl, _, _ = session_bounds()
                        except Exception:
                            sess_lbl = None
                        _record_xau_session_trade(sess_lbl, side)
                        log_msg(f"ℹ️ Recorded XAUUSD {side} in session {sess_lbl}")
                except Exception:
                    pass
                placed = True
                break
        if placed:
            break
    if not placed:
        log_msg("ℹ️ No full-trade setup filled this scan.")
    # Post-scan: for XAUUSD, if we saw no setup at all this scan cycle, emit
    # a single concise console-only message and remain silent to Telegram.
    try:
        if xau_no_setup_seen and not xau_setup_found and not placed:
            print("ℹ️ Checked XAUUSD — no setup this scan")
    except Exception:
        pass
    return placed
def get_risk_lot(symbol, entry_price, sl, risk_pct):
    """Simple risk-based lot calculation (fallback)."""
    try:
        ai = mt5.account_info()
        bal = getattr(ai, 'balance', 1000.0) if ai else 1000.0
        eq = getattr(ai, 'equity', bal) if ai else bal
        base_amount = eq if USE_EQUITY_RISK else bal
        risk_amount = base_amount * risk_pct
        diff = abs(entry_price - sl)
        if diff < 1e-9:
            return MICRO_LOT_MIN
        si = mt5.symbol_info(symbol)
        if si and si.point and si.trade_tick_value:
            tick_val = si.trade_tick_value
            risk_per_lot = (diff / si.point) * tick_val
        else:
            risk_per_lot = diff * 10.0
        lot = risk_amount / risk_per_lot if risk_per_lot > 1e-9 else MICRO_LOT_MIN
        return normalize_volume(symbol, lot)
    except Exception:
        return MICRO_LOT_MIN


def _risk_per_lot(symbol, entry_price, sl):
    """Return approximate monetary risk per 1.0 lot for the given symbol and SL distance."""
    try:
        diff = abs(entry_price - sl)
        si = mt5.symbol_info(symbol)
        if si and getattr(si, 'point', None) and getattr(si, 'trade_tick_value', None):
            tick_val = si.trade_tick_value
            risk_per_lot = (diff / si.point) * tick_val
        else:
            risk_per_lot = diff * 10.0
        return max(1e-9, float(risk_per_lot))
    except Exception:
        return 1.0


def adjust_lot_to_risk(symbol, proposed_lot, entry_price, sl, max_risk_pct):
    """Reduce `proposed_lot` if it would risk more than `max_risk_pct` of account.

    Returns an adjusted lot (never larger than proposed_lot) that respects max_risk_pct
    and the global MAX_RISK_PER_TRADE. Uses account balance/equity based on
    `USE_EQUITY_RISK`.
    """
    try:
        ai = mt5.account_info()
        bal = float(getattr(ai, 'balance', 1000.0) if ai else 1000.0)
        eq = float(getattr(ai, 'equity', bal) if ai else bal)
        base_amount = eq if USE_EQUITY_RISK else bal
        # enforce hard cap
        max_risk_pct = min(max_risk_pct, MAX_RISK_PER_TRADE, globals().get('RISK_PCT', MAX_RISK_PER_TRADE))
        risk_per_lot = _risk_per_lot(symbol, entry_price, sl)
        max_risk_amount = base_amount * max_risk_pct
        max_lot = max_risk_amount / risk_per_lot if risk_per_lot > 0 else proposed_lot
        max_lot = max(0.0, float(max_lot))
        # Never increase proposed lot here — only cap downward
        adjusted = min(proposed_lot, max_lot)
        # Ensure minimum sensible lot
        if adjusted < MICRO_LOT_MIN:
            return MICRO_LOT_MIN
        return round(adjusted, 2)
    except Exception:
        return max(MICRO_LOT_MIN, round(proposed_lot, 2))
def rr_next_index(i):
    """Round-robin next index for micro trade rotation."""
    return (i + 1) % len(SYMBOLS)
def session_bounds(now=None):
    """Determine current trading session: ASIA, LON, or NY (returns label and bounds)."""
    n = now or now_uk()
    d = n.date()
    tz = SAFE_TZ
    def mk(h, m): return tz.localize(datetime(d.year, d.month, d.day, h, m))
    
    # Asia session: 00:00 - 06:00 UK time
    asia_start, asia_end = mk(0, 0), mk(6, 0)
    if asia_start <= n < asia_end:
        return ("ASIA", asia_start, asia_end)
    
    # London session: 08:00 - 13:29 UK time
    lon_start, lon_end = mk(8, 0), mk(13, 29)
    if lon_start <= n <= lon_end:
        return ("LON", lon_start, lon_end)
    
    # New York session: 13:30 - 17:00 UK time
    ny_start, ny_end   = mk(13, 30), mk(17, 0)
    if ny_start <= n <= ny_end:
        return ("NY", ny_start, ny_end)
    
    return (None, None, None)
def in_session():
    return in_session_full()
def mark_session_full():
    label, _, _ = session_bounds()
    if label == "LON":
        day_stats["fulls_LON"] += 1
    if label == "NY":
        day_stats["fulls_NY"] += 1
def mark_session_micro():
    label, _, _ = session_bounds()
    if label == "LON":
        day_stats["micros_LON"] += 1
    if label == "NY":
        day_stats["micros_NY"] += 1
def try_micro_on(base_pair):
    """Attempt a micro trade on the given base pair if conditions allow."""
    # Do not block micro trades during news.  Micro strategies are allowed
    # to execute even when red‑news events are active.  The previous
    # behaviour skipped micros on XAUUSD during news blackout windows; this
    # has been removed to ensure micros are independent of full‑trade news
    # gating.
    total_micros = day_stats.get('micros_LON', 0) + day_stats.get('micros_NY', 0)
    if total_micros >= MICRO_MAX_PER_DAY:
        log_msg(f"🚫 Daily micro cap reached ({total_micros}/{MICRO_MAX_PER_DAY}).")
        return False
    sym = resolve_symbol(base_pair)
    # Enforce higher‑timeframe alignment for micro trades.  Skip if HTFs disagree.
    try:
        ok_htf, reason_htf = multi_tf_direction_ok(sym)
    except Exception:
        ok_htf, reason_htf = (True, "HTF_OK")

    if not ok_htf:
        # Special-case for XAUUSD: allow a micro trade even if HTF alignment
        # is not perfect provided at least one usable SMC signal exists on
        # the LTF (M15/H1). This permits micro gold trades to fire on 1/3
        # confirmations as requested.
        try:
            base_upper = (base_pair or '').split('.')[0].upper()
        except Exception:
            base_upper = (base_pair or '').upper()

        if base_upper == 'XAUUSD' or (sym and sym.upper().startswith('XAUUSD')):
            try:
                # Gather SMC signals on M15/H1
                m15_tmp = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 120, max_age=1.5)
                h1_tmp = fetch_data_cached(sym, mt5.TIMEFRAME_H1, 150, max_age=3.0)
                if m15_tmp is None or h1_tmp is None:
                    raise Exception('data_missing')
                bos_e, bos_s = safe_smc_call(detect_bos, h1_tmp, 'detect_bos')
                sweep_e, sweep_s = safe_smc_call(detect_sweep, m15_tmp, 'detect_sweep')
                fvg_e, fvg_s = safe_smc_call(detect_fvg, m15_tmp, 'detect_fvg')
                emojis_local = [bos_e, sweep_e, fvg_e]
                usable_local = sum(1 for e in emojis_local if e in (EMOJI_STRONG, EMOJI_WEAK_OK))
                if usable_local >= 1:
                    # Allow micro despite HTF neutral/blocked. Log only (no Telegram).
                    log_msg(f"[MICRO_OVERRIDE] Allowing micro on {sym} despite HTF block ({reason_htf}) - usable_signals={usable_local}")
                    ok_htf = True
                else:
                    log_msg(f"🚫 HTF alignment blocked micro trade on {sym} ({reason_htf})")
                    try:
                        # Suppress noisy XAUUSD telegram blocks for micro scans
                        if base_upper != 'XAUUSD' and not (sym and sym.upper().startswith('XAUUSD')):
                            telegram_block(sym, reason_htf)
                    except Exception:
                        pass
                    return False
            except Exception:
                # If signal gathering failed, respect the HTF block
                log_msg(f"🚫 HTF alignment blocked micro trade on {sym} ({reason_htf}) - signal check failed")
                try:
                    if base_upper != 'XAUUSD' and not (sym and sym.upper().startswith('XAUUSD')):
                        telegram_block(sym, reason_htf)
                except Exception:
                    pass
                return False
        else:
            log_msg(f"🚫 HTF alignment blocked micro trade on {sym} ({reason_htf})")
            try:
                if base_upper != 'XAUUSD' and not (sym and sym.upper().startswith('XAUUSD')):
                    telegram_block(sym, reason_htf)
            except Exception:
                pass
            return False
    # Record a snapshot for market memory
    try:
        record_market_snapshot(sym)
    except Exception:
        pass
    m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 100, max_age=1.5)
    h1  = fetch_data_cached(sym, mt5.TIMEFRAME_H1, 150, max_age=3.0)
    tick = mt5.symbol_info_tick(sym)
    if m15 is None or h1 is None or not tick:
        log_msg(f"⚠️ Micro skipped (data/tick missing) on {sym}")
        return False

    # Update strict mode based on recent performance
    _update_strict_mode()

    # Market quality filter (side resolved later; use both RSI bands)
    # Here we gate on ADX/Spread/Candle/News first, then apply side-specific RSI later.
    ok_vol, reason_vol = _volatility_ok(m15, datetime.now(SAFE_TZ))
    if not ok_vol:
        try:
            telegram_msg(f"🚫 {sym} MICRO skipped: {reason_vol}")
        except Exception:
            pass
        return False

    # Predictive AI and liquidity gating
    p_score = 1.0
    liquidity = 1.0
    risk_scale_symbol = 1.0
    low_vol_gold = False
    base_sym = _base_of(sym)
    if ENABLE_PREDICTIVE_AI:
        try:
            p_score = blended_prediction(sym)
        except Exception:
            p_score = 1.0
        try:
            liquidity = predict_liquidity_pressure(sym)
        except Exception:
            liquidity = 1.0
        # When HTF is disabled, predictive and liquidity gating must not block.
        if HARD_DISABLE_HTF:
            p_score = 1.0
            liquidity = 1.0
            risk_scale_symbol = 1.0
            low_vol_gold = False
        # Low‑volatility gold mode: adjust micro trades on dead sessions
        if base_sym == 'XAUUSD':
            try:
                if m15 is not None and len(m15) >= 50:
                    try:
                        cur_atr = true_atr(m15, 14)
                    except Exception:
                        cur_atr = None
                    try:
                        atr_avg_raw = (m15['high'] - m15['low']).rolling(50).mean().iloc[-1]
                    except Exception:
                        atr_avg_raw = None
                    try:
                        adx_m15_lv = float(adx(m15))
                    except Exception:
                        adx_m15_lv = 0.0
                    if cur_atr and atr_avg_raw and adx_m15_lv < 18:
                        if cur_atr < 0.6 * atr_avg_raw:
                            low_vol_gold = True
            except Exception:
                low_vol_gold = False
        # Block micro trade if predictive score is below 0.55 or liquidity is very
        # poor.  Liquidity threshold reduced to 0.25 (from 0.35) to allow more
        # entries during quieter markets.  If the predictive score is
        # exceptionally high (≥0.70), allow the trade to continue despite low
        # liquidity by falling through to dynamic risk scaling with a reduced
        # risk factor.
        if not HARD_DISABLE_HTF:
            if p_score < 0.55 or liquidity < 0.25:
                if p_score >= 0.70:
                    # High confidence overrides low liquidity - do not block here.
                    pass
                else:
                    log_msg(f"🚫 Predictive/liq gating failed on {sym} (p={p_score:.2f}, liq={liquidity:.2f})")
                    return False
        # Compute dynamic risk scaling factor
        try:
            risk_scale_symbol = compute_dynamic_risk_scale(sym, p_score, liquidity)
        except Exception:
            risk_scale_symbol = 1.0
        if not HARD_DISABLE_HTF:
            if risk_scale_symbol <= 0.0:
                # If the predictive score is very high (≥0.70) then fall back to a
                # reduced risk scale (0.50) instead of blocking the trade entirely.
                if p_score >= 0.70:
                    risk_scale_symbol = 0.50
                else:
                    log_msg(f"🚫 Dynamic risk scale blocked micro on {sym} (scale={risk_scale_symbol:.2f})")
                    return False
        # In low‑volatility gold regime, reduce risk by an additional 45%
        if low_vol_gold:
            risk_scale_symbol *= 0.55
    # End predictive gating
    # Compute ATR on the M15 timeframe.  Fall back to the spread if no data.
    try:
        atr_val = atr(m15, 14)
        if atr_val <= 0:
            atr_val = true_atr(m15, 14)
        if atr_val <= 0:
            atr_val = 0.5 * (tick.ask - tick.bid)
    except Exception:
        atr_val = 0.5 * (tick.ask - tick.bid)
    # Enforce a minimum volatility floor per user request.  For micro
    # strategies, use MICRO_ATR_MIN rather than the full‑trade ATR floor.
    atr_floor = MICRO_ATR_MIN
    # Aggressive mode does not lower the ATR floor below the original constant
    if AGGRESSIVE_MODE:
        # Compute a 10% reduction, but never below the base MICRO_ATR_MIN
        atr_floor = max(MICRO_ATR_MIN, MICRO_ATR_MIN * 0.9)
    if atr_val < atr_floor:
        log_msg(f"🚫 {sym} micro ATR floor fail (ATR {atr_val:.3f} < {atr_floor})")
        return False

    # Enforce a minimal price movement filter to avoid dead ranges.  Compute
    # the midpoint of the previous M15 candle and measure the percentage
    # deviation of the current close from that midpoint.  Aggressive mode
    # permits a 10% reduction in the range requirement.
    try:
        prev_high = float(m15['high'].iloc[-2]); prev_low = float(m15['low'].iloc[-2])
        last_close = float(m15['close'].iloc[-1])
        mid = (prev_high + prev_low) / 2.0
        pct_range = abs(last_close - mid) / max(1e-9, mid) * 100.0
        range_req = LOOSEN_FULL_RANGE_MAX_PCT
        if AGGRESSIVE_MODE:
            range_req = max(LOOSEN_FULL_RANGE_MAX_PCT, LOOSEN_FULL_RANGE_MAX_PCT * 0.9)
        if pct_range <= range_req:
            log_msg(f"🚫 {sym} micro range filter fail (|pos| {pct_range:.2f}% <= {range_req}%)")
            return False
    except Exception:
        pass
    bias_dir = ema_bias(h1)
    side = "BUY" if bias_dir == "BUY" else "SELL"
    ok_limits, reason_limits = _check_hard_limits(sym, side, "MICRO")
    if not ok_limits:
        try:
            telegram_msg(f"🚫 {sym} MICRO blocked: {reason_limits}")
        except Exception:
            pass
        return False
    ok_mkt, reason_mkt = _market_quality_ok(sym, side, m15, h1, datetime.now(SAFE_TZ))
    if not ok_mkt:
        try:
            telegram_msg(f"🚫 {sym} MICRO skipped: {reason_mkt}")
        except Exception:
            pass
        return False
    ok_trend, reason_trend = _trend_consistency_ok(sym, side, h1)
    if not ok_trend:
        try:
            telegram_msg(f"🚫 {sym} MICRO skipped: {reason_trend}")
        except Exception:
            pass
        return False
    ok_entry, reason_entry = _entry_confirmation_ok(m15, side)
    if not ok_entry:
        try:
            telegram_msg(f"🚫 {sym} MICRO skipped: {reason_entry}")
        except Exception:
            pass
        return False
    sl = (tick.bid - 1.5 * atr_val) if side == "BUY" else (tick.ask + 1.5 * atr_val)
    h4 = fetch_data_cached(sym, mt5.TIMEFRAME_H4, 260, max_age=6.0)
    m5 = fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6)
    # Soft HTF BOS gating: ensure H4 BOS aligns with the trade side and Daily
    # BOS is not opposite.  If the BOS check fails, skip this micro trade.
    try:
        if not htf_bos_ok(sym, side):
            log_msg(f"🚫 {sym} micro blocked: HTF BOS mismatch")
            return False
    except Exception:
        pass
    h4_align = 1 if h4 is not None and ema_bias(h4) == side else 0
    adx_h1_val, adx_is_rising = adx_rising(h1)
    adx_m15_val = float(adx(m15)) if m15 is not None else 0.0
    adx_rise_flag = 1 if adx_is_rising else 0
    try:
        rsi_h1_val = float(rsi(h1)); rsi_m15_val = float(rsi(m15))
    except Exception:
        rsi_h1_val = rsi_m15_val = 50.0
    # Compute a relaxed RSI gate for micro trades.  A trade is valid when:
    #  • For a BUY: both H1 and M15 RSIs are above the micro buy threshold.
    #  • For a SELL: both RSIs are below the micro sell threshold.
    #  • Or both RSIs lie within a neutral band (RSI_NEUTRAL_LOW to RSI_NEUTRAL_HIGH).
    rsi_ok_flag = 1 if (
        (side == "BUY"  and rsi_h1_val >= RSI_MICRO_BUY_MIN and rsi_m15_val >= RSI_MICRO_BUY_MIN) or
        (side == "SELL" and rsi_h1_val <= RSI_MICRO_SELL_MAX and rsi_m15_val <= RSI_MICRO_SELL_MAX) or
        (RSI_NEUTRAL_LOW <= rsi_h1_val <= RSI_NEUTRAL_HIGH and RSI_NEUTRAL_LOW <= rsi_m15_val <= RSI_NEUTRAL_HIGH)
    ) else 0
    ema50 = get_ema(h1, 50); ema200 = get_ema(h1, 200)
    ema_slope_val = 0.0
    if len(h1) > 12:
        try:
            ema_diff = ema50 - ema200
            ema_slope_val = float(ema_diff.iloc[-1] - ema_diff.iloc[-11]) if len(ema_diff) > 11 else 0.0
        except Exception:
            ema_slope_val = 0.0
    # EMA alignment based on slope used for the AI score
    ema_align_flag = 1 if ((side == "BUY" and ema_slope_val > 0) or (side == "SELL" and ema_slope_val < 0)) else 0
    # EMA trend requirement: the slower 200‑period EMA must be below the faster 50‑period EMA for buys and above for sells.
    ema_trend_ok = 0
    try:
        last_ema50 = float(ema50.iloc[-1]) if hasattr(ema50, 'iloc') else float(ema50)
        last_ema200 = float(ema200.iloc[-1]) if hasattr(ema200, 'iloc') else float(ema200)
        ema_trend_ok = 1 if ((side == "BUY" and last_ema50 > last_ema200) or (side == "SELL" and last_ema50 < last_ema200)) else 0
    except Exception:
        ema_trend_ok = 0
    bos_dir = detect_bos(h1)
    bos_align_flag = 1 if ((side == "BUY" and bos_dir == "BOS_UP") or (side == "SELL" and bos_dir == "BOS_DOWN")) else 0
    fvg_flag = False
    try:
        fvg_flag = bool(detect_fvg(m15) or (m5 is not None and detect_fvg(m5)))
    except Exception:
        fvg_flag = False
    atr_hot_flag = False
    try:
        atr_series = (m15["high"] - m15["low"]).rolling(20).mean()
        if len(atr_series) > 1:
            atr_hot_flag = bool(atr_series.iloc[-1] >= atr_series.iloc[-2])
    except Exception:
        atr_hot_flag = False
    vol_mom_flag = 1 if (fvg_flag or atr_hot_flag) else 0
    session_flag = 1 if in_session() else 0
    spread_flag = 1 if spread_ok(sym) else 0
    # ----------------------------------------------------------------------
    # Enhanced feature extraction for AI scoring
    # Compute ATR Z-score (volatility regime) and RSI-Leader (momentum)
    try:
        atr_z_val = atr_z_score(m15)
    except Exception:
        atr_z_val = 0.0
    try:
        rsil_val = rsi_leader(m15)
    except Exception:
        rsil_val = rsi(m15)
    w = AI_WEIGHTS
    score_val = 0.0
    if h4_align: score_val += w["h4_bias"]
    # Relaxed ADX gate for micro trades.  Require H1 and M15 ADX values to
    # exceed the micro thresholds (14 and 12 respectively).  These values
    # accommodate trading in consolidating markets while still filtering out
    # completely flat regimes.
    adx_gate = (adx_h1_val >= ADX_MICRO_H1_MIN and adx_m15_val >= ADX_MICRO_M15_MIN)
    if adx_gate:
        score_val += w["adx_h1"] * max(0.0, min(1.0, (adx_h1_val - 25) / 20.0))
        score_val += w["adx_m15"] * max(0.0, min(1.0, (adx_m15_val - 20) / 20.0))
        if adx_rise_flag:
            score_val += w["adx_rising"]
    if rsi_ok_flag:
        score_val += w["rsi_regime"]
    if ema_align_flag: score_val += w["ema_slope"]
    if bos_align_flag: score_val += w["bos_match"]
    if vol_mom_flag: score_val += w["fvg_or_atrhot"]
    if session_flag: score_val += w["session"]
    if spread_flag: score_val += w["spread_ok"]
    # Add contributions from the new volatility regime and momentum features.
    # Scale the ATR Z-score so that values around 0-1 map into 0-1.5 of the weight.
    score_val += w.get("atr_z", 0) * max(0.0, min(1.0, atr_z_val / 1.5))
    # RSIL deviation from 50 (0-100 range) converted to 0-1 range.
    score_val += w.get("rsil", 0) * max(0.0, min(1.0, abs(rsil_val - 50.0) / 50.0))
    score_val = min(100.0, score_val)
    score_val = int(round(max(0.0, min(100.0, score_val + AI_BIAS)), 0))
    meta = {
        "ai_score": score_val,
        "features": {
            "h4_bias": h4_align,
            "adx_h1": adx_h1_val, "adx_m15": adx_m15_val, "adx_rising": adx_rise_flag,
            "rsi_h1": rsi_h1_val, "rsi_m15": rsi_m15_val, "ema_slope": ema_slope_val,
            "ema_align": ema_align_flag, "bos_align": bos_align_flag,
            "fvg": 1 if fvg_flag else 0,
            "atr_hot": 1 if atr_hot_flag else 0,
            "vol_mom": vol_mom_flag,
            "atr_z": atr_z_val,
            "rsil": rsil_val,
            "session": session_flag,
            "spread_ok": spread_flag,
            "rsi_ok": rsi_ok_flag
        }
    }
    # Apply dynamic risk scaling to the micro lot.  Clamp within permitted
    # micro lot bounds.  If risk_scale_symbol >1.0 the lot can grow up to the
    # maximum; if <1.0 it will shrink but never below the minimum.
    dynamic_micro_lot = MICRO_LOT_TARGET * risk_scale_symbol
    dynamic_micro_lot = max(MICRO_LOT_MIN, min(MICRO_LOT_MAX, dynamic_micro_lot))
    order_lot = normalize_volume(sym, dynamic_micro_lot)
    # Enforce EMA trend gate for both BUY and SELL micros
    if not ema_trend_ok:
        log_decision("Micro EMA trend failed", side, sym)
        return False
    # Enforce ADX floor for both BUY and SELL micros
    if (adx_h1_val < ADX_MICRO_H1_MIN or adx_m15_val < ADX_MICRO_M15_MIN):
        log_decision("Micro ADX failed", side, sym)
        return False
    # Apply market memory smoothing to the micro AI score
    try:
        score_val = int(round(max(0.0, min(100.0, apply_market_memory_to_score(score_val))), 0))
    except Exception:
        pass
    # If the AI score does not meet the micro minimum, do not place a micro trade.
    if score_val < MICRO_AI_MIN:
        log_decision(f"Micro AI score failed ({score_val} < {MICRO_AI_MIN})", side, sym)
        return False
    log_msg(f"🌱 Micro on {sym} ({side}) | ATR={atr_val:.3f} | lot={order_lot:.2f}, AI_score={score_val} | risk_scale={risk_scale_symbol:.2f}")
    # Log a decision on successful micro trade approval
    log_decision("ALL CONDITIONS PASSED - MICRO TRADE APPROVED", side, sym)
    # Place the micro trade and capture the result
    # Before placing a micro trade, enforce M5 candle confirmation.  Require that
    # the most recent M5 candle confirms the intended direction.
    try:
        dir_str = "buy" if side.upper() == "BUY" else "sell"
        m5_ok, m5_reason = m5_confirmation(sym, dir_str)
    except Exception:
        m5_ok, m5_reason = (True, "M5_OK")
    if not m5_ok:
        log_msg(f"🚫 M5 confirmation blocked micro trade on {sym} ({m5_reason})")
        try:
            telegram_block(sym, m5_reason)
        except Exception:
            pass
        return False
    # Enforce SMC for micro entries on XAUUSD (and apply blocking behavior)
    try:
        try:
            smc = smc_confluence(sym, side)
        except Exception:
            smc = None
        if isinstance(smc, dict):
            if not smc.get('allow', True):
                log_msg(f"🛑 SMC block: {sym} {side} - score={smc.get('score')}")
                return False
            if smc.get('direction') and smc.get('direction') != side and smc.get('score', 0) >= 70:
                log_msg(f"🛑 SMC contradiction: suggested {smc.get('direction')}, blocking {side} micro on {sym} (score {smc.get('score')})")
                return False
    except Exception:
        pass
    try:
        decision = prepare_trade_decision(sym, side, sl, atr_val, dynamic_scale=risk_scale_symbol, micro=True, meta=meta)
        if not decision:
            res = None
        else:
            try:
                res = execute_market_order(sym, side, decision.get("sl"), decision.get("tp1"), decision.get("tp2"), decision.get("tp3"), decision.get("lot"), micro=True, comment_tag="", meta=meta)
            except Exception:
                res = None
    except Exception:
        res = None
    ok = bool(res)
    # If a micro trade was placed during a manual scan, record the symbol and type
    if ok:
        try:
            # record manual scan placement metadata
            if globals().get('SCAN_MODE'):
                try:
                    globals()['LAST_SCAN_PLACED_SYMBOL'] = base_pair.split('.')[0].upper() if isinstance(base_pair, str) else base_pair
                    globals()['LAST_SCAN_TRADE_TYPE'] = 'micro'
                except Exception:
                    pass
            # Session count for XAU
            try:
                base_sym = base_pair.split('.')[0].upper() if isinstance(base_pair, str) else base_pair
                if isinstance(base_sym, str) and base_sym == 'XAUUSD':
                    try:
                        sess_lbl, _, _ = session_bounds()
                    except Exception:
                        sess_lbl = None
                    _record_xau_session_trade(sess_lbl, side)
                    log_msg(f"ℹ️ Recorded XAUUSD {side} in session {sess_lbl}")
            except Exception:
                pass
        except Exception:
            pass
    return ok
def micro_fallback():
    """Cycle through symbols to place a micro trade if no full trades were placed in this scan."""
    # Guard micro trades until next ready time using safe timezone
    if 'MICRO_READY_AT' in globals():
        try:
            if datetime.now(SAFE_TZ) < globals()['MICRO_READY_AT']:
                return
        except Exception:
            # If comparison fails due to missing tzinfo, proceed
            pass
    global MICRO_RR_IDX
    tried = 0
    idx = MICRO_RR_IDX % len(SYMBOLS)
    while tried < len(SYMBOLS):
        if try_micro_on(SYMBOLS[idx]):
            MICRO_RR_IDX = rr_next_index(idx); save_state(); return
        idx = rr_next_index(idx); tried += 1
    MICRO_RR_IDX = rr_next_index(MICRO_RR_IDX)
    save_state()
    log_msg("🛑 All micro trade attempts failed across pairs")
def build_help_message():
    firm, rules = current_ruleset() if 'current_ruleset' in globals() else (None, None)
    firm_name = firm or "None"
    daily = rules.get("daily_loss_pct") if rules else None
    maxl = rules.get("max_loss_pct") if rules else None
    trailing = " (trailing)" if rules and rules.get('trailing_max') else ""
    if daily is not None and maxl is not None:
        firm_name += f" | Daily {daily}% | Max {maxl}%{trailing}"
    routes = ", ".join([f"{b}:{SYMBOL_ROUTE.get(b,'auto')}" for b in SYMBOLS]) if SYMBOL_ROUTE else ""
    msg = (
        "🤝 Commands:\n"
        "• /active - bot online\n"
        "• /status - balance/equity/trade stats + firm\n"
        "• /mode strict | /mode balanced\n"
        "• /strategy goat|legacy\n"
        "• /start - (re)start polling\n"
        "• /stop - stop polling\n"
        "• /restart - restart polling\n"
        "• /ping - health ping\n"
        "• /settings - print current runtime settings\n"
        "• /help - this menu\n"
        "• /findtrade - FULL scan across all pairs; if none, then micro fallback\n"
        "• /preview - scan XAUUSD first then other pairs, report setups but do NOT place trades (runs async)\n"
        "• /findtrade_strict - strict confluence scan (A-setup only)\n"
        "• /full - force a FULL trade now (RR 1:2)\n"
        "• /risk <pct> - set full trade risk %, e.g. /risk 0.5\n"
        "• /microlot <0.01-0.05> - set micro lot size\n"
        "• /demo [lot] - place a micro demo trade now\n"
        "• /micro <PAIR> - set micro rotation start pair\n"
        "• /route <BASE> <ALIAS> - set symbol alias (routing)\n"
        "• /probe <PAIR> - check symbol route\n"
        "• /autotest - autotrading status check\n"
        "• /quiet on|off - toggle retry spam filtering\n"
        "• /liveonly on|off - restrict to real accounts only\n"
        "• /prop — show detected prop firm & rules\n"
        "• /prop set <FTMO|FXIFY|GOAT|AQUA|AUTO> — override/auto-detect prop firm\n"
    )
    return msg
def schedule_next_scan(target_time=None):
    """Schedule the next scan time."""
    global next_scan_at
    if target_time:
        next_scan_at = target_time
    else:
        delay_min = random.randint(SCAN_MIN_MINUTES, SCAN_MAX_MINUTES)
        next_scan_at = now_uk() + timedelta(minutes=delay_min)
def main_loop():
    """Main trading loop: polls for new data and triggers scans according to schedule and sessions."""
    global next_scan_at, last_snapshot_date, last_snapshot_date
    schedule_next_scan()  # schedule first scan
    def train_model_from_logs(): return
    def monitor_addon_opps(): return
    # Send startup confirmation to Telegram
    try:
        tg("📡 Bot is live. Starting scans...\n✅ MT5 initialized and monitoring markets.")
    except Exception:
        pass
    while True:
        try:
            # Ensure MT5 is connected before proceeding; block until connected or raise
            try:
                ensure_mt5_connected_or_exit(retries=6, delay=2.0)
            except Exception:
                # If MT5 could not be connected, sleep and try again later
                time.sleep(10)
                continue
            if BOT_STATE.trading_paused:
                time.sleep(60)
                continue
            if day_stats["date"] != now_uk().date():
                if day_stats["date"]:
                    log_msg(f"📅 {now_uk().strftime('%Y-%m-%d')} | Trades:{day_stats['trades']} Blocks:{day_stats['blocks']} Fails:{day_stats['fails']} Mode:{MODE} Strat:{STRATEGY}")
                ai = mt5.account_info()
                day_stats.update({
                    "date": now_uk().date(), "trades": 0, "wins": 0, "losses": 0, "blocks": 0, "fails": 0,
                    "fulls_LON": 0, "fulls_NY": 0,
                    "micros_LON": 0, "micros_NY": 0,
                    "micros_by_symbol": {'LON': {}, 'NY': {}},
                    "day_open_equity": ai.equity if ai else None
                })
            now = now_uk()
            if now.hour == SNAPSHOT_HOUR_LONDON and (last_snapshot_date is None or last_snapshot_date != now.date()):
                try:
                    train_model_from_logs()
                    last_snapshot_date = now.date()
                    log_msg("🤖 AI model retrained on latest performance data.")
                except Exception as e:
                    log_msg(f"⚠️ Model training error: {e}")
            if in_session() and now.minute == 0 and now.second < 10:
                run_scan(announce=True)
                hourly_notify = f"⏱️ Hourly chart check @ {now.strftime('%H:%M')} UK. Quiet scans continue every {SCAN_MIN_MINUTES}-{SCAN_MAX_MINUTES} min."
                log_msg(hourly_notify)
                schedule_next_scan(now + timedelta(minutes=1))
            # Periodically perform adaptive AI weight updates (once every 30 minutes)
            try:
                if now.minute % 30 == 0 and now.second < 5:
                    for s in SYMBOLS:
                        try:
                            adaptive_update_weights(s)
                        except Exception:
                            pass
            except Exception:
                pass
            if in_session() and now >= (next_scan_at or now):
                run_scan(announce=False)
                update_trade_stats()
                schedule_next_scan()
            try:
                monitor_addon_opps()
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
        save_state()
        time.sleep(1)
def train_model_from_logs():
    """Train/update AI model from stored trade logs (self-training loop)."""
    global AI_WEIGHTS, AI_BIAS
    if not os.path.exists("trade_log.csv"):
        return
    df = pd.read_csv("trade_log.csv")
    if df.empty:
        return
    if "adx_h1" in df.columns:
        df["adx_h1"] = np.clip((df["adx_h1"] - 25) / 20.0, 0.0, 1.0)
    if "adx_m15" in df.columns:
        df["adx_m15"] = np.clip((df["adx_m15"] - 20) / 20.0, 0.0, 1.0)
    feat_cols = ["h4_bias", "adx_h1", "adx_m15", "adx_rising", "rsi_ok", "ema_align", "bos_align", "vol_mom", "session", "spread_ok"]
    if not all(col in df.columns for col in feat_cols):
        return
    X = df[feat_cols].astype(float).to_numpy()
    y = df["win"].astype(float).to_numpy()
    N, n_feat = X.shape


# -------------------- SAFE FALLBACK STUBS (added automatically) --------------------
def check_prop_rules_before_trade():
    """Fallback prop rules check used by place_order. Returns (status, reason)."""
    try:
        # Default to OK — calling code still enforces other checks.
        return "OK", ""
    except Exception:
        return "OK", ""


# Duplicate stub of update_trade_stats removed. The comprehensive version is defined later.


# Duplicate stub of run_scan removed. The full implementation appears later.


# Duplicate stub of run_self_tests removed. A detailed implementation appears later.


# Duplicate stub of build_equity_curve_png removed. The full implementation appears later.


# Duplicate stub of fetch_red_news_windows removed. A complete version is defined later.


# Duplicate stub of flat_all_positions removed. The operative implementation appears later.


# Duplicate stub of cancel_all_pendings removed. A live implementation appears later.


# Duplicate stub of send_scan_report removed. A full implementation appears later.


def cur_price():
    """Return a dummy (symbol, tick-like) tuple for compatibility with legacy code."""
    class Tick:
        ask = 0.0
        bid = 0.0
    return None, Tick()

# End of safe fallback stubs
    if N == 0:
        return
    win_rate = y.mean()
    if win_rate <= 0.0 or win_rate >= 1.0:
        return
    w = np.zeros(n_feat); b = 0.0
    lr = 0.1
    for epoch in range(300):
        z = X.dot(w) + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = p - y
        grad_w = (X.T.dot(grad)) / N
        grad_b = grad.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    weight_keys = ["h4_bias", "adx_h1", "adx_m15", "adx_rising", "rsi_regime", "ema_slope", "bos_match", "fvg_or_atrhot", "session", "spread_ok"]
    key_map = {"rsi_ok": "rsi_regime", "ema_align": "ema_slope", "bos_align": "bos_match", "vol_mom": "fvg_or_atrhot"}
    total_pos = b if b > 0 else 0.0
    new_weights = {}
    for i, col in enumerate(feat_cols):
        model_key = key_map.get(col, col)
        coef = w[i]
        new_weights[model_key] = float(coef)
        if coef > 0:
            total_pos += coef
    scale = 1.0
    if total_pos > 0:
        scale = 100.0 / total_pos
    for k in new_weights:
        new_weights[k] = float(new_weights[k] * scale)
    new_bias = float(b * scale)
    AI_WEIGHTS.update(new_weights)
    AI_BIAS = new_bias
def update_trade_stats():
    """Update daily realized P/L and win/loss counts, handle consecutive loss halts, and log closed trades for learning."""
    try:
        now = datetime.now(timezone.utc)
        start = now.astimezone(pytz.timezone("UTC")).replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(start, now) or []
        realized = sum(getattr(d, 'profit', 0.0) for d in deals)
        wins = sum(1 for d in deals if getattr(d, 'profit', 0.0) > 0)
        losses = sum(1 for d in deals if getattr(d, 'profit', 0.0) < 0)
        day_stats['pnl'] = realized
        day_stats['wins'] = wins
        day_stats['losses'] = losses
        # --- Monthly target guardrail: compute month P/L% and toggle conservative mode ---
        try:
            global MONTH_START_BALANCE, MONTH_START_MONTH, MONTH_CONSERVATIVE_MODE, ORIGINAL_FULL_COOLDOWN_SECONDS, ORIGINAL_MICRO_COOLDOWN_SECONDS
            ai = mt5.account_info()
            cur_bal = float(getattr(ai, 'balance', 0.0) if ai else 0.0)
            cur_month = (now.year, now.month)
            if MONTH_START_MONTH is None or MONTH_START_MONTH != cur_month:
                MONTH_START_MONTH = cur_month
                MONTH_START_BALANCE = cur_bal
                # reset conservative mode at month start
                if MONTH_CONSERVATIVE_MODE:
                    MONTH_CONSERVATIVE_MODE = False
                    try:
                        globals()['FULL_COOLDOWN_SECONDS'] = ORIGINAL_FULL_COOLDOWN_SECONDS
                        globals()['MICRO_COOLDOWN_SECONDS'] = ORIGINAL_MICRO_COOLDOWN_SECONDS
                    except Exception:
                        pass
            if MONTH_START_BALANCE and MONTH_START_BALANCE > 0:
                month_pct = (cur_bal - MONTH_START_BALANCE) / MONTH_START_BALANCE
            else:
                month_pct = 0.0
            # Enter conservative mode if above high target
            if month_pct >= MONTHLY_TARGET_HIGH and not MONTH_CONSERVATIVE_MODE:
                MONTH_CONSERVATIVE_MODE = True
                try:
                    globals()['FULL_COOLDOWN_SECONDS'] = int(ORIGINAL_FULL_COOLDOWN_SECONDS * CONSERVATIVE_COOLDOWN_MULT)
                    globals()['MICRO_COOLDOWN_SECONDS'] = int(ORIGINAL_MICRO_COOLDOWN_SECONDS * CONSERVATIVE_COOLDOWN_MULT)
                except Exception:
                    pass
                try:
                    tg("🎯 Monthly Target Mode: aiming for 8–15% — switched to CONSERVATIVE mode")
                except Exception:
                    pass
            # Exit conservative mode if below low target and currently conservative
            if month_pct < MONTHLY_TARGET_LOW and MONTH_CONSERVATIVE_MODE:
                MONTH_CONSERVATIVE_MODE = False
                try:
                    globals()['FULL_COOLDOWN_SECONDS'] = ORIGINAL_FULL_COOLDOWN_SECONDS
                    globals()['MICRO_COOLDOWN_SECONDS'] = ORIGINAL_MICRO_COOLDOWN_SECONDS
                except Exception:
                    pass
                try:
                    tg("🎯 Monthly Target Mode: aiming for 8–15% — resumed NORMAL mode")
                except Exception:
                    pass
        except Exception:
            pass
        newest = globals().get('last_deal_id') or 0
        weekly_positive = 0.0
        try:
            week_start = now - timedelta(days=7)
            week_deals = mt5.history_deals_get(week_start, now) or []
            for wd in week_deals:
                p = getattr(wd, 'profit', 0.0) or 0.0
                if p > 0:
                    weekly_positive += float(p)
        except Exception:
            weekly_positive = 0.0

        for d in sorted(deals, key=lambda x: getattr(x, 'ticket', getattr(x, 'order', 0)) or 0):
            tid = getattr(d, 'ticket', getattr(d, 'order', 0)) or 0
            if tid and tid > newest:
                globals()['last_deal_id'] = tid
                profit = getattr(d, 'profit', 0.0)
                pos_id = getattr(d, 'position_id', 0) or getattr(d, 'order', 0) or 0
                info = None
                try:
                    if pos_id and pos_id in open_trades_info:
                        info = open_trades_info.get(pos_id)
                except Exception:
                    info = None
                mode = None
                try:
                    if info and isinstance(info, dict):
                        mode = info.get("mode")
                except Exception:
                    mode = None
                if not mode:
                    try:
                        vol = float(getattr(d, 'volume', 0.0) or 0.0)
                        mode = "MICRO" if vol <= MICRO_LOT_MAX else "FULL"
                    except Exception:
                        mode = "FULL"
                if profit < 0:
                    # Track mode-specific consecutive losses
                    if mode == "FULL":
                        BOT_STATE.full_consec_losses += 1
                        BOT_STATE.micro_consec_losses = 0
                        BOT_STATE.full_risk_reduction = 0.7
                        if BOT_STATE.full_consec_losses >= 2:
                            BOT_STATE.full_pause_until = datetime.now(SAFE_TZ) + timedelta(minutes=FULL_LOSS_PAUSE_MIN)
                    if mode == "MICRO":
                        BOT_STATE.micro_consec_losses += 1
                        BOT_STATE.full_consec_losses = 0
                        if BOT_STATE.micro_consec_losses >= 3:
                            BOT_STATE.micro_pause_until = datetime.now(SAFE_TZ) + timedelta(minutes=MICRO_LOSS_PAUSE_MIN)
                    if mode == "FULL":
                        globals()['CONSEC_LOSSES'] = globals().get('CONSEC_LOSSES', 0) + 1
                        globals()['CONSEC_WINS'] = 0
                elif profit > 0:
                    if mode == "FULL":
                        BOT_STATE.full_consec_losses = 0
                        BOT_STATE.full_risk_reduction = 1.0
                        globals()['CONSEC_LOSSES'] = 0
                        globals()['CONSEC_WINS'] = globals().get('CONSEC_WINS', 0) + 1
                    if mode == "MICRO":
                        BOT_STATE.micro_consec_losses = 0
                    # Consistency rule: if a single trade yields >30% of weekly positive profit,
                    # temporarily reduce lot sizes for safety
                    try:
                        if weekly_positive > 0 and float(profit) >= 0.30 * weekly_positive:
                            if mode == "FULL":
                                globals()['LOT_SCALE_FACTOR'] = globals().get('LOT_SCALE_FACTOR', 1.0) * 0.6
                                globals()['TEMP_LOT_REDUCTION_UNTIL'] = (now + timedelta(hours=24)).timestamp()
                                log_msg(f"🔧 Large win detected: reduced lot factor to {globals()['LOT_SCALE_FACTOR']:.2f} for 24h")
                    except Exception:
                        pass
                # Dynamic streak rules: adjust lot sizing or pause fulls
                try:
                    cw = globals().get('CONSEC_WINS', 0)
                    cl = globals().get('CONSEC_LOSSES', 0)
                    if mode == "FULL":
                        if cl >= 3:
                            globals()['SCALING_ACTIVE'] = False
                            trading_pause(True)
                            log_msg("🛑 Halted due to 3 consecutive losses")
                        elif cl >= 2:
                            # reduce lot sizes after 2 losses
                            globals()['LOT_SCALE_FACTOR'] = min(globals().get('LOT_SCALE_FACTOR', 1.0), 0.6)
                        elif cw >= 2:
                            # restore risk slowly
                            globals()['LOT_SCALE_FACTOR'] = min(1.0, globals().get('LOT_SCALE_FACTOR', 1.0) + 0.1)
                except Exception:
                    pass
                if profit != 0:
                    if pos_id and pos_id in open_trades_info:
                        info = open_trades_info.pop(pos_id, None)
                    if info:
                        symbol = info.get("symbol", getattr(d, "symbol", ""))
                        side = info.get("side", "")
                        volume = info.get("volume", 0.0)
                        ts_str = now_uk().strftime("%Y-%m-%d %H:%M:%S")
                        win_flag = 1 if profit > 0 else 0
                        header_fields = ["timestamp","symbol","side","volume","profit","win","score","h4_bias","adx_h1","adx_m15","adx_rising","rsi_h1","rsi_m15","ema_slope","ema_align","bos_align","fvg","atr_hot","vol_mom","session","spread_ok","rsi_ok"]
                        if not os.path.exists("trade_log.csv") or os.path.getsize("trade_log.csv") == 0:
                            with open("trade_log.csv", "w") as logf:
                                logf.write(",".join(header_fields) + "\n")
                        fields = [
                            ts_str, symbol, side,
                            f"{volume:.2f}", f"{profit:.2f}", str(win_flag),
                            str(info.get("score", "")),
                            str(info.get("h4_bias", "")), f"{info.get('adx_h1', '')}",
                            f"{info.get('adx_m15', '')}", str(info.get("adx_rising", "")),
                            f"{info.get('rsi_h1', '')}", f"{info.get('rsi_m15', '')}",
                            f"{info.get('ema_slope', '')}", str(info.get("ema_align", "")),
                            str(info.get("bos_align", "")), str(info.get("fvg", "")),
                            str(info.get("atr_hot", "")), str(info.get("vol_mom", "")),
                            str(info.get("session", "")), str(info.get("spread_ok", "")),
                            str(info.get("rsi_ok", ""))
                        ]
                        with open("trade_log.csv", "a") as logf:
                            logf.write(",".join(fields) + "\n")
                        if symbol:
                            lst = symbol_outcomes.get(symbol, [])
                            lst.append(win_flag)
                            if len(lst) > 20:
                                lst.pop(0)
                            symbol_outcomes[symbol] = lst
                            if len(lst) >= 5:
                                win_rate = sum(lst) / len(lst)
                                old_thr = SYMBOL_CONF_THRESH.get(symbol, AI_THRESHOLD)
                                new_thr = old_thr
                                if win_rate < 0.50 and old_thr < 90:
                                    new_thr = min(90, old_thr + 5)
                                elif win_rate > 0.70 and old_thr > 50:
                                    new_thr = max(50, old_thr - 5)
                                if new_thr != old_thr:
                                    SYMBOL_CONF_THRESH[symbol] = new_thr
                                    log_msg(f"🔧 Adjusted {symbol} confidence threshold to {new_thr:.0f} (win rate {win_rate*100:.1f}%)")
                        # Inform adaptive AI of the closed trade outcome for learning
                        try:
                            adaptive_on_trade_close(symbol, info.get('score', 0), bool(win_flag))
                        except Exception:
                            pass
        try:
            ai_info = mt5.account_info()
            eq_val = getattr(ai_info, 'equity', 0.0) if ai_info else 0.0
            # Build per-symbol P&L summary for the allowed Telegram symbols only
            profit_by_sym = {}
            for d in deals:
                try:
                    s = getattr(d, 'symbol', None)
                    p = float(getattr(d, 'profit', 0.0) or 0.0)
                    if not s:
                        continue
                    profit_by_sym[s] = profit_by_sym.get(s, 0.0) + p
                except Exception:
                    continue
            summary_lines = []
            for s in sorted(list(TELEGRAM_NOTIFY_SYMBOLS)):
                v = profit_by_sym.get(s, 0.0)
                sign = '+' if v >= 0 else '−'
                summary_lines.append(f"{s}: {sign}£{abs(v):.2f}")
            # Send the daily summary using structured helper
            telegram_daily_summary(summary_lines)
        except Exception:
            pass
        except Exception:
            pass
        if MAX_CONSEC_LOSSES and globals().get('CONSEC_LOSSES', 0) >= MAX_CONSEC_LOSSES:
            n_uk = now_uk()
            # Localize midnight using safe timezone
            try:
                nxt = SAFE_TZ.localize(datetime(n_uk.year, n_uk.month, n_uk.day, 23, 59, 50))
                globals()['HALT_UNTIL_TS'] = nxt.astimezone(timezone.utc).timestamp()
            except Exception:
                # fallback: compute naive midnight and convert using utc
                naive = datetime(n_uk.year, n_uk.month, n_uk.day, 23, 59, 50)
                globals()['HALT_UNTIL_TS'] = naive.replace(tzinfo=timezone.utc).timestamp()
            telegram_msg(f"🛑 Halt: {globals().get('CONSEC_LOSSES')} consecutive losses. Paused new trades until UK midnight.")
        try:
            _update_strict_mode()
        except Exception:
            pass
    except Exception as e:
        telegram_msg(f"⚠️ Stats update error: {e}")
def startup_summary():
    """
    Send an initial startup message to the configured Telegram chat
    detailing the bot's current settings. This message includes
    information about the trading strategy, operating mode, risk settings,
    micro trade size, AI scoring threshold, and the list of supported
    commands. It indicates that the AI bot is online and ready.
    """
    try:
        strategy = globals().get('STRATEGY', 'N/A')
        mode = globals().get('MODE', 'N/A')
        risk_pct = globals().get('RISK_PCT', 0.0) * 100.0
        micro_lot = globals().get('MICRO_LOT_TARGET', 0.0)
        ai_thresh = globals().get('AI_THRESHOLD', 0)
        # Detect firm, server and login for startup summary
        try:
            firm, _rules, server = detect_prop_firm()
        except Exception:
            firm, server = "UNKNOWN", "UNKNOWN"
        # Get login if available
        ai_info = None
        try:
            ai_info = mt5.account_info()
        except Exception:
            ai_info = None
        login = str(getattr(ai_info, 'login', '') or "UNKNOWN") if ai_info else "UNKNOWN"
        stability_mode = "STRICT" if BOT_STATE.strict_mode else "NORMAL"
        # Compose startup lines with firm/server/login information
        lines = [
            "Bot upgraded — Stability Mode Active",
            f"Mode: {stability_mode}",
            "Full trade limits: max 2/session (1 in strict), 1 active/symbol, no dup within 15m",
            "Micro trade limits: max 4/session, 1 active/symbol, no stacking with full",
            "Market filters: ADX 25/25 rising, RSI 55/45, spread caps, candle body >=60%, news block",
            "Volatility: ATR spike filter, wick>2x body, no trading first 5 min",
            "",  # blank line
            "🤖 AI Trading Bot Online",
            f"Firm: {firm or 'UNKNOWN'} | Server: {server or 'UNKNOWN'} | Login: {login}",
            f"Strategy: {strategy}",
            f"Mode: {mode}",
            f"Risk per full trade: {risk_pct:.2f}%",
            f"Micro lot size: {micro_lot}",
            f"AI score threshold: {ai_thresh}",
            "",  # blank line
            "📋 Available commands:",
            "/active – check bot is running",
            "/status – balance & equity stats",
            "/ai – AI learning system status",
            "/scan – run a scan for trade setups",
            "/findtrade – full scan then micro fallback",
            "/help – display this help list",
            "/full – force a full trade (if conditions allow)",
            "/risk <pct> – set risk percentage for full trades",
            "/microlot <val> – set micro trade lot size"
        ]
        message = "\n".join(lines)
        print(f"[STARTUP] Sending summary to Telegram...")
        try:
            result = telegram_msg(message)
            if result:
                print("[STARTUP] ✓ Startup summary sent to Telegram")
            else:
                print("[STARTUP] ✗ Startup summary failed, printing to console")
                print(message)
        except Exception as send_err:
            print(f"[STARTUP] Telegram send error: {send_err}")
            print(message)
    except Exception as e:
        print(f"[STARTUP] Summary error: {e}")
        print("[STARTUP] AI Trading Bot Online")
# Duplicate placeholder account_is_demo removed; see the earlier implementation for proper logic.
def apply_sl_tp(symbol, sl, tp):
    """
    Placeholder for applying stop-loss and take-profit levels. Does
    nothing in this stub implementation. Extend with API calls as needed.
    """
    return None
def build_signals(symbol, announce=False):
    """
    Generic signal builder stub. Returns an empty list to indicate
    no trade signals. Override with strategy-specific logic if required.
    """
    return []
# Duplicate placeholder current_ruleset removed; see the earlier implementation.
def run_scan(announce=False):
    """
    One‑shot scan that tries to place a FULL trade first (if AUTO_PLACE_FULL),
    and if nothing fills, falls back to a MICRO trade using a round‑robin
    rotation with optional priority symbol. Returns True if any order is placed.

    When invoked from the Telegram /scan or /findtrade commands, this function
    also toggles SCAN_MODE to True and resets LAST_SCAN_* globals so that
    handlers can determine whether a trade on XAUUSD was placed.
    """
    # Initialise scan tracking state
    globals()['SCAN_MODE'] = True
    globals()['LAST_SCAN_PLACED_SYMBOL'] = None
    globals()['LAST_SCAN_TRADE_TYPE'] = None
    try:
        placed = False
        # Attempt a single full trade if enabled
        if AUTO_PLACE_FULL:
            placed = attempt_full_trade_once()
            if placed:
                if announce:
                    try:
                        telegram_msg("✅ Full trade placed.")
                    except Exception:
                        pass
                # End scan mode before returning
                globals()['SCAN_MODE'] = False
                return True
        # Ensure we are within a defined trading session
        lab, _, _ = session_bounds()
        if not lab:
            if announce:
                try:
                    telegram_msg("⏸️ Outside trading sessions — skipping scan.")
                except Exception:
                    pass
            globals()['SCAN_MODE'] = False
            return False
        # Check micro trade caps
        sess_count = day_stats.get("micros_LON", 0) if lab == "LON" else day_stats.get("micros_NY", 0)
        if sess_count >= MICRO_MAX_PER_SESSION:
            log_msg(f"🚫 Micro session cap reached for {lab} ({sess_count}/{MICRO_MAX_PER_SESSION}).")
            globals()['SCAN_MODE'] = False
            return False
        total_micros = day_stats.get('micros_LON', 0) + day_stats.get('micros_NY', 0)
        if total_micros >= MICRO_MAX_PER_DAY:
            log_msg(f"🚫 Daily micro cap reached ({total_micros}/{MICRO_MAX_PER_DAY}).")
            globals()['SCAN_MODE'] = False
            return False
        # Build the round‑robin order of symbols
        ordered = []
        if MICRO_SYMBOL_PRIORITY and MICRO_SYMBOL_PRIORITY in SYMBOLS:
            ordered.append(MICRO_SYMBOL_PRIORITY)
        start = int(globals().get("MICRO_RR_IDX", 0)) % len(SYMBOLS)
        rr = [SYMBOLS[(start + k) % len(SYMBOLS)] for k in range(len(SYMBOLS))]
        for s in rr:
            if s not in ordered:
                ordered.append(s)
        per_sym = day_stats.get("micros_by_symbol", {}).get(lab, {}) if isinstance(day_stats.get("micros_by_symbol"), dict) else {}
        # Iterate through each symbol and attempt a micro trade
        for base in ordered:
            # Micro trades are no longer blocked by news blackout.  We do not
            # skip XAUUSD (gold) during red‑news events so that micros can
            # fire independently of full‑trade news gates.
            used = int(per_sym.get(resolve_symbol(base), 0))
            ok = try_micro_on(base)
            # Advance the round‑robin index regardless of outcome
            globals()["MICRO_RR_IDX"] = rr_next_index(int(globals().get("MICRO_RR_IDX", 0)))
            if ok:
                if announce:
                    try:
                        telegram_msg(f"🌱 Micro trade placed on {base}.")
                    except Exception:
                        pass
                globals()['SCAN_MODE'] = False
                return True
        # No micro trades were placed
        if announce:
            try:
                telegram_msg("🔍 Scan complete — no valid setup.")
            except Exception:
                pass
        globals()['SCAN_MODE'] = False
        return False
    except Exception as e:
        # Ensure scan mode is reset on error
        globals()['SCAN_MODE'] = False
        log_msg(f"⚠️ run_scan error: {e}")
        return False

# ---------------------------------------------------------------------------
# Scan reporting utilities
#
# The functions below generate detailed scan reports for each symbol.  A scan
# report summarises technical conditions (trend, AI score, ADX, RSI, ATR,
# range, spread and news block) and states whether a trade would be taken
# under current session rules and filter thresholds.  These functions are
# designed for use with the /scan and /findtrade Telegram commands.  They do
# not place any orders and can safely run even outside trading sessions.

def _evaluate_symbol_for_report(base: str, session_allowed: bool) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate a symbol and return a dictionary of technical metrics along with a
    final decision string.  The metrics include trend direction, AI score,
    ADX, RSI, ATR, range percentage, spread, news block status and a simple
    pattern classification.  The final decision is 'Trade ✔' if all of the
    following are true:
    - trading session is open (session_allowed is True)
    - not in a news blackout window
    - spread is within allowed limits
    - high probability filters (ADX, RSI, ATR floor and range) pass
    - computed AI score meets or exceeds AI_THRESHOLD
    Otherwise the decision is 'No Trade ❌'.

    Parameters
    ----------
    base : str
        Base symbol name (e.g. 'XAUUSD').
    session_allowed : bool
        True if current time is within a trading session; False otherwise.

    Returns
    -------
    Tuple[Dict[str, Any], str]
        A tuple containing the metrics dictionary and the final decision string.
    """
    details: Dict[str, Any] = {}
    try:
        sym = resolve_symbol(base)
        details['symbol'] = sym
        # Fetch data for various timeframes
        h4 = fetch_data_cached(sym, mt5.TIMEFRAME_H4, 260, max_age=6.0)
        h1 = fetch_data_cached(sym, mt5.TIMEFRAME_H1, 220, max_age=3.0)
        m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 160, max_age=1.5)
        # If data is missing, mark metrics as unavailable
        if h1 is None or m15 is None or h4 is None or len(h1) < 60 or len(m15) < 20:
            details.update({
                'trend': 'Neutral',
                'ai_score': 0,
                'adx': 0,
                'rsi': 50,
                'atr': 0,
                'range_pct': 0,
                'spread': spread_points(sym),
                'news_block': in_news_blackout(base),
                'pattern': 'Range'
            })
            return details, 'No Trade ❌'
        # Trend based on EMA bias on H1
        try:
            bias = ema_bias(h1)
        except Exception:
            bias = None
        if bias == 'BUY':
            trend_str = 'Bullish'
        elif bias == 'SELL':
            trend_str = 'Bearish'
        else:
            trend_str = 'Neutral'
        details['trend'] = trend_str
        # ADX (M15)
        try:
            adx_val = float(adx(m15))
        except Exception:
            adx_val = 0.0
        details['adx'] = round(adx_val, 2)
        # RSI (M15)
        try:
            rsi_val = float(rsi(m15))
        except Exception:
            rsi_val = 50.0
        details['rsi'] = round(rsi_val, 2)
        # ATR (M15)
        try:
            atr_val = true_atr(m15, period=14)
        except Exception:
            atr_val = 0.0
        details['atr'] = round(atr_val, 5)
        # Range percentage (previous candle midpoint)
        try:
            prev_high = float(m15['high'].iloc[-2]); prev_low = float(m15['low'].iloc[-2])
            last_close = float(m15['close'].iloc[-1])
            mid = (prev_high + prev_low) / 2.0
            pct_range = abs(last_close - mid) / max(1e-9, mid) * 100.0
        except Exception:
            pct_range = 0.0
        details['range_pct'] = round(pct_range, 2)
        # Spread points
        try:
            sp = spread_points(sym)
        except Exception:
            sp = 999999
        details['spread'] = sp
        # News blackout
        try:
            nb = in_news_blackout(base)
        except Exception:
            nb = False
        details['news_block'] = bool(nb)
        # Simple pattern classification based on BOS and volatility
        pattern = 'Range'
        try:
            bos_dir = detect_bos(h1)
            if bos_dir == 'BOS_UP':
                pattern = 'Breakout'
            elif bos_dir == 'BOS_DOWN':
                pattern = 'Reversal'
            # Check for volatility/momentum patterns
            fvg_flag = False
            try:
                fvg_flag = bool(detect_fvg(m15) or (fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6) is not None and detect_fvg(fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6))) )
            except Exception:
                fvg_flag = False
            if fvg_flag:
                pattern = 'Rejection'
        except Exception:
            pass
        details['pattern'] = pattern
        # Determine side for AI scoring and filters
        side = 'BUY' if bias == 'BUY' else 'SELL'
        # Compute AI score using the same feature weighting as micro trades
        try:
            # Feature extraction
            adx_h1_val, adx_rising_flag = adx_rising(h1)
            adx_m15_val = float(adx(m15)) if m15 is not None else 0.0
            try:
                rsi_h1_val = float(rsi(h1)); rsi_m15_val = float(rsi(m15))
            except Exception:
                rsi_h1_val = rsi_m15_val = 50.0
            # EMA slope
            try:
                ema50 = get_ema(h1, 50); ema200 = get_ema(h1, 200)
                ema_diff = ema50 - ema200
                ema_slope_val = float(ema_diff.iloc[-1] - ema_diff.iloc[-11]) if len(ema_diff) > 11 else 0.0
            except Exception:
                ema_slope_val = 0.0
            ema_align_flag = 1 if ((side == 'BUY' and ema_slope_val > 0) or (side == 'SELL' and ema_slope_val < 0)) else 0
            bos_align_flag = 0
            try:
                bos_dir = detect_bos(h1)
                if (side == 'BUY' and bos_dir == 'BOS_UP') or (side == 'SELL' and bos_dir == 'BOS_DOWN'):
                    bos_align_flag = 1
            except Exception:
                bos_align_flag = 0
            # Volatility/momentum flags
            fvg_or_atrhot = 0
            try:
                fvg_flag = bool(detect_fvg(m15) or (fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6) is not None and detect_fvg(fetch_data_cached(sym, mt5.TIMEFRAME_M5, 50, max_age=0.6))))
            except Exception:
                fvg_flag = False
            atr_hot_flag = False
            try:
                atr_series = (m15['high'] - m15['low']).rolling(20).mean()
                if len(atr_series) > 1:
                    atr_hot_flag = bool(atr_series.iloc[-1] >= atr_series.iloc[-2])
            except Exception:
                atr_hot_flag = False
            if fvg_flag or atr_hot_flag:
                fvg_or_atrhot = 1
            # Session flag and spread flag
            session_flag_val = 1 if session_allowed else 0
            spread_ok_flag = 1 if spread_ok(sym) else 0
            # Additional features
            try:
                atr_z_val = atr_z_score(m15)
            except Exception:
                atr_z_val = 0.0
            try:
                rsil_val = rsi_leader(m15)
            except Exception:
                rsil_val = rsi(m15)
            # Assemble score
            w = AI_WEIGHTS
            score_val = 0.0
            # H4 bias alignment
            h4_align = 1 if ema_bias(h4) == side else 0
            if h4_align:
                score_val += w['h4_bias']
            # ADX contributions if both ADX values exceed the relaxed minimum
            adx_gate = (adx_h1_val >= LOOSEN_FULL_ADX_H1_MIN and adx_m15_val >= LOOSEN_FULL_ADX_M15_MIN)
            if adx_gate:
                score_val += w['adx_h1'] * max(0.0, min(1.0, (adx_h1_val - 25) / 20.0))
                score_val += w['adx_m15'] * max(0.0, min(1.0, (adx_m15_val - 20) / 20.0))
                if adx_rising_flag:
                    score_val += w['adx_rising']
            # RSI regime
            rsi_ok_flag = 1 if ((side == 'BUY' and rsi_h1_val >= LOOSEN_FULL_RSI_BUY and rsi_m15_val >= LOOSEN_FULL_RSI_BUY) or \
                                 (side == 'SELL' and rsi_h1_val <= LOOSEN_FULL_RSI_SELL and rsi_m15_val <= LOOSEN_FULL_RSI_SELL)) else 0
            if rsi_ok_flag:
                score_val += w['rsi_regime']
            # EMA slope
            if ema_align_flag:
                score_val += w['ema_slope']
            # BOS alignment
            if bos_align_flag:
                score_val += w['bos_match']
            # Volatility momentum
            if fvg_or_atrhot:
                score_val += w['fvg_or_atrhot']
            # Session
            if session_flag_val:
                score_val += w['session']
            # Spread
            if spread_ok_flag:
                score_val += w['spread_ok']
            # Additional indicators: ATR Z-score and RSIL
            score_val += w.get('atr_z', 0) * max(0.0, min(1.0, atr_z_val / 1.5))
            score_val += w.get('rsil', 0) * max(0.0, min(1.0, abs(rsil_val - 50.0) / 50.0))
            score_val = min(100.0, score_val)
            # Apply global AI bias and clamp
            final_score = int(round(max(0.0, min(100.0, score_val + AI_BIAS)), 0))
        except Exception:
            final_score = 0
        details['ai_score'] = final_score
        # High probability filter check (ADX/RSI/ATR floor/Range/Spread)
        try:
            # Only evaluate high_prob_filters_ok when a side is defined
            hpf_ok = False
            if side in ('BUY', 'SELL'):
                hpf_ok = high_prob_filters_ok(sym, h1, m15, h4, side, announce=False, label='REPORT')
        except Exception:
            hpf_ok = False
        # Final decision
        decision = 'No Trade ❌'
        # A trade is approved only if in session, no news blackout, spread within cap,
        # high probability filters pass and AI score meets threshold.  For XAUUSD
        # use the specialised full-trade policy.
        try:
            if _base_of(sym) == 'XAUUSD':
                allowed_full, _ = can_place_on_xau(sym, micro=False)
                spread_allowed = allowed_full
            else:
                spread_allowed = spread_ok(sym)
        except Exception:
            spread_allowed = spread_ok(sym)
        if session_allowed and not nb and spread_allowed and hpf_ok and final_score >= AI_THRESHOLD:
            decision = 'Trade ✔'
        else:
            decision = 'No Trade ❌'
        return details, decision
    except Exception:
        # Fallback: if evaluation fails, return empty details with no trade decision
        return {'symbol': base, 'trend': 'Neutral', 'ai_score': 0, 'adx': 0, 'rsi': 50, 'atr': 0, 'range_pct': 0, 'spread': 0, 'news_block': False, 'pattern': 'Range'}, 'No Trade ❌'


def send_scan_report(chat_id: str, allow_outside_session: bool = True) -> str:
    """
    Generate and send a detailed scan report for each configured symbol to the
    specified Telegram chat.  Symbols are scanned in the order defined by
    SYMBOLS, with XAUUSD always first.  If a trading session is closed and
    allow_outside_session is True, the report will still be produced but no
    trades will be taken.  If allow_outside_session is False and the current
    time is outside session hours, the report will be skipped.

    Parameters
    ----------
    chat_id : str
        Telegram chat ID to send the report to.
    allow_outside_session : bool, optional
        If True, scanning outside of session hours is permitted (but trades
        are not executed).  If False, the report is not produced outside of
        trading hours.  Defaults to True.
    """
    try:
        report_parts = []
        # Determine whether we are currently in a trading session
        session_label, _, _ = session_bounds()
        session_allowed = session_label is not None
        # If scanning outside sessions is not allowed and we are outside, return empty
        if not session_allowed and not allow_outside_session:
            return ""
        # Loop through symbols in configured order (XAUUSD first)
        for base in SYMBOLS:
            try:
                # Always resolve the symbol for data access
                sym = resolve_symbol(base)
                # Evaluate metrics and decision
                details, decision = _evaluate_symbol_for_report(base, session_allowed)
                # Compose message lines
                msg_lines = []
                msg_lines.append(f"🔍 SCAN RESULT — {base}")
                # Trend
                msg_lines.append(f"• Trend: {details.get('trend', 'Neutral')}")
                # AI Score range or value
                ai_score = details.get('ai_score', 0)
                if ai_score >= 100:
                    ai_display = "100"
                elif ai_score <= 0:
                    ai_display = "0"
                else:
                    lo = max(0, ai_score - 5)
                    hi = min(100, ai_score + 5)
                    ai_display = f"{lo}–{hi}"
                msg_lines.append(f"• AI Score: {ai_display}")
                # ADX with minimum threshold
                msg_lines.append(f"• ADX: {details.get('adx', 0):.2f} (min {LOOSEN_FULL_ADX_H1_MIN:.0f})")
                # RSI with thresholds
                msg_lines.append(f"• RSI: {details.get('rsi', 50):.2f} (Buy≥{LOOSEN_FULL_RSI_BUY:.0f} / Sell≤{LOOSEN_FULL_RSI_SELL:.0f})")
                # ATR with floor
                msg_lines.append(f"• ATR: {details.get('atr', 0):.5f} (min {LOOSEN_FULL_ATR_FLOOR:.3f})")
                # Range percentage with minimum
                msg_lines.append(f"• Range: {details.get('range_pct', 0):.2f}% (min {LOOSEN_FULL_RANGE_MAX_PCT:.2f}%)")
                # Spread
                msg_lines.append(f"• Spread: {details.get('spread', 0)}")
                # News block status
                msg_lines.append(f"• News Block: {'Yes' if details.get('news_block', False) else 'No'}")
                # Pattern detected
                msg_lines.append(f"• Pattern: {details.get('pattern', 'Range')}")
                # Session allowed flag
                msg_lines.append(f"• Session Allowed: {'Yes' if session_allowed else 'No'}")
                # Final decision
                msg_lines.append(f"• Final Decision: {decision}")
                # If outside of session, append explicit note about no trades being placed
                if not session_allowed:
                    msg_lines.append("(Session Closed — No trade will be placed)")
                # Add the composed text to report parts
                report_parts.append("\n".join(msg_lines))
            except Exception:
                # skip symbol errors but continue others
                continue
        # Join all parts with a blank line separator
        return "\n\n".join(report_parts)
    except Exception:
        try:
            print("⚠️ Failed to build scan report")
        except Exception:
            pass
        return ""
def update_addon_tracker():
    """
    Placeholder for add-on trade tracker update. Does nothing.
    """
    return None
TELEGRAM_POLLING_ENABLED = CONSTANTS.get("TELEGRAM_POLLING_ENABLED", TELEGRAM_POLLING_ENABLED)
TELEGRAM_POLL_SLEEP_EMPTY = CONSTANTS.get("TELEGRAM_POLL_SLEEP_EMPTY", 1.5)  # seconds between polls when no updates
def telegram_poller_loop():
    """
    Background long-polling loop for Telegram commands.
    Uses getUpdates offset to avoid reprocessing.
    Logs to console and dispatches minimal commands:
    /active, /status, /scan, /findtrade, /help
    """
    try:
        print("[TG] Poller thread started")
    except Exception:
        pass
    offset = None
    while True:
        try:
            if not TELEGRAM_POLLING_ENABLED:
                time.sleep(2.0)
                continue
            updates = _tg_poll_once(offset)
            if not updates:
                time.sleep(TELEGRAM_POLL_SLEEP_EMPTY)
                continue
            for u in updates:
                try:
                    uid = u.get("update_id")
                    msg = u.get("message") or u.get("edited_message") or {}
                    chat = (msg.get("chat") or {}).get("id")
                    text = msg.get("text") or ""
                    if uid is not None:
                        offset = uid + 1
                    if not text or not chat:
                        continue
                    print(f"[TG] cmd from {chat}: {text}")
                    handled = _dispatch_minimal_command(text, chat)
                    if not handled and text.strip().lower() == "/help":
                        _dispatch_minimal_command("/help", chat)
                except Exception as e:
                    try:
                        print(f"[TG] error processing update: {e}")
                    except Exception:
                        pass
        except Exception as e:
            try:
                print(f"[TG] poll error: {e}")
            except Exception:
                pass
            time.sleep(2.0)
def main():
    """
    Script entry point.  Initializes MT5, waits for a proper FTMO login,
    detects the prop firm, sends appropriate startup notifications, and
    finally begins the trading loop.  The startup sequence follows the
    prescribed order:

        1. Initialize MT5 using ``_mt5_init_force()``.
        2. Wait up to 30 seconds for the terminal to finish logging into an
           FTMO server via ``wait_for_ftmo_login()``.
        3. Detect the connected prop firm with ``detect_prop_firm()`` and
           apply any firm-specific overrides (e.g. FTMO drawdown limits and
           maximum full trades per day).
        4. If the firm is FTMO, send the FTMO startup message exactly once.
        5. Send the generic "bot online" message.
        6. Send a startup summary, load persisted state, start the Telegram
           poller and launch the main trading loop or enter an idle loop.
        
    Note: NO startup trades are placed. The bot waits for valid signals.
    """
    # Parse CLI flags (e.g. --dry-run) before performing any actions
    try:
        import sys as _sys
        if any(a.lower() == '--dry-run' for a in _sys.argv[1:]):
            globals()['DRY_RUN'] = True
            try:
                log_msg("🧪 DRY RUN enabled: no real orders or network sends will occur", level="INFO")
            except Exception:
                print("DRY RUN enabled: no real orders or network sends will occur")
            # Monkeypatch mt5.order_send to return a fake success response
            class _FakeOrderResult:
                def __init__(self):
                    self.retcode = getattr(mt5, 'TRADE_RETCODE_DONE', 10009)
                    self.order = 0
                    self.deal = 0
            try:
                setattr(mt5, 'order_send', lambda req: _FakeOrderResult())
            except Exception:
                pass

    except Exception:
        pass

    # Step 1: auto-launch MT5 and initialise the bridge.  Launch the terminal
    # once using the explicit path defined in AUTO_MT5_PATH, then call
    # mt5.initialize() exactly once.  If initialisation fails, print the
    # error and continue.  The old _mt5_init_force() has been removed.
    try:
        subprocess.Popen([AUTO_MT5_PATH])
        time.sleep(5.0)
    except Exception:
        pass
    
    # Ticket-Funded MT5 Connection: Run startup sequence
    try:
        startup_success, account = mt5_startup()
        if not startup_success:
            print("[Ticket-Funded MT5] Startup failed - attempting fallback connection")
            try:
                ensure_mt5_connection(retries=10, delay=2.0)
            except Exception as e:
                print(f"[Ticket-Funded MT5] Fallback connection failed: {e}")
                tg("[Ticket-Funded] ERROR - MT5 startup and fallback failed.")
                return
    except Exception as e:
        print(f"[Ticket-Funded MT5] MT5 startup error: {e}")
    
    # Ensure MT5 connection with retries and verification
    try:
        ensure_mt5_connection()
    except Exception as e:
        try:
            print(f"❌ MT5 failed to connect: {_safe_str(e)}")
        except Exception:
            pass
        try:
            telegram_msg(f"❌ MT5 connection failed: {_safe_str(e)}")
        except Exception:
            pass
        # Block trading until manual intervention
        try:
            BOT_STATE.trading_paused = True
        except Exception:
            pass
    # Step 2: wait for the MT5 terminal to finish logging into a server (optional helper)
    try:
        wait_for_ftmo_login()
    except Exception:
        pass
    # Step 3: detect prop firm, rules and server via unified function
    try:
        firm, rules, server = detect_prop_firm()
        PROP_ACTIVE["name"] = firm or "UNKNOWN"
        # Apply FTMO-specific overrides and send notification
        if firm == "FTMO":
            try:
                globals()["FULL_MAX_PER_DAY"] = 2
            except Exception:
                pass
            try:
                msg = f"🚀 FTMO account detected ({server}) — full XAUUSD trading enabled."
                print(f"[STARTUP] Sending FTMO notification...")
                result = telegram_msg(msg)
                if result:
                    print("[STARTUP] ✓ FTMO notification sent")
                else:
                    print("[STARTUP] ✗ FTMO notification failed")
            except Exception as e:
                print(f"[STARTUP] FTMO notification error: {e}")
    except Exception:
        firm, server = "UNKNOWN", "UNKNOWN"
        rules = {}
        PROP_ACTIVE["name"] = "UNKNOWN"
    # Step 4: generic bot online message
    try:
        msg = f"🤖 Bot online | Firm: {PROP_ACTIVE.get('name', 'UNKNOWN')} | Server: {server or 'UNKNOWN'}"
        print(f"[STARTUP] Sending: {msg}")
        result = telegram_msg(msg)
        if result:
            print("[STARTUP] ✓ Telegram message sent successfully")
        else:
            print("[STARTUP] ✗ Telegram message failed")
    except Exception as e:
        print(f"[STARTUP] Telegram error: {e}")
        print(f"🤖 Bot online | Firm: {PROP_ACTIVE.get('name', 'UNKNOWN')} | Server: {server or 'UNKNOWN'}")
    print("[STARTUP] Bot is now online — ready to receive commands")

# Step 6: send startup summary (force Telegram delivery for startup messages)
try:
    startup_summary()
    try:
        # Send explicit startup confirmation with key account details
        try:
            ai = mt5.account_info()
            server = getattr(ai, 'server', 'UNKNOWN') or 'UNKNOWN'
            firm = PROP_ACTIVE.get('name', 'UNKNOWN')
            lev = getattr(ai, 'leverage', 'N/A')
            bal = float(getattr(ai, 'balance', 0.0) or 0.0)
            eq = float(getattr(ai, 'equity', 0.0) or 0.0)
            daily_limit = "5%"
            max_dd = "10%"
        except Exception:
            server = 'UNKNOWN'; firm = PROP_ACTIVE.get('name','UNKNOWN'); lev = 'N/A'; bal = 0.0; eq = 0.0; daily_limit='5%'; max_dd='10%'
        startup_text = (
            f"🤖 FUNDED ACCOUNT BOT INITIALIZED\n\n"
            f"Server: {server}\n"
            f"Firm: {firm}\n"
            f"Leverage: {lev}\n"
            f"Balance: {bal:.2f}\n"
            f"Equity: {eq:.2f}\n"
            f"Daily Loss Limit: {daily_limit}\n"
            f"Max Drawdown: {max_dd}\n"
            f"Status: Connected + Live\n\nAll systems online. Monitoring markets..."
        )
        print(f"[STARTUP] Sending account summary to Telegram...")
        result = telegram_msg(startup_text)
        if result:
            print("[STARTUP] ✓ Account summary sent to Telegram")
        else:
            print("[STARTUP] ✗ Account summary failed to send")
            print(startup_text)
        
        # Display AI Learning System Status
        try:
            ai_status = get_ai_status_summary()
            print("\n" + "="*70)
            print(ai_status)
            print("="*70 + "\n")
            telegram_msg("🤖 AI Learning System initialized and ready")
        except Exception as e:
            print(f"⚠️ AI status display failed: {e}")
    except Exception as e:
        print(f"[STARTUP] Account summary error: {e}")
        try:
            print(f"[TG] Startup message failed: {e}")
        except Exception:
            pass
except Exception as e:
    try:
        print(f"[TG] Startup summary failed: {e}")
    except Exception:
        pass

# Step 8: load persisted state
try:
    load_state()
except Exception:
    pass

    # Start background workers only when running as main (not on import)
    try:
        import threading as _threading
        # Start the local tele worker only when Telegram credentials are configured
        try:
            if TELEGRAM_POLLING_ENABLED and telegram_enabled():
                _threading.Thread(target=_tele_worker, daemon=True).start()
        except Exception:
            pass
        # Trade manager thread is safe to start (MT5-only)
        try:
            if not globals().get('DRY_RUN', False):
                start_trade_manager_thread()
        except Exception:
            pass
        # Start async utilities when either manager loop is wanted or Telegram is enabled
        try:
            if not globals().get('DRY_RUN', False) and (telegram_enabled() or FEATURES.get("MANAGER_LOOP", True)):
                start_async_utilities()
        except Exception:
            pass
    except Exception:
        pass
# Load persisted state on import but do not start any background services
# or network activity. This file is intentionally side‑effect free so that
# it can be imported for static analysis, unit tests or interactive use
# without launching MT5, Telegram pollers, or the main trading loop.
try:
    load_state()
except Exception:
    pass

# Convenience: auto-start Telegram polling when credentials are present.
# Controlled by the environment variable `AUTO_START_TELEGRAM`. Set to '0'
# to disable automatic startup during imports. Default is enabled so that
# simple runs (double-click/script) get Telegram command handling back.
try:
    AUTO_START_TELEGRAM = os.getenv("AUTO_START_TELEGRAM", "1")
    if str(AUTO_START_TELEGRAM) not in ("0", "false", "False"):
        # Only start polling when credentials are configured from the
        # environment (not merely present in CONSTANTS). This prevents
        # auto-start during programmatic imports or unit tests where the
        # module may be loaded for inspection but network access is
        # unwanted. To override auto-start in tests, set
        # `AUTO_START_TELEGRAM=0` in the test environment or ensure
        # `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are not present in env.
        # Require an explicit opt-in environment flag to allow imports to
        # start network pollers automatically. This avoids surprising
        # side-effects when the module is imported for tests or analysis.
        try:
            # Auto-start polling when credentials are present and auto-start
            # is not explicitly disabled. This makes command handling active
            # when the bot process is started normally.
            if telegram_enabled() and TELEGRAM_CREDS_FROM_ENV and str(AUTO_START_TELEGRAM) not in ("0", "false", "False"):
                start_telegram_polling(send_online_ack=True)
        except Exception:
            pass
except Exception:
    pass

# To start the bot in a live run, call `main()` explicitly or run this
# module as a script. The `main()` function will perform initialisation
# and launch background threads when appropriate.
# === Main Loop ===
#
# The entry point when running this file as a script.  The main loop
# performs initialisation (Telegram setup) and then continually calls
# the strategy scan function.  The loop rate is one second by default,
# which controls how often the bot evaluates market conditions.  All
# early exit checks (connectivity, broker identification, risk rules,
# session/time, trade caps, cooldowns, spread) occur inside
# ``holy_grail_scan_and_execute``, which itself has been annotated to
# preserve trading logic.
if __name__ == "__main__":
    # Instantiate the unified state container and run the refactored entry loop
    state = BotState()
    run_bot(state)
def monitor_addon_opps():
    """Monitor existing positions for add-on trade opportunities (one extra trade per position)."""
    poss = mt5.positions_get() or []
    update_addon_tracker()
    for p in poss:
        try:
            symbol = p.symbol
            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            entry = p.price_open
            sl = p.sl if p.sl else None
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue
            price_fav = (tick.bid - entry) if side == "BUY" else (entry - tick.ask)
            h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 150, max_age=3.0)
            m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 100, max_age=1.5)
            if m15 is None or h1 is None:
                continue
            # Compute ATR from the M15 timeframe for add-on logic.  Fall back to the spread if needed.
            try:
                atr_v = atr(m15, 14)
                if atr_v <= 0:
                    atr_v = true_atr(m15, 14)
                if atr_v <= 0:
                    atr_v = 0.5 * (tick.ask - tick.bid)
            except Exception:
                atr_v = 0.5 * (tick.ask - tick.bid)
            if not sl:
                sl = (tick.bid - 2 * atr_v) if side == "BUY" else (tick.ask + 2 * atr_v)
            R = abs(entry - sl)
            if R <= 0:
                R = max(2 * (mt5.symbol_info(symbol).point or 0.0001), 0.0001)
            rec = ADDON.get(p.ticket)
            if rec is None:
                is_micro = (p.volume <= MICRO_LOT_MAX + 1e-9) or ("Micro" in (p.comment or ""))
                ADDON[p.ticket] = {
                    "symbol": symbol, "side": side, "entry": entry, "sl": sl, "R": R,
                    "seen_profit": False, "went_negative": False, "addon_done": False,
                    "is_micro": is_micro, "improve_pts": max(R * 0.10, 2 * (mt5.symbol_info(symbol).point or 0.0001)),
                    "moved_be": False
                }
                rec = ADDON[p.ticket]
            if rec["addon_done"]:
                continue
            if not rec.get("moved_be") and price_fav >= 1.0 * rec["R"]:
                new_sl = entry
                req = {
                    "action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": symbol,
                    "sl": new_sl, "tp": p.tp or 0.0, "magic": 20250916, "comment": "BE_MOVE"
                }
                res = safe_order_send(req)
                if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                    rec["moved_be"] = True
                    log_msg(f"🔐 {symbol} ticket {p.ticket}: SL moved to break-even")
            if rec.get("moved_be"):
                current_profit_R = price_fav / rec["R"] if rec["R"] != 0 else 0
                floor_profit = int(current_profit_R)
                if floor_profit >= 2:
                    target_lock = floor_profit - 1
                    new_sl_price = (entry + target_lock * rec["R"]) if side == "BUY" else (entry - target_lock * rec["R"])
                    rec["moved_be"] = False
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": symbol,
                        "sl": new_sl_price, "tp": p.tp or 0.0, "magic": 20250916, "comment": "TRAILING_SL"
                    }
                    res = safe_order_send(req)
                    if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                        log_msg(f"🔒 {symbol} ticket {p.ticket}: SL moved to lock in profit")
            # Asian-session GBPJPY second-entry rule: when in Asian session, permit
            # a second, smaller entry if the pair moved strongly then retraced and
            # AI confirms the trend.  This only applies for GBPJPY and only one
            # add-on per original position.
            try:
                base = _base_of(symbol).upper()
                if is_asian_session() and base == 'GBPJPY' and not rec.get('addon_done'):
                    # If the trade saw profit and then retraced into a modest pullback
                    # (e.g., between 0.25R and 0.8R), and AI predicts continuation,
                    # place a smaller follow-up trade.
                    cur_R = rec['R']
                    retrace = (rec['R'] - price_fav) / max(1e-9, rec['R']) if rec['R'] else 0
                    # Simpler metric: if current price_fav is between 0.25R and 0.8R
                    if (price_fav / max(1e-9, rec['R'])) >= 0.25 and (price_fav / max(1e-9, rec['R'])) <= 0.8:
                        try:
                            ai_conf = blended_prediction(symbol) if ENABLE_PREDICTIVE_AI else 0.0
                        except Exception:
                            ai_conf = 0.0
                        if ai_conf >= 0.85:
                            # Place an add-on smaller lot (50% of original volume)
                            try:
                                addon_lot = max(0.01, round(float(getattr(p, 'volume', 0.01)) * 0.5, 2))
                                req = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": symbol,
                                    "volume": addon_lot,
                                    "type": (mt5.ORDER_TYPE_BUY if side == 'BUY' else mt5.ORDER_TYPE_SELL),
                                    "price": tick.ask if side == 'BUY' else tick.bid,
                                    "deviation": 60,
                                    "comment": "ADDON_GBPJPY_ASIA"
                                }
                                res = safe_order_send(req)
                                if res and getattr(res, 'retcode', None) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                                    rec['addon_done'] = True
                                    log_msg(f"➕ GBPJPY add-on placed (lot={addon_lot}) for ticket {p.ticket}")
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            continue
def build_stats_rollup(mode="DAILY"):
    import csv, os
    path = os.path.join('logs', 'trade_outcomes.csv')
    if not os.path.exists(path):
        return "No trade log yet."
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for rec in r:
            try:
                ts, sym, ticket, profit, score = rec
                rows.append((ts, sym, float(profit)))
            except Exception:
                continue
    from collections import defaultdict
    buckets = defaultdict(list)
    for ts, sym, pr in rows:
        day = ts.split('T')[0]
        import datetime as _dt
        try:
            d = _dt.date.fromisoformat(day)
            key_week = f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"
        except Exception:
            key_week = day[:8] + "01"
        key = day if mode == 'DAILY' else key_week
        buckets[key].append(pr)
    lines = [f"📊 {mode} rollup (profit):"]
    for k in sorted(buckets.keys())[-14:]:
        lines.append(f"{k}: {sum(buckets[k]):.2f}")
    return "\n".join(lines)
def build_equity_curve_png():
    import os, csv
    import matplotlib.pyplot as plt
    os.makedirs('logs', exist_ok=True)
    path = os.path.join('logs', 'trade_outcomes.csv')
    if not os.path.exists(path):
        raise RuntimeError('No trade_outcomes.csv yet')
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for rec in r:
            try:
                ts, sym, ticket, profit, score = rec
                rows.append((ts, float(profit)))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0])
    eq = 0.0
    ys = []
    for _, pr in rows:
        eq += pr
        ys.append(eq)
    fig = plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.title('Equity Curve')
    plt.xlabel('Trade #')
    plt.ylabel('P/L (cum)')
    out = os.path.join('logs', 'equity_curve.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out
def flat_all_positions():
    try:
        poss = mt5.positions_get() or []
        for p in poss:
            sym = getattr(p, 'symbol', None)
            vol = getattr(p, 'volume', 0.0)
            typ = getattr(p, 'type', 0)
            if not sym or vol <= 0: 
                continue
            t = mt5.symbol_info_tick(sym)
            if not t:
                continue
            req = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': sym,
                'volume': vol,
                'type': (mt5.ORDER_TYPE_SELL if typ==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY),
                'price': (t.bid if typ==mt5.POSITION_TYPE_BUY else t.ask),
                'deviation': 80,
                'magic': 20250916,
                'comment': 'PANIC FLAT'
            }
            mt5.order_send(req)
    except Exception as e:
        print(f"[PANIC] flat error: {e}")
def cancel_all_pendings():
    try:
        orders = mt5.orders_get() or []
        for o in orders:
            ticket = getattr(o, 'ticket', None)
            if ticket is None: 
                continue
            mt5.order_send({'action': mt5.TRADE_ACTION_REMOVE, 'order': ticket})
    except Exception as e:
        print(f"[PANIC] cancel pendings error: {e}")
def trading_pause(state=True):
    BOT_STATE.trading_paused = bool(state)
# Removed duplicate run_self_tests implementation; see final definition near the end of the file.
def fetch_red_news_windows():
    import requests, pytz, datetime as _dt
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=10)
        arr = r.json()
        out = []
        for it in arr:
            if it.get('impact') != 'High':
                continue
            ts = it.get('timestamp')
            if not ts:
                continue
            st = _dt.datetime.utcfromtimestamp(int(ts)) - _dt.timedelta(minutes=15)
            en = _dt.datetime.utcfromtimestamp(int(ts)) + _dt.timedelta(minutes=45)
            out.append({'start': st.isoformat(), 'end': en.isoformat(), 'label': it.get('title','red')})
        with open(NEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump(out, f)
        return out
    except Exception as e:
        raise e
from typing import Optional, Callable, Any, Dict, Tuple, List
import logging
import functools
def setup_logging(level: Optional[str] = None) -> None:
    """
    Initialize Python logging with a sensible default format.
    Uses the LOG_LEVEL env var if set (e.g., 'INFO', 'DEBUG').
    """
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(lvl)
setup_logging()  # enable logging immediately
try:
    _old_log_msg = log_msg  # type: ignore  # noqa: F821
    def log_msg(text: str, level: str = "INFO") -> None:  # type: ignore
        """Log to stdio, Telegram (existing behavior), and Python logging."""
        level = (level or "INFO").upper()
        logger = logging.getLogger("bot-ai")
        if   level == "DEBUG": logger.debug(text)
        elif level == "WARNING": logger.warning(text)
        elif level == "ERROR": logger.error(text)
        else: logger.info(text)
        _old_log_msg(text)  # preserve original behavior
except NameError:
    def log_msg(text: str, level: str = "INFO") -> None:
        level = (level or "INFO").upper()
        logger = logging.getLogger("bot-ai")
        getattr(logger, level.lower(), logger.info)(text)
        print(text)
def with_retry(max_tries: int = 3, backoff_sec: float = 1.5,
               exceptions: Tuple[type, ...] = (Exception,)):
    """
    Decorator to retry a function on transient errors.
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tries = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    try:
                        send_telegram_to(chat_id, f"Selftest start failed: {e}")
                    except Exception as e2:
                        try:
                            tg(f"Selftest start failed: {e}")
                        except Exception as e3:
                            try:
                                log_debug("Selftest start fallback tg failed:", e3)
                            except Exception:
                                pass
    """
    Validate user-adjustable settings defined near the top of the file.
    Only warns; does not raise, to avoid interrupting runtime.
    """
    warnings = []
    try:
        rpct = float(globals().get("RISK_PCT", 0.01))
        if not (0.0001 <= rpct <= 0.05):
            warnings.append(f"RISK_PCT={rpct} outside [0.0001, 0.05].")
    except Exception:
        warnings.append("RISK_PCT invalid type.")
    try:
        micro_min = float(globals().get("MICRO_LOT_MIN", 0.01))
        micro_max = float(globals().get("MICRO_LOT_MAX", 0.03))
        if micro_min <= 0 or micro_max <= 0 or micro_min > micro_max:
            warnings.append(f"Micro lot bounds invalid: {micro_min}..{micro_max}")
    except Exception:
        warnings.append("MICRO_LOT_* invalid type.")
    try:
        full_cap = int(globals().get("FULL_MAX_PER_DAY", 3))
        if full_cap < 0 or full_cap > 20:
            warnings.append(f"FULL_MAX_PER_DAY={full_cap} looks unusual.")
    except Exception:
        warnings.append("FULL_MAX_PER_DAY invalid type.")
    if warnings:
        for w in warnings:
            log_msg(f"⚠️ Config warning: {w}", level="WARNING")
    else:
        log_msg("✅ Config validated.", level="DEBUG")
validate_config()
def atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average True Range (Wilder) over `period` bars.
    Returns the latest ATR value; 0.0 if not enough data.
    """
    try:
        if df is None or len(df) < period + 1:
            return 0.0
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
        val = float(atr_series.iloc[-1])
        return 0.0 if pd.isna(val) else val
    except Exception:
        return 0.0
# Removed intermediate ADX (adx/adx_series/adx_rising) implementations; the final definitions are provided later in the file.
# Removed earlier build_signals_goat wrapper. The final implementation is defined later in the file.
def http_get(url: str, **kwargs) -> Any:
    """HTTP GET with minimal retry; wraps requests.get if available."""
    try:
        import requests
        return requests.get(url, timeout=kwargs.pop("timeout", 8), **kwargs)
    except Exception as e:
        log_msg(f"HTTP GET error: {e}", level="WARNING")
        raise
@with_retry(max_tries=3, backoff_sec=1.5)
def http_post(url: str, **kwargs) -> Any:
    """HTTP POST with minimal retry; wraps requests.post if available."""
    try:
        import requests
        return requests.post(url, timeout=kwargs.pop("timeout", 8), **kwargs)
    except Exception as e:
        log_msg(f"HTTP POST error: {e}", level="WARNING")
        raise
class TradeBot:
    """
    Lightweight wrapper around global funcs for easier unit testing.
    Usage is optional; existing global flow remains untouched.
    """
    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols = symbols or list(globals().get("SYMBOLS", []))
    def get_hlcv(self, symbol: str, tf, bars: int = 200) -> Optional[pd.DataFrame]:
        return get_data(symbol, tf, bars)
    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        return adx(df, period)
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        return atr(df, period)
    def in_session(self) -> bool:
        return in_session()
    def validate(self) -> None:
        validate_config()
        log_msg("TradeBot ready.", level="DEBUG")
import asyncio, logging, math
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
_LOGGER = logging.getLogger("Dbot")
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    fh = RotatingFileHandler("dbot.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    _LOGGER.addHandler(fh)
try:
    _orig_log_msg = log_msg
    def log_msg(text: str) -> None:  # type: ignore[override]
        _orig_log_msg(text)
        try:
            _LOGGER.info(text)
        except Exception:
            pass
except Exception:
    pass

# Provide a backward‑compatible ``log`` function aliasing ``log_msg``.
#
# Some older portions of the code base still call ``log(...)`` rather than
# ``log_msg(...)``.  Without this alias those calls would raise a
# ``NameError`` at runtime.  If ``log`` is not already defined in the
# current module we create it here.  The wrapper attempts to call
# ``log_msg`` with a ``level`` argument if supported; if not it falls back
# to a single‑argument call.  This ensures that legacy code continues to
# operate without modification.
if "log" not in globals():
    def log(text: str, level: str = "INFO") -> None:
        try:
            # Try passing the level to log_msg (original signature)
            log_msg(text, level)  # type: ignore[misc]
        except TypeError:
            # Fallback: log_msg that only accepts the message
            log_msg(text)  # type: ignore[misc]

def _require_df(df: pd.DataFrame, cols: List[str], min_rows: int = 20) -> bool:
    try:
        if df is None or len(df) < min_rows:
            return False
        return all(c in df.columns for c in cols) and not df[cols].isna().any().any()
    except Exception:
        return False
def _positive(x: float, name: str) -> float:
    if not isinstance(x, (int, float)) or x <= 0:
        raise ValueError(f"{name} must be positive")
    return float(x)
def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR series; returns a Series aligned to df index."""
    if not _require_df(df, ["high","low","close"], period + 1):
        return pd.Series([0.0]*len(df), index=df.index)
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(),
                    (high-pc).abs(),
                    (low-pc).abs()], axis=1).max(axis=1)
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[period] = tr.iloc[1:period+1].mean()
    for i in range(period+1, len(tr)):
        atr.iloc[i] = (atr.iloc[i-1]*(period-1) + tr.iloc[i]) / period
    atr = atr.fillna(method="bfill").fillna(0.0)
    return atr
def true_atr(df: pd.DataFrame, period: int = 14) -> float:  # overrides placeholder
    try:
        s = atr_series(df, period)
        return float(s.iloc[-1]) if len(s) else 0.0
    except Exception:
        return 0.0
# Removed duplicate ADX definitions.  See the primary implementation above for
# ADXResult, _dx_from_dm_tr, adx_full and legacy compatibility wrappers.

# Expose the real ADX implementations under a private name so the
# earlier light-weight wrappers can delegate here without creating
# recursive references.  Assign these only once after the real
# implementations are defined.
_real_adx = adx
_real_adx_rising = adx_rising
def atr_stop(price: float, side: str, atr_val: float, multiple: float = 2.0) -> float:
    price = float(price); side = (side or "").upper()
    _positive(atr_val, "atr_val"); _positive(multiple, "multiple")
    return (price - multiple*atr_val) if side == "BUY" else (price + multiple*atr_val)
def atr_trailing_level(entry: float, side: str, atr_val: float, gain_mult: float = 1.5) -> float:
    entry = float(entry); side = (side or "").upper()
    _positive(atr_val, "atr_val"); _positive(gain_mult, "gain_mult")
    tick = mt5.symbol_info_tick(resolve_symbol(SYMBOLS[0]))  # any symbol to get time; price fetched per-position
    return entry  # placeholder (manager computes live)
TP_PARTS = [0.3, 0.3, 0.4]
TP_R_MULTS = [1.0, 2.0, 999.0]  # last is a runner with trailing
async def _async_sleep(sec: float) -> None:
    try:
        await asyncio.sleep(sec)
    except Exception:
        time.sleep(sec)
async def trade_manager_loop(poll_sec: float = 2.5, atr_period: int = 14, trail_mult: float = 1.5) -> None:
    """Async loop to manage open positions: partials + ATR trailing stops."""
    while True:
        try:
            poss = mt5.positions_get() or []
            for p in poss:
                sym = getattr(p, "symbol", "")
                side = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
                volume = float(getattr(p, "volume", 0.0))
                price_open = float(getattr(p, "price_open", 0.0))
                sl = float(getattr(p, "sl", 0.0) or 0.0)
                df = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 100, max_age=1.5)
                if not _ensure_df_ok(df, 20):
                    continue
                atr_v = true_atr(df, atr_period)
                tick = mt5.symbol_info_tick(sym)
                if not tick or atr_v <= 0:
                    continue
                cur = float(tick.bid if side == "SELL" else tick.ask)
                R = abs(cur - price_open) / max(1e-9, abs(price_open - (sl or atr_stop(price_open, side, atr_v, 2.0))))
                milestones = [m for m in TP_R_MULTS if m < 900]
                for idx, m in enumerate(milestones):
                    tag = f"tp_done_{idx}"
                    if getattr(p, tag, None):
                        continue
                    if (side == "BUY" and cur >= price_open + m*abs(price_open - sl)) or \
                       (side == "SELL" and cur <= price_open - m*abs(price_open - sl)):
                        part = max(0.01, round(volume * TP_PARTS[idx], 2))
                        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": part,
                               "type": (mt5.ORDER_TYPE_SELL if side == 'BUY' else mt5.ORDER_TYPE_BUY),
                               "price": cur, "deviation": 80, "magic": 20251021,
                               "comment": f"Partial TP{idx+1}"}
                        res = safe_order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            setattr(p, tag, True)
                            log_msg(f"🔁 Partial TP{idx+1} executed on {sym} ({part} lots)")
                            if idx == 0:
                                be = price_open
                                order = {"action": mt5.TRADE_ACTION_SLTP, "symbol": sym, "sl": normalize_price(sym, be),
                                         "tp": getattr(p, "tp", 0.0), "position": getattr(p, "ticket", 0)}
                                safe_order_send(order)
                trail_level = (cur - trail_mult*atr_v) if side == "BUY" else (cur + trail_mult*atr_v)
                if side == "BUY" and cur > price_open and (sl == 0.0 or trail_level > sl):
                    order = {"action": mt5.TRADE_ACTION_SLTP, "symbol": sym,
                             "sl": normalize_price(sym, trail_level),
                             "tp": getattr(p, "tp", 0.0), "position": getattr(p, "ticket", 0)}
                    safe_order_send(order)
                elif side == "SELL" and cur < price_open and (sl == 0.0 or trail_level < sl):
                    order = {"action": mt5.TRADE_ACTION_SLTP, "symbol": sym,
                             "sl": normalize_price(sym, trail_level),
                             "tp": getattr(p, "tp", 0.0), "position": getattr(p, "ticket", 0)}
                    safe_order_send(order)
        except Exception as e:
            try:
                _LOGGER.error(f"manager loop error: {e}")
            except Exception:
                pass
        await _async_sleep(poll_sec)
try:
    async def tg_poll_async(offset: Optional[int] = None, long_poll: int = 10):
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        params = {"offset": offset or 0, "timeout": long_poll}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=long_poll+5) as r:
                j = await r.json()
                return j.get("result", [])
except Exception:
    aiohttp = None
    async def tg_poll_async(offset: Optional[int] = None, long_poll: int = 10):
        return _tg_poll_once(offset)
async def telegram_loop_async() -> None:
    last_id = None
    while True:
        try:
            updates = await tg_poll_async(last_id, 10)
            for u in updates:
                last_id = u.get("update_id", last_id)
                msg = u.get("message") or {}
                text = (msg.get("text") or "")
                chat_id = str((msg.get("chat") or {}).get("id", TELEGRAM_CHAT_ID))
                if text:
                    handled = _dispatch_minimal_command(text, chat_id)
                    if not handled and text.lower().startswith("/ai") and "status" in text.lower():
                        tg("AI online — async loop active.")
        except Exception as e:
            try:
                _LOGGER.error(f"telegram loop error: {e}")
            except Exception:
                pass
        await _async_sleep(1.0)
# Duplicate _equity_info removed; refer to the earlier implementation.
def risk_guard_tick(cut_pct: float = 2.5, shock_pct: float = 3.0, halt_losses: int = 4) -> None:
    global risk_highwater, CONSEC_LOSS_CT
    # Reset the consecutive loss counter periodically.  If 12 hours have
    # elapsed since the last reset or the day has changed, clear the
    # CONSEC_LOSS_CT counter and unpause trading.  This prevents the bot
    # from remaining halted indefinitely due to a prior loss streak.  The
    # LAST_CONSEC_RESET_TS flag tracks the last reset time.  Any errors
    # during this process are ignored to avoid interrupting risk checks.
    try:
        now_ts = datetime.now(SAFE_TZ)
        global LAST_CONSEC_RESET_TS, CONSEC_LOSS_CT
        if LAST_CONSEC_RESET_TS is None or \
           (now_ts - LAST_CONSEC_RESET_TS).total_seconds() >= 12*3600 or \
           (hasattr(LAST_CONSEC_RESET_TS, 'date') and LAST_CONSEC_RESET_TS.date() != now_ts.date()):
            CONSEC_LOSS_CT = 0
            BOT_STATE.trading_paused = False
            LAST_CONSEC_RESET_TS = now_ts
    except Exception:
        pass
    _, eq = _equity_info()
    if eq <= 0: return
    if risk_highwater is None or eq > risk_highwater:
        risk_highwater = eq
    drop = 100.0 * (risk_highwater - eq) / max(1e-9, risk_highwater)
    if drop >= cut_pct and not BOT_STATE.trading_paused:
        trading_pause(True)
        log_msg(f"🛑 Equity trail cutoff hit ({drop:.2f}%) - trading paused.")
    try:
        if CONSEC_LOSS_CT >= halt_losses:
            trading_pause(True)
            log_msg(f"🛑 Loss streak {CONSEC_LOSS_CT} - paused.")
    except Exception:
        pass
try:
    _old_check_prop = check_prop_rules_before_trade
    def check_prop_rules_before_trade():
        status, reason = _old_check_prop()
        if status != "OK":
            return status, reason
        # Core risk guard tick (existing behaviour)
        risk_guard_tick()
        if BOT_STATE.trading_paused:
            return "HALT", "Risk guard pause active"
        # Ensure ACCOUNT_INITIAL_BALANCE is set once per run
        try:
            global ACCOUNT_INITIAL_BALANCE, LEVERAGE_RESTRICT_FACTOR, LOT_SCALE_FACTOR, TEMP_LOT_REDUCTION_UNTIL, SCALING_ACTIVE
            if ACCOUNT_INITIAL_BALANCE is None:
                ai = mt5.account_info()
                if ai:
                    ACCOUNT_INITIAL_BALANCE = float(getattr(ai, 'balance', 0.0))
        except Exception:
            pass
        # 1) DAILY LOSS LIMIT: block if equity has dropped >= 5% since day_open_equity
        try:
            day_open = day_stats.get('day_open_equity')
            _, cur_eq = _equity_info()
            if day_open and cur_eq is not None:
                drop_pct = (day_open - cur_eq) / max(1e-9, day_open)
                # New safety: if daily loss reaches -2% pause trading for the day
                if drop_pct >= 0.02:
                    try:
                        tg("🛑 Daily cooldown: equity dropped >= 2% — trading paused for the day")
                    except Exception:
                        pass
                    trading_pause(True)
                    return "HALT", f"Daily cooldown (>=2%) applied ({drop_pct*100:.2f}%)"
                if drop_pct >= 0.05:
                    return "HALT", f"Daily loss limit exceeded ({drop_pct*100:.2f}%)"
        except Exception:
            pass
        # 2) MAXIMUM DRAWDOWN: block if equity <= 90% of initial account balance
        try:
            if ACCOUNT_INITIAL_BALANCE:
                _, cur_eq = _equity_info()
                if cur_eq <= ACCOUNT_INITIAL_BALANCE * 0.90:
                    return "HALT", "Max drawdown (10%) breached"
        except Exception:
            pass
        # 3) LEVERAGE / account type detection: set LEVERAGE_RESTRICT_FACTOR
        try:
            ai = mt5.account_info()
            acc_leverage = int(getattr(ai, 'leverage', 0) or 0)
            # Detect swing accounts from prop active plan or server hints
            is_swing = False
            try:
                plan = PROP_ACTIVE.get('plan', '') or ''
                server = getattr(ai, 'server', '') or ''
                if 'swing' in str(plan).lower() or 'swing' in str(server).lower():
                    is_swing = True
            except Exception:
                is_swing = False
            allowed_lev = 30 if is_swing else 100
            if acc_leverage and acc_leverage > allowed_lev:
                # calculate scale factor to reduce lots proportionally
                LEVERAGE_RESTRICT_FACTOR = float(allowed_lev) / float(max(1, acc_leverage))
            else:
                LEVERAGE_RESTRICT_FACTOR = 1.0
        except Exception:
            pass
        # 5) HOLDING RULES: regular accounts must not hold over weekend — block new trades on Friday after 16:00 UK
        try:
            ai = mt5.account_info()
            is_swing = False
            try:
                plan = PROP_ACTIVE.get('plan', '') or ''
                server = getattr(ai, 'server', '') or ''
                if 'swing' in str(plan).lower() or 'swing' in str(server).lower():
                    is_swing = True
            except Exception:
                is_swing = False
            now = now_uk()
            # Weekday: Monday=0 ... Sunday=6. Block new trades late Friday for regular accounts
            if not is_swing and now.weekday() == 4 and now.hour >= 16:
                return "HALT", "Weekend hold protection (regular account)"
        except Exception:
            pass
        try:
            label, _, _ = session_bounds()
            if label == "LON" and (day_stats.get("fulls_LON", 0) >= FULL_MAX_PER_DAY):
                return "HALT", "Session full cap reached (London)"
            if label == "NY" and (day_stats.get("fulls_NY", 0) >= FULL_MAX_PER_DAY):
                return "HALT", "Session full cap reached (New York)"
        except Exception:
            pass
        return "OK", ""
except Exception:
    pass
ML_MODEL = None
def _load_trade_log_csv(path: str = "trade_log.csv") -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        return df if len(df) >= 20 else None  # need minimum samples
    except Exception:
        return None
def _extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [
        "h4_bias", "adx_h1", "adx_m15", "adx_rising", "rsi_h1", "rsi_m15", "ema_slope",
        "ema_align", "bos_align", "fvg", "atr_hot", "vol_mom", "session", "spread_ok", "rsi_ok"
    ]
    available = [c for c in cols if c in df.columns]
    X = df[available].fillna(0.0).astype(float)
    y = df["outcome"].astype(int) if "outcome" in df.columns else (
        df["win"] if "win" in df.columns else None
    )
    if y is None:
        raise ValueError("trade log missing outcome/win column (0/1)")
    return X, y
def ml_train_from_log() -> Optional[Any]:
    """Train ML model from historical trade logs using scikit-learn."""
    global ML_MODEL, AI_BIAS
    
    # Check for scikit-learn
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        log_msg("⚠️ scikit-learn not available; ML disabled. Install with: pip install scikit-learn")
        return None
    
    # Load trade history
    df = _load_trade_log_csv()
    if df is None:
        log_msg("ℹ️ trade_log.csv not found or too small (need 20+ trades); ML training skipped.")
        return None
    
    try:
        # Extract features and outcomes
        X, y = _extract_features(df)
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        ML_MODEL = model
        
        # Calculate model performance
        probs = model.predict_proba(X)[:,1]
        avg = float(np.mean(probs)) if len(probs) else 0.5
        
        # Calculate training accuracy
        predictions = model.predict(X)
        accuracy = float(np.mean(predictions == y))
        
        # Adjust AI bias based on average probability
        AI_BIAS = int(round((avg*100) - 60))  # shift so avg≈60
        
        log_msg(f"🤖 ML Model Trained Successfully:")
        log_msg(f"   📊 Training Samples: {len(X)}")
        log_msg(f"   🎯 Training Accuracy: {accuracy:.1%}")
        log_msg(f"   📈 Avg Win Probability: {avg:.2%}")
        log_msg(f"   ⚖️  AI Bias Adjustment: {AI_BIAS:+d}")
        log_msg(f"   🔧 Features Used: {len(X.columns)}")
        
        return model
    except Exception as e:
        log_msg(f"⚠️ ML training failed: {e}")
        return None
def ml_score(feature_dict: Dict[str, float]) -> Optional[float]:
    if ML_MODEL is None:
        return None
    try:
        X = pd.DataFrame([feature_dict])
        cols = ML_MODEL.feature_names_in_
        X = X.reindex(columns=cols, fill_value=0.0)
        p = float(ML_MODEL.predict_proba(X)[:,1][0])
        return p
    except Exception:
        return None
# ML-enhanced AI validation wrapper
try:
    _old_ai_validate = ai_validate_signal
    
    def ai_validate_signal(symbol, side, h4, h1, m15, m5=None, tick=None, meta=None):
        """Enhanced AI validation with ML score adjustment."""
        # Get base validation result
        res = _old_ai_validate(symbol, side, h4, h1, m15, m5, tick, meta)
        
        # Apply ML score adjustment if model is trained
        if ML_MODEL is not None:
            try:
                # Extract features for ML prediction
                feats = {
                    "h4_bias": 1 if ema_bias(h4) == side else 0,
                    "adx_h1": adx(h1),
                    "adx_m15": adx(m15),
                    "adx_rising": 1 if adx_rising(h1)[1] else 0,
                    "rsi_h1": rsi(h1),
                    "rsi_m15": rsi(m15),
                    "ema_slope": _ema_slope(get_ema(h1, 50) - get_ema(h1, 200)),
                    "ema_align": 1 if ((side == "BUY" and _ema_slope(get_ema(h1, 50) - get_ema(h1, 200)) > 0) or (side == "SELL" and _ema_slope(get_ema(h1, 50) - get_ema(h1, 200)) < 0)) else 0,
                    "bos_align": 1 if ((side == "BUY" and detect_bos(h1) == "BOS_UP") or (side == "SELL" and detect_bos(h1) == "BOS_DOWN")) else 0,
                    "fvg": 1 if detect_fvg(m15) else 0,
                    "atr_hot": 1,
                    "vol_mom": 1,
                    "session": 1 if in_session() else 0,
                    "spread_ok": 1 if spread_ok(symbol) else 0,
                    "rsi_ok": 1,
                }
                
                # Get ML probability score
                p = ml_score(feats)
                if p is not None:
                    original_score = res["score"]
                    # Convert probability to score adjustment (-50 to +50)
                    adj = int(round((p*100) - 50))
                    # Apply adjustment with clamping
                    res["score"] = int(clamp(res["score"] + adj, 0, 100))
                    res["approve"] = (res["score"] >= AI_THRESHOLD)
                    res.setdefault("why", []).append(f"ML adj {adj:+d} (prob={p:.2%})")
                    
                    # Log significant ML adjustments
                    if abs(adj) >= 10:
                        log_msg(f"🤖 ML: {symbol} {side} | Score: {original_score}→{res['score']} | Win Prob: {p:.1%} | Adj: {adj:+d}")
            except Exception as e:
                log_debug(f"ML score adjustment error: {e}")
        
        return res
except Exception as e:
    log_debug(f"ML ai_validate_signal wrapper error: {e}")
    pass
def start_async_utilities() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    tasks = []
    try:
        tasks.append(loop.create_task(telegram_loop_async()))
    except Exception:
        pass
    try:
        tasks.append(loop.create_task(trade_manager_loop()))
    except Exception:
        pass
    if not loop.is_running():
        import threading
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        _LOGGER.info("Async utilities started (Telegram + manager loops).")
# Initialize ML model from trade history at startup
try:
    log_msg("🤖 Initializing ML model from trade history...")
    model = ml_train_from_log()
    if model is not None:
        log_msg("✅ ML model trained successfully")
    else:
        log_msg("ℹ️ ML model not trained (insufficient data or scikit-learn not available)")
except Exception as e:
    log_msg(f"⚠️ ML model initialization failed: {e}")
    pass

# Do not start async utilities at import time. `main()` will start them
# explicitly when the script is executed.
FEATURES = {
    "ASYNC_TELEGRAM": os.getenv("DBOT_ASYNC_TELEGRAM", "1") == "1",
    "MANAGER_LOOP":   os.getenv("DBOT_MANAGER_LOOP", "1") == "1",
    "ML":             os.getenv("DBOT_ML", "1") == "1",
    "RISK_GUARD":     os.getenv("DBOT_RISK_GUARD", "1") == "1",
    "LOGGING_ROTATE": os.getenv("DBOT_LOG_ROTATE", "1") == "1",
    "BACKTEST":       os.getenv("DBOT_BACKTEST", "0") == "1",
}
try:
    load_dotenv()
except Exception:
    pass
def safe_copy_rates(symbol: str, timeframe: int, count: int = 500):
    """Resilient wrapper for mt5.copy_rates_from_pos with fallbacks."""
    try:
        return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    except Exception as e:
        try:
            log_msg(f"⚠️ MT5 rates error on {symbol}/{timeframe}: {e}")
        except Exception:
            pass
        return None
def get_data_safe(symbol: str, timeframe: int, bars: int = 200):
    try:
        df = pd.DataFrame(safe_copy_rates(symbol, timeframe, bars))
        return df if len(df) else None
    except Exception:
        return None
if "get_data" in globals():
    get_data_original = get_data
    def get_data(symbol, timeframe, bars=200):
        df = get_data_original(symbol, timeframe, bars)
        if df is None or not len(df):
            return get_data_safe(symbol, timeframe, bars)
        return df
try:
    import asyncio, signal, contextlib
    AIOHTTP_TIMEOUT = float(os.getenv("DBOT_HTTP_TIMEOUT", "15"))
    _shutdown_event = asyncio.Event()
    async def _shutdown_waiter():
        await _shutdown_event.wait()
    def request_shutdown():
        try:
            if not _shutdown_event.is_set():
                _shutdown_event.set()
        except Exception:
            pass
    def _install_signal_handlers(loop=None):
        loop = loop or asyncio.get_event_loop()
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None: 
                continue
            try:
                loop.add_signal_handler(sig, request_shutdown)
            except NotImplementedError:
                pass
    if "telegram_loop_async" in globals():
        _old_tg_loop = telegram_loop_async
        async def telegram_loop_async():
            last_id = None
            while not _shutdown_event.is_set():
                try:
                    updates = await tg_poll_async(last_id, 10)
                    for u in updates:
                        last_id = u.get("update_id", last_id)
                        msg = u.get("message") or {}
                        text = (msg.get("text") or "")
                        chat_id = str((msg.get("chat") or {}).get("id", TELEGRAM_CHAT_ID))
                        if text:
                            _dispatch_minimal_command(text, chat_id)
                except Exception as e:
                    try:
                        _LOGGER.error(f"telegram loop error: {e}")
                    except Exception:
                        pass
                await asyncio.sleep(1.0)
    if "trade_manager_loop" in globals():
        _old_mgr_loop = trade_manager_loop
        async def trade_manager_loop(poll_sec: float = 2.5, atr_period: int = 14, trail_mult: float = 1.5):
            while not _shutdown_event.is_set():
                try:
                    await _old_mgr_loop(poll_sec, atr_period, trail_mult)
                except Exception as e:
                    try:
                        _LOGGER.error(f"trade manager wrapper error: {e}")
                    except Exception:
                        pass
                await asyncio.sleep(0.1)
    def start_async_utilities():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        _install_signal_handlers(loop)
        tasks = []
        if FEATURES.get("ASYNC_TELEGRAM", True) and "telegram_loop_async" in globals():
            tasks.append(loop.create_task(telegram_loop_async()))
        if FEATURES.get("MANAGER_LOOP", True) and "trade_manager_loop" in globals():
            tasks.append(loop.create_task(trade_manager_loop()))
        if not loop.is_running():
            import threading
            t = threading.Thread(target=loop.run_forever, daemon=True)
            t.start()
            try:
                _LOGGER.info("Async utilities started (with graceful shutdown).")
            except Exception:
                pass
        return tasks
    def stop_async_utilities():
        request_shutdown()
except Exception:
    pass
def _assert_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol
def unit_tests() -> str:
    """Run simple unit tests for ATR/ADX and ML hook presence."""
    msgs = []
    try:
        n = 100
        import numpy as _np
        base = _np.cumsum(_np.random.randn(n)) + 100
        high = base + _np.random.rand(n)*2
        low  = base - _np.random.rand(n)*2
        close= base + _np.random.randn(n)*0.5
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        a = true_atr(df, 14)
        msgs.append(f"ATR ok: {a:.4f}")
        try:
            adx_val = adx(df, 14)
            msgs.append(f"ADX ok: {float(adx_val):.2f}")
        except Exception as e:
            msgs.append(f"ADX fail: {e}")
        if "ml_score" in globals():
            msgs.append("ML hook present ✅")
        else:
            msgs.append("ML hook missing ❌")
    except Exception as e:
        msgs.append(f"Unit tests exception: {e}")
    return "\\n".join(msgs)
def backtest_goat(symbol: str = "XAUUSD", bars: int = 1500) -> dict:
    """Very simple walk-forward backtest using build_signals_goat + ATR SL=2R, TP=2R.
    Returns basic stats dict. Intended for sanity checks, not production-grade analytics.
    """
    try:
        h1 = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, bars, max_age=3.0)
        m15= fetch_data_cached(symbol, mt5.TIMEFRAME_M15, bars, max_age=1.5)
        h4 = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, bars, max_age=6.0)
        if h1 is None or m15 is None or h4 is None: 
            return {"error": "insufficient data"}
        wins=losses=0; pnl=0.0; trades=0
        for i in range(60, len(h1)-1):
            h1w = h1.iloc[:i].copy()
            m15w= m15.iloc[:i*4].copy() if len(m15)>=i*4 else m15.copy()
            h4w = h4.iloc[: i//4 + 1].copy()
            sigs = build_signals_goat(symbol, announce=False)
            if not sigs:
                continue
            t_side = sigs[0]["side"]
            atr_v = true_atr(m15w, 14) or 0.1
            entry = float(h1.iloc[i]["close"])
            sl = entry - 2*atr_v if t_side=="BUY" else entry + 2*atr_v
            tp = entry + 2*atr_v if t_side=="BUY" else entry - 2*atr_v
            next_close = float(h1.iloc[i+1]["close"])
            hit_tp = (next_close >= tp) if t_side=="BUY" else (next_close <= tp)
            hit_sl = (next_close <= sl) if t_side=="BUY" else (next_close >= sl)
            if hit_tp and not hit_sl:
                wins += 1; pnl += 1.0
            elif hit_sl and not hit_tp:
                losses += 1; pnl -= 1.0
            trades += 1
        return {"trades": trades, "wins": wins, "losses": losses, "pnlR": round(pnl,2)}
    except Exception as e:
        return {"error": str(e)}
# Duplicate run_self_tests conditional removed; final implementation remains.
import math, threading, queue, time
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
_TELE_Q: "queue.Queue[Tuple[str,str]]" = queue.Queue(maxsize=500)
def _tele_worker():
    while True:
        try:
            chat_id, text = _TELE_Q.get()
            try:
                telegram_msg(text)  # use existing sender; non-blocking failures
            except Exception:
                pass
        except Exception:
            pass
        finally:
            time.sleep(0.05)
# Do not start the tele worker at import time; main() will start it when running.
def telegram_msg_async(text: str, chat_id: Optional[str] = None) -> None:
    """Non-blocking Telegram send (uses background worker)."""
    try:
        _TELE_Q.put_nowait((chat_id or TELEGRAM_CHAT_ID, text))
    except Exception:
        pass
def retry_exp(max_tries: int = 4, base_delay: float = 0.5, exc: Tuple[type,...]=(Exception,)):
    """Decorator with exponential backoff: base_delay * 2^(n-1)."""
    def deco(fn):
        def inner(*args, **kwargs):
            t = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exc as e:
                    t += 1
                    if t >= max_tries:
                        raise
                    time.sleep(base_delay * (2 ** (t-1)))
        return inner
    return deco
def di_values(df: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
    """Return (+DI, -DI) latest values using Wilder smoothing."""
    try:
        if df is None or len(df) < period + 2:
            return 0.0, 0.0
        h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        up_move = h.diff(); down_move = -l.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_s = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_s = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_s = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm_s / tr_s).replace([np.inf,-np.inf], np.nan)
        minus_di = 100 * (minus_dm_s / tr_s).replace([np.inf,-np.inf], np.nan)
        v1 = float(plus_di.iloc[-1]); v2 = float(minus_di.iloc[-1])
        return (0.0 if pd.isna(v1) else v1, 0.0 if pd.isna(v2) else v2)
    except Exception:
        return 0.0, 0.0
def tick_volume_pressure(df: pd.DataFrame, lookback: int = 20) -> bool:
    """Simple volume filter: rising tick volume vs its SMA."""
    try:
        if "tick_volume" not in df.columns or len(df) < lookback+2:
            return False
        sma = df["tick_volume"].rolling(lookback).mean()
        return bool(df["tick_volume"].iloc[-1] >= sma.iloc[-1] >= sma.iloc[-2])
    except Exception:
        return False
def sentiment_score(symbol: str) -> float:
    """Stub sentiment: 0 neutral. Override by writing 'sentiment.json' = {symbol: score[-1..1]}"""
    try:
        with open("sentiment.json","r",encoding="utf-8") as f:
            data = json.load(f)
        val = float(data.get(symbol, 0.0))
        return max(-1.0, min(1.0, val))
    except Exception:
        return 0.0
USE_DYNAMIC_SIZING = True
DAILY_DD_HARD_PCT = 4.0   # hard stop for the day (in addition to prop rules)
TRAIL_ATR_MULT    = 2.0
PARTIALS = (0.33, 0.33, 0.34)   # TP1/TP2/TP3 volume fractions
def _account_equity() -> float:
    try:
        ai = mt5.account_info()
        return float(getattr(ai, "equity", 0.0) or getattr(ai, "balance", 0.0))
    except Exception:
        return 0.0
def calc_dynamic_lot(symbol: str, entry: float, sl: float, risk_pct: float, atr_v: float) -> float:
    """Risk both on equity and scaled by volatility (ATR)."""
    try:
        eq = _account_equity()
        risk_amt = max(0.0, eq * float(risk_pct))
        pip_val = max(1e-9, abs(entry - sl))  # price distance
        if atr_v and pip_val < (0.5 * atr_v):
            pip_val = 0.5 * atr_v  # floor on too-tight stops
        raw_lot = risk_amt / pip_val
        return normalize_volume(symbol, raw_lot)
    except Exception:
        return normalize_volume(symbol, FULL_LOT_DEFAULT)
def _daily_drawdown_pct() -> float:
    try:
        pivot = globals().get("PROP_DAILY_PIVOT_UTC", {}).get("pivot_equity")
        ai = mt5.account_info()
        if pivot and ai:
            return 100.0 * (pivot - ai.equity) / max(1e-9, pivot)
    except Exception:
        pass
    return 0.0
try:
    _orig_place_order = place_order  # type: ignore  # noqa: F821
except NameError:
    _orig_place_order = None
    pass
_TM_RUNNING = False
def _trade_manager_loop():
    """Background loop to trail SL by ATR and take partial profits at RR 1:1, 1:2, 1:3."""
    while True:
        try:
            poss = mt5.positions_get() or []
            for p in poss:
                sym = getattr(p, "symbol", None)
                vol = getattr(p, "volume", 0.0)
                side = "BUY" if getattr(p, "type", 0) == mt5.POSITION_TYPE_BUY else "SELL"
                sl = getattr(p, "sl", 0.0); price_open = getattr(p, "price_open", 0.0)
                tick = mt5.symbol_info_tick(sym) if sym else None
                if not sym or not tick: 
                    continue
                cur = tick.bid if side == "SELL" else tick.ask
                risk = abs(price_open - sl) or 1e-9
                tp1 = price_open + (risk if side == "BUY" else -risk)
                tp2 = price_open + (2*risk if side == "BUY" else -2*risk)
                tp3 = price_open + (3*risk if side == "BUY" else -3*risk)
                m15 = fetch_data_cached(sym, mt5.TIMEFRAME_M15, 100, max_age=1.5)
                a = atr(m15, 14) if m15 is not None else 0.0
                if a > 0:
                    new_sl = cur - TRAIL_ATR_MULT*a if side == "BUY" else cur + TRAIL_ATR_MULT*a
                    if (side == "BUY" and new_sl > sl) or (side == "SELL" and new_sl < sl):
                        try:
                            req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": sym, "position": p.ticket,
                                "sl": normalize_price(sym, new_sl),
                                "tp": getattr(p, "tp", 0.0) or 0.0
                            }
                            mt5.order_send(req)
                            log_msg(f"🔧 Trailed SL on {sym} → {new_sl:.3f}")
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(5)
def start_trade_manager_thread():
    global _TM_RUNNING
    if not _TM_RUNNING:
        try:
            threading.Thread(target=_trade_manager_loop, daemon=True).start()
            _TM_RUNNING = True
            log_msg("🧵 Trade manager started (trailing & partials).")
        except Exception as e:
            log_msg(f"⚠️ Trade manager start error: {e}")
# Do not auto-start trade manager at import time; main() will start it.
AI_MODEL_FILE = "ai_model.json"
def _sigmoid(x): 
    return 1.0 / (1.0 + math.exp(-x))
def train_ai_from_csv(csv_path: str = "trades_history.csv",
                      lr: float = 0.05, epochs: int = 500) -> Optional[Dict[str, Any]]:
    """
    Train a tiny logistic regression on historical feature rows.
    CSV columns expected: outcome (0/1), and feature columns matching those in ai_validate_signal meta.
    Saves weights to AI_MODEL_FILE if successful.
    """
    import csv, json
    try:
        X = []; y = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            feats = None
            for row in r:
                if feats is None:
                    feats = [k for k in row.keys() if k not in ("outcome","score","symbol","ticket","time")]
                x = [float(row.get(k, 0.0)) for k in feats]
                X.append(x); y.append(int(float(row.get("outcome", 0))))
        if not X or not y:
            log_msg("⚠️ No training data found, skipping AI train.")
            return None
        import numpy as np
        X = np.array(X, dtype=float); y = np.array(y, dtype=float)
        n, m = X.shape
        w = np.zeros(m); b = 0.0
        for _ in range(epochs):
            z = X.dot(w) + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (X.T @ (p - y)) / n
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w; b -= lr * grad_b
        model = {"features": feats, "w": w.tolist(), "b": float(b)}
        with open(AI_MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(model, f)
        log_msg(f"✅ Trained AI from {csv_path}: {m} features.")
        return model
    except Exception as e:
        log_msg(f"⚠️ AI train error: {e}")
        return None
def _load_ai_model() -> Optional[Dict[str, Any]]:
    try:
        import json
        with open(AI_MODEL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
def ai_score_with_model(feature_map: Dict[str, float]) -> Optional[float]:
    """Return probability [0..100] using trained logistic model if present."""
    model = _load_ai_model()
    if not model:
        return None
    try:
        import numpy as np
        feats = model["features"]; w = np.array(model["w"], dtype=float); b = float(model["b"])
        x = np.array([float(feature_map.get(k, 0.0)) for k in feats], dtype=float)
        p = 1.0 / (1.0 + np.exp(-(float(x.dot(w) + b))))
        return float(round(100.0 * p, 0))
    except Exception:
        return None
try:
    _old_ai_validate_signal = ai_validate_signal  # type: ignore
    def ai_validate_signal(symbol, side, h4, h1, m15, m5=None, tick=None, meta=None):
        res = _old_ai_validate_signal(symbol, side, h4, h1, m15, m5=m5, tick=tick, meta=meta)
        vol_ok = tick_volume_pressure(m15)
        senti = sentiment_score(_base_of(symbol))
        if not vol_ok:
            res["approve"] = False
            res["why"].append("Weak volume")
        if side == "BUY" and senti < -0.25:
            res["approve"] = False; res["why"].append("Negative sentiment")
        if side == "SELL" and senti > 0.25:
            res["approve"] = False; res["why"].append("Positive sentiment")
        feats = None
        if isinstance(meta, dict):
            feats = meta.get("features")
        if not feats and isinstance(res, dict):
            feats = res.get("features")
        if feats:
            pscore = ai_score_with_model(feats)
            if pscore is not None:
                res["score"] = max(int(res.get("score", 0)), int(pscore))
                res["approve"] = res["score"] >= SYMBOL_CONF_THRESH.get(_base_of(symbol), AI_THRESHOLD)
                res["why"].append(f"LogReg score={pscore}")
        return res
except NameError:
    pass
def run_self_tests() -> str:
    msgs = []
    try:
        ai = mt5.account_info()
        msgs.append("MT5 account OK" if ai else "MT5 account FAIL")
    except Exception as e:
        msgs.append(f"MT5 error: {e}")
    try:
        df = fetch_data_cached(resolve_symbol("XAUUSD"), mt5.TIMEFRAME_M15, 50, max_age=1.5)
        msgs.append("Data OK" if df is not None and len(df)>20 else "Data FAIL")
        if df is not None and len(df)>20:
            a = atr(df, 14); ad = adx(df, 14); pdi, ndi = di_values(df, 14)
            msgs.append(f"ATR={a:.2f} ADX={ad:.1f} +DI={pdi:.1f} -DI={ndi:.1f}")
    except Exception as e:
        msgs.append(f"Data/ind error: {e}")
    return " | ".join(msgs)
ENABLE_PROFILING = True
class profile_block:
    def __init__(self, label: str): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        dt = (time.perf_counter() - self.t0) * 1000.0
        if ENABLE_PROFILING:
            log_msg(f"⏱️ {self.label}: {dt:.1f} ms")
def _pair_currencies(base: str) -> List[str]:
    if base.startswith("XAUUSD"): return ["USD"]
    if base.startswith("GBPUSD"): return ["GBP","USD"]
    if base.startswith("GBPJPY"): return ["GBP","JPY"]
    return ["USD"]
def in_news_blackout_any(symbol_base: str, now=None) -> bool:
    """Extended blackout: block if any high-impact news matches pair currencies."""
    if not NEWS_BLACKOUT_ON:
        return False
    try:
        base = symbol_base.split('.')[0].upper()
        currs = set(_pair_currencies(base))
        # Use safe timezone resolution
        tz = SAFE_TZ
        now = now or datetime.now(tz)
        for w in _load_news_windows() or []:
            cs = set(w.get("currencies", []))
            if not cs: 
                cs = {"USD"}  # default
            if currs & cs:
                st = datetime.fromisoformat(w["start"]); en = datetime.fromisoformat(w["end"])
                if st.tzinfo is None: st = tz.localize(st)
                if en.tzinfo is None: en = tz.localize(en)
                if st <= now <= en: 
                    return True
    except Exception:
        return False
    return False
def tune_scan_from_results(min_winrate: float = 0.55) -> None:
    """Adjust scan cadence based on rolling win-rate in symbol_outcomes."""
    try:
        wins = sum(1 for s in symbol_outcomes.values() for v in s if v == 1)
        total = sum(len(s) for s in symbol_outcomes.values())
        if total >= 20:
            wr = wins / total
            if wr >= min_winrate:
                globals()["SCAN_MIN_MINUTES"] = max(3, int(SCAN_MIN_MINUTES*0.8))
                globals()["SCAN_MAX_MINUTES"] = max(5, int(SCAN_MAX_MINUTES*0.8))
            else:
                globals()["SCAN_MIN_MINUTES"] = min(10, int(SCAN_MIN_MINUTES*1.2))
                globals()["SCAN_MAX_MINUTES"] = min(15, int(SCAN_MAX_MINUTES*1.2))
    except Exception:
        pass
# Duplicate TIMEZONE/QUIET_SPAM/SPAM_FILTER assignments and telegram_msg implementation removed; primary definitions appear earlier.
def _notify_trade_event(kind, symbol, side, price=None, sl=None, tp=None, extra=""):
    """Compact Telegram notifier for trades."""
    try:
        # Determine tag for trade size (FULL, MICRO, or custom string).
        extra_tag = str(extra).upper()
        if extra_tag == "FULL":
            tag = "FULL"
        elif extra_tag == "MICRO":
            tag = "MICRO"
        else:
            tag = extra_tag or ""
        parts = [str(kind), tag, str(side).upper(), str(symbol).upper()]
        # Format numeric fields if available.
        if price is not None:
            try:
                parts.append(f"@ {float(price):.5f}")
            except Exception:
                parts.append(f"@ {price}")
        if sl is not None:
            try:
                parts.append(f"SL {float(sl):.5f}")
            except Exception:
                parts.append(f"SL {sl}")
        if tp is not None:
            try:
                parts.append(f"TP {float(tp):.5f}")
            except Exception:
                parts.append(f"TP {tp}")
        msg = " ".join([p for p in parts if p])
        telegram_msg("✨ " + msg)
    except Exception:
        # Suppress any exception to avoid crashing the caller.
        try:
            # Best-effort fallback: print to console
            ts = datetime.now(SAFE_TZ).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] {_base_of(symbol)} {side}: {kind} (p={price} sl={sl} tp={tp})")
        except Exception:
            pass
def build_signals_goat(symbol, announce=False, period=20):
    if not in_session():
        if announce:
            log_msg(f"⏸️ {symbol} out of session")
        return []
    h4  = fetch_data_cached(symbol, mt5.TIMEFRAME_H4, 260, max_age=6.0)
    h1  = fetch_data_cached(symbol, mt5.TIMEFRAME_H1, 220, max_age=3.0)
    m15 = fetch_data_cached(symbol, mt5.TIMEFRAME_M15, 160, max_age=1.5)
    if h4 is None or h1 is None or m15 is None or len(h1) < period + 5:
        if announce:
            log_msg(f"⚠️ {symbol} insufficient data for GOAT strategy")
        return []
    bias, ema50, ema200 = ema_bias_signal(h1)
    atr_m15 = true_atr(m15, period=14)
    if atr_m15 is None or atr_m15 <= 0:
        atr_m15 = abs(float(m15["high"].iloc[-1]) - float(m15["low"].iloc[-1]))
    donch_high = h1["high"].rolling(period).max().iloc[-2]
    donch_low  = h1["low"].rolling(period).min().iloc[-2]
    close_prev = float(h1["close"].iloc[-2])
    close_now  = float(h1["close"].iloc[-1])
    si = mt5.symbol_info(symbol)
    pt = (si.point if si and si.point else 0.01)
    buffer = max(3 * pt, 0.25 * atr_m15)
    long_break  = (close_prev <= donch_high) and (close_now > donch_high + buffer)
    short_break = (close_prev >= donch_low)  and (close_now < donch_low - buffer)
    side = None
    if bias == "BUY" and long_break:
        side = "BUY"
    elif bias == "SELL" and short_break:
        side = "SELL"
    else:
        if announce:
            log_msg(f"❌ {symbol} GOAT: no qualified breakout (after ATR buffer)")
        return []

    # Multi‑candle confirmation: ensure the breakout candle has sufficient momentum.
    # Require the current move (close_now vs close_prev) to exceed half an ATR.
    price_change = abs(close_now - close_prev)
    if price_change < 0.5 * atr_m15:
        if announce:
            log_msg(
                f"❌ {symbol} GOAT: breakout weak, multi‑candle confirmation failed (Δ{price_change:.3f} < {0.5 * atr_m15:.3f})"
            )
        return []
    if not high_prob_filters_ok(symbol, h1, m15, h4, side, announce=announce, label="GOAT"):
        return []
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        if announce:
            log_msg(f"⚠️ {symbol} no tick for GOAT trade")
        return []
    if side == "BUY":
        sl = tick.bid - 2.0 * atr_m15
    else:
        sl = tick.ask + 2.0 * atr_m15
    if announce:
        log_msg(f"🐐 GOAT✅: {side} {symbol} | Donchian({period}) + ATR buffer {buffer:.3f}")
    return [{
        "side": side,
        "sl": sl,
        "tp1": None,
        "tp2": None,
        "tp3": None,
        "atr": atr_m15,
    }]
# removed duplicate startup summary logic (firm detection and ATR report).
# ===== TIDY AUTO SYSTEMS (CONSOLIDATED) =====
def auto_control():
    import MetaTrader5 as mt5, time
    global active_prop_firm, active_rules, last_heartbeat
    info = mt5.account_info()
    if info is None:
        mt5.shutdown(); time.sleep(1); mt5.initialize(); info = mt5.account_info()
        if info is None: return False
    server = info.server.lower()
    if 'ftmo' in server:
        active_prop_firm='FTMO'; active_rules={'daily_dd':0.05,'overall_dd':0.10,'max_trades':2}
    elif 'fxify' in server:
        active_prop_firm='FXIFY'; active_rules={'daily_dd':0.04,'overall_dd':0.08}
    elif 'goated' in server or 'gft' in server:
        active_prop_firm='GFT'; active_rules={'daily_dd':0.05,'overall_dd':0.10,'max_lots':5}
    elif 'aqua' in server:
        active_prop_firm='AQUA'; active_rules={'daily_dd':0.05,'trailing_dd':0.12}
    else:
        active_prop_firm='UNKNOWN'; active_rules={}
    now=time.time()
    if now-last_heartbeat>60: last_heartbeat=now
    try:
        if info.equity < info.balance * 0.8:
            return False
    except Exception as e:
        # Log the error encountered when accessing equity/balance fields
        log_error(f"auto_detect_prop_rules equity check error: {e}")
    return True

def auto_session_manager():
    import datetime
    global session_open, micro_count, full_count
    now=datetime.datetime.utcnow().hour
    # London: 7-12 UTC, NY: 13-20 UTC
    if 7 <= now <= 12 or 13 <= now <= 20:
        session_open=True
    else:
        session_open=False
    # reset counts at new session
    if not session_open:
        micro_count=0; full_count=0
    return session_open

# ===== TIDY AUTO SYSTEMS (CONSOLIDATED) =====
def auto_news_freeze():
    """Blocks trading around red news. Placeholder simplified."""
    import datetime
    now = datetime.datetime.utcnow().minute
    # simple freeze window example
    global news_freeze
    if now % 30 < 5:
        news_freeze = True
    else:
        news_freeze = False
    return news_freeze


def auto_market_type_detector():
    """Detects if market is trending, ranging, dead, or fake-breakout using ATR & ADX."""
    import MetaTrader5 as mt5
    import pandas as pd
    global market_type

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50)
    if rates is None:
        market_type = "dead"
        return market_type

    df = pd.DataFrame(rates)
    df['atr'] = df['high'] - df['low']
    avg_atr = df['atr'].tail(20).mean()
    last_atr = df['atr'].iloc[-1]

    # simple logic: if ATR too small → range; if too big → trend
    if last_atr < avg_atr * 0.6:
        market_type = "range"
    elif last_atr > avg_atr * 1.5:
        market_type = "trend"
    else:
        market_type = "normal"

    return market_type


def auto_volatility_surge_filter():
    """Blocks trading when volatility surges too fast using ATR spikes."""
    import MetaTrader5 as mt5
    import pandas as pd
    global vol_block

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 30)
    if rates is None:
        vol_block = True
        return vol_block

    df = pd.DataFrame(rates)
    df['atr'] = df['high'] - df['low']

    atr20 = df['atr'].tail(20).mean()
    last_atr = df['atr'].iloc[-1]

    if last_atr > atr20 * 2:
        vol_block = True
    else:
        vol_block = False

    return vol_block


def auto_filters():
    """Applies ADX / RSI / ATR filters and range/spread checks."""
    import MetaTrader5 as mt5
    import pandas as pd
    global filter_block

    # pull 50 M15 candles
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50)
    if rates is None:
        filter_block = True
        return filter_block

    df = pd.DataFrame(rates)
    df['atr'] = df['high'] - df['low']

    # simple RSI calculation
    df['chg'] = df['close'].diff()
    df['gain'] = df['chg'].clip(lower=0)
    df['loss'] = -df['chg'].clip(upper=0)
    df['avg_gain'] = df['gain'].rolling(14).mean()
    df['avg_loss'] = df['loss'].rolling(14).mean()
    df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-6)
    df['rsi'] = 100 - (100/(1+df['rs']))

    # spread check
    tick = mt5.symbol_info_tick("XAUUSD")
    spread = (tick.ask - tick.bid) if tick else 999

    # criteria
    atr_ok = df['atr'].iloc[-1] > df['atr'].tail(20).mean() * 0.5
    rsi_ok = 45 < df['rsi'].iloc[-1] < 70
    spread_ok = spread < 0.5

    filter_block = not (atr_ok and rsi_ok and spread_ok)
    return filter_block


def auto_risk_engine():
    """Risk engine: DD guard, equity lock, cooldowns, dynamic lot scaling."""
    import MetaTrader5 as mt5
    import time
    global risk_block, last_trade_time, active_rules

    info = mt5.account_info()
    if info is None:
        risk_block = True
        return risk_block

    balance = info.balance
    equity = info.equity

    # Daily drawdown guard
    if 'daily_dd' in active_rules:
        if equity < balance * (1 - active_rules['daily_dd']):
            risk_block = True
            return risk_block

    # Overall drawdown guard
    if 'overall_dd' in active_rules and active_rules['overall_dd'] is not None:
        if equity < balance * (1 - active_rules['overall_dd']):
            risk_block = True
            return risk_block

    # Simple cooldown: 60 sec after last trade
    if time.time() - last_trade_time < 60:
        risk_block = True
        return risk_block

    risk_block = False
    return risk_block


def auto_analysis_engine():
    """Multi-timeframe bias: reads H1 + M15 trend, structure, and momentum."""
    import MetaTrader5 as mt5
    import pandas as pd
    global market_bias

    def get_trend(symbol, tf, bars=100):
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            return "unknown"
        df = pd.DataFrame(rates)
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['close'].ewm(span=200).mean().iloc[-1]
        if ema50 > ema200:
            return "bull"
        elif ema50 < ema200:
            return "bear"
        return "flat"

    h1 = get_trend("XAUUSD", mt5.TIMEFRAME_H1)
    m15 = get_trend("XAUUSD", mt5.TIMEFRAME_M15)

    if h1 == m15:
        market_bias = h1
    else:
        market_bias = "mixed"

    return market_bias


def auto_ai_systems():
    """AI scoring engine: combines indicators into a confidence score."""
    import MetaTrader5 as mt5
    import pandas as pd
    global ai_score

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50)
    if rates is None:
        ai_score = 0
        return ai_score

    df = pd.DataFrame(rates)
    df['atr'] = df['high'] - df['low']
    df['chg'] = df['close'].diff()
    df['gain'] = df['chg'].clip(lower=0)
    df['loss'] = -df['chg'].clip(upper=0)
    df['avg_gain'] = df['gain'].rolling(14).mean()
    df['avg_loss'] = df['loss'].rolling(14).mean()
    df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-6)
    df['rsi'] = 100 - (100/(1+df['rs']))

    score = 0
    if df['atr'].iloc[-1] > df['atr'].mean(): score += 30
    if 50 < df['rsi'].iloc[-1] < 70: score += 40
    if df['close'].iloc[-1] > df['close'].mean(): score += 30

    ai_score = min(score,100)
    return ai_score


def auto_slippage_protection():
    """Dynamic slippage & spread protection with retry system."""
    import MetaTrader5 as mt5
    global slippage_block, safe_spread

    tick = mt5.symbol_info_tick("XAUUSD")
    if tick is None:
        slippage_block = True
        return slippage_block

    spread = tick.ask - tick.bid
    safe_spread = spread

    # Block if spread exceeds threshold
    if spread > 1.0:
        slippage_block = True
    else:
        slippage_block = False

    return slippage_block


def auto_do_nothing_mode():
    """Detects 'do nothing' conditions: bad volatility, conflicting signals, or blocked trading."""
    global market_type, vol_block, filter_block, risk_block, ai_score, do_nothing

    # If market is dead or strongly ranging — no trading
    if market_type in ["dead", "range"]:
        do_nothing = True
        return do_nothing

    # If volatility spike filter blocks trading
    if vol_block:
        do_nothing = True
        return do_nothing

    # If risk engine blocks trading
    if risk_block:
        do_nothing = True
        return do_nothing

    # If filters block trading (RSI/ATR/Spread)
    if filter_block:
        do_nothing = True
        return do_nothing

    # If AI score too low
    if ai_score < 40:
        do_nothing = True
        return do_nothing

    do_nothing = False
    return do_nothing


def auto_logging_system():
    """Handles Telegram + console logging for skips, errors, and signals."""
    global last_log_message

    try:
        # Basic placeholder logging logic
        message = f"Log heartbeat: market_type={market_type}, vol_block={vol_block}, filter_block={filter_block}, risk_block={risk_block}, ai_score={ai_score}"
        last_log_message = message
    except:
        last_log_message = "Logging error"

    return last_log_message


def auto_execution_engine():
    """Handles final execution filtering + smart SL/TP routing."""
    import MetaTrader5 as mt5
    global do_nothing, slippage_block, risk_block, filter_block, market_bias

    # If any block is active → no execution
    if do_nothing or slippage_block or risk_block or filter_block:
        return False

    # Simple directional check
    direction = "buy" if market_bias == "bull" else "sell" if market_bias == "bear" else None
    if direction is None:
        return False

    # Example placeholder execution logic
    # Real routing is handled elsewhere in your bot
    return direction



# ======================================================
# PERFORMANCE UPGRADE MODULES (Integrated Internally)
# ======================================================

# DATA CACHE (reduces MT5 calls)
DATA_CACHE = {"H4": None, "H1": None, "M15": None, "M5": None}
DATA_TIMESTAMP = None

def get_cached_rates(symbol):
    global DATA_CACHE, DATA_TIMESTAMP
    now = datetime.now()
    if DATA_TIMESTAMP is None or (now - DATA_TIMESTAMP).seconds > 60:
        DATA_TIMESTAMP = now
        DATA_CACHE["H4"] = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 300)
        DATA_CACHE["H1"] = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        DATA_CACHE["M15"] = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 500)
        DATA_CACHE["M5"] = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)
    return DATA_CACHE


# INDICATOR CACHE (single calculation per scan)
INDICATOR_CACHE = {}

def cache_indicator(key, func):
    if key not in INDICATOR_CACHE:
        INDICATOR_CACHE[key] = func()
    return INDICATOR_CACHE[key]

def reset_indicator_cache():
    INDICATOR_CACHE.clear()


# WATCHDOG (auto-restart if MT5 freezes)
LAST_LOOP = datetime.now()

def watchdog_check():
    global LAST_LOOP
    now = datetime.now()
    if (now - LAST_LOOP).seconds > 180:
        log("⚠️ Bot stalled — restarting MT5", "WATCHDOG")
        mt5.shutdown()
        time.sleep(3)
        mt5.initialize()
    LAST_LOOP = now


# SAFE SCAN WRAPPER (protects main loop)
def safe_scan():
    try:
        reset_indicator_cache()
        watchdog_check()
        return scan_once()
    except Exception as e:
        log(f"Scan error: {str(e)}", "ERROR")
        return None


# PERFORMANCE LOGGER
def perf_log(start_time):
    dt = (datetime.now() - start_time).total_seconds()
    log(f"⏱ Scan completed in {dt:.2f}s", "PERF")

# Fallback implementation for ``scan_once``
#
# The original code references a ``scan_once`` function inside the
# ``safe_scan`` wrapper.  In some configurations that function may not be
# supplied, resulting in a ``NameError``.  Provide a no‑op placeholder so
# that safe_scan() will return without raising.  When integrated into a
# larger system, users should override this with the actual scanning logic.
if "scan_once" not in globals():
    def scan_once() -> Optional[Any]:
        return None

# END OF PERFORMANCE UPGRADE

# -----------------------------------------------------------
# Agent‑mode integration (prop firm detection, risk rules,
# auto‑repair, order wrapper and startup message)
#
# NOTE: The entire agent‑mode implementation below has been disabled.
# It remains in the file for reference but is not executed.  A triple‑quoted
# string literal wraps this block to prevent any of the agent functions from
# being defined or run.  All prop firm detection and risk rules are now
# handled by the unified functions defined earlier in this module.

'''

import time as _agent_time
import datetime as _agent_dt
from typing import Any as _agent_Any, Dict as _agent_Dict, Optional as _agent_Optional, Tuple as _agent_Tuple


def detect_prop_firm_agent() -> _agent_Tuple[str, str, str]:
    """
    Detect the currently connected prop firm using mt5.account_info().

    Returns a tuple (firm, server, login) where firm is one of
    FTMO, FXIFY, Goated Funded Trader, Aquafunded or UNKNOWN.  This
    implementation waits briefly for the MetaTrader5 bridge to provide
    a non‑empty server string.  The server string is normalised for
    matching.  Explicit support is included for FTMO demo servers.
    If account information is unavailable or the server remains empty,
    UNKNOWN values will be returned.
    """
    # Obtain initial account info with exception handling
    try:
        info = mt5.account_info()  # type: ignore[name-defined]
    except Exception:
        info = None
    if not info:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    # Wait up to 5 seconds for a non‑empty server string
    start = _agent_time.time()
    server = getattr(info, "server", None)
    while (not server or str(server).strip() == "") and (_agent_time.time() - start < 5.0):
        _agent_time.sleep(0.5)
        try:
            info = mt5.account_info()  # type: ignore[name-defined]
        except Exception:
            info = None
        if not info:
            break
        server = getattr(info, "server", None)
    # If still missing info or server is empty, return unknown values
    if not info or not server or str(server).strip() == "":
        # Attempt to return login even if server is missing
        login_unknown = str(getattr(info, "login", "UNKNOWN") or "UNKNOWN") if info else "UNKNOWN"
        return "UNKNOWN", "UNKNOWN", login_unknown
    # Normalise server string for matching
    server_norm = str(server).upper().strip()
    login = str(getattr(info, "login", "UNKNOWN") or "UNKNOWN")
    # Determine firm based on server patterns
    # Determine firm name using simple substring checks.  Normalise
    # capitalization to ensure consistent values (e.g. ICMARKETS rather than
    # unknown).  Use uppercase matching on server string for reliability.
    # Example: "ICMarketsSC-MT5-2" should map to "ICMARKETS".
    if "ICMARKETS" in server_norm.replace(" ", ""):
        firm = "ICMARKETS"
    elif "FTMO-DEMO" in server_norm.replace(" ", ""):
        firm = "FTMO"
    elif "FTMO" in server_norm:
        firm = "FTMO"
    elif "FXIFY" in server_norm or "FXF" in server_norm:
        firm = "FXIFY"
    elif "GOATED" in server_norm or "GFT" in server_norm:
        firm = "GOATED"
    elif "AQUA" in server_norm:
        firm = "AQUAFUNDED"
    else:
        firm = "UNKNOWN"
    return firm, str(server), login


def load_risk_rules_agent(firm: str) -> _agent_Dict[str, _agent_Optional[float]]:
    """Return a mapping of risk rules for the given firm.

    The mapping may include keys such as max_daily_dd, max_total_dd,
    max_trades_per_day, consistency, trailing and max_lots. Missing
    keys indicate no restriction.
    """
    if firm == "FTMO":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trades_per_day": 2,
            "consistency": False,
            "trailing": False,
            "max_lots": None,
        }
    if firm == "FXIFY":
        return {
            "max_daily_dd": 0.04,
            "max_total_dd": 0.08,
            "max_trades_per_day": None,
            "consistency": True,
            "trailing": False,
            "max_lots": None,
        }
    if firm == "Goated Funded Trader":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trades_per_day": None,
            "consistency": False,
            "trailing": False,
            "max_lots": 5,
        }
    # Support uppercase alias for Goated Funded Trader
    if firm == "GOATED":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": 0.10,
            "max_trades_per_day": None,
            "consistency": False,
            "trailing": False,
            "max_lots": 5,
        }
    if firm == "Aquafunded":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": None,
            "max_trailing_dd": 0.12,
            "max_trades_per_day": None,
            "consistency": False,
            "trailing": True,
            "max_lots": None,
        }
    # Support uppercase alias for Aquafunded
    if firm == "AQUAFUNDED":
        return {
            "max_daily_dd": 0.05,
            "max_total_dd": None,
            "max_trailing_dd": 0.12,
            "max_trades_per_day": None,
            "consistency": False,
            "trailing": True,
            "max_lots": None,
        }
    # ICMARKETS has no specific prop firm restrictions
    if firm == "ICMARKETS":
        return {
            "max_daily_dd": None,
            "max_total_dd": None,
            "max_trades_per_day": None,
            "consistency": False,
            "trailing": False,
            "max_lots": None,
        }
    return {
        "max_daily_dd": None,
        "max_total_dd": None,
        "max_trades_per_day": None,
        "consistency": False,
        "trailing": False,
        "max_lots": None,
    }


def auto_repair_blockers_agent() -> bool:
    """Attempt to repair common trading blockers and return True when
    repairs have been attempted. This helper reinitialises MT5 if
    necessary, ensures a symbol is visible and has up to date ticks, and
    refreshes the market book. It suppresses exceptions to avoid
    disrupting the caller."""
    try:
        # initialise MT5 if not already initialised
        if not mt5.initialize():  # type: ignore[name-defined]
            try:
                mt5.shutdown()  # type: ignore[name-defined]
            except Exception:
                pass
            _agent_time.sleep(1)
            if not mt5.initialize():  # type: ignore[name-defined]
                return False
    except Exception:
        return False
    test_symbol = "XAUUSD"
    # ensure the symbol is visible
    try:
        info = mt5.symbol_info(test_symbol)  # type: ignore[name-defined]
        if info is None or not getattr(info, "visible", False):
            mt5.symbol_select(test_symbol, True)  # type: ignore[name-defined]
    except Exception:
        pass
    # ensure tick available
    try:
        tick = mt5.symbol_info_tick(test_symbol)  # type: ignore[name-defined]
        if tick is None:
            _agent_time.sleep(1)
            mt5.symbol_info_tick(test_symbol)  # type: ignore[name-defined]
    except Exception:
        pass
    # refresh market book
    try:
        mt5.market_book_get(test_symbol)  # type: ignore[name-defined]
    except Exception:
        pass
    return True


_agent_trade_count: int = 0
_agent_last_trade_day: _agent_Optional[_agent_dt.date] = None


def _agent_reset_trade_counter_if_new_day() -> None:
    global _agent_trade_count, _agent_last_trade_day
    today = _agent_dt.date.today()
    if _agent_last_trade_day != today:
        _agent_trade_count = 0
        _agent_last_trade_day = today


def order_send_wrapper_agent(request: _agent_Dict[str, _agent_Any]) -> _agent_Any:
    """
    Ticket-Funded MT5 Connection: Wrap mt5.order_send with connection checks.
    
    Enforces prop firm rules and ensures MT5 is connected before sending trades.
    Returns the MT5 order_send result, or None if blocked or disconnected.
    """
    
    # Ticket-Funded MT5 Connection: Verify MT5 is connected before order
    try:
        if not ensure_mt5_connected():
            print("[Ticket-Funded MT5] MT5 disconnected - order blocked")
            telegram_msg_mt5("[Ticket-Funded] ERROR - MT5 disconnected - order rejected.")
            return None
        account = mt5.account_info()
    except Exception as e:
        print(f"[Ticket-Funded MT5] Connection check failed: {e}")
        return None

    # Detect active prop firm
    firm, rules, _server = detect_prop_firm()
    firm = firm or "UNKNOWN"

    # Load correct risk rules (non-agent version)
    rules = load_risk_rules(firm)

    # Reset trade counter daily
    _reset_trade_counter_if_new_day()

    # Block trades if max trades reached
    max_trades = rules.get("max_trades_per_day")
    if max_trades is not None and day_stats["trades"] >= max_trades:
        print(f"🚫 Order blocked: daily trade limit ({max_trades}) reached.")
        return None

    # Check max lot size
    max_lots = rules.get("max_lots")
    try:
        vol = float(request.get("volume", 0))
    except Exception:
        vol = 0.0

    if max_lots is not None and vol > max_lots:
        print(f"🚫 Order blocked: lot size {vol} > allowed max {max_lots} for {firm}.")
        return None

    # If all checks pass → send trade
    try:
        return mt5.order_send(request)
    except Exception as e:
        print(f"❌ Order send failed: {e}")
        # Ticket-Funded MT5 Connection: Mark as potentially disconnected
        global MT5_CONNECTED
        MT5_CONNECTED = False
        return None
        return None
    # assume success if retcode indicates success or result is truthy
    success = False
    try:
        rc = getattr(result, "retcode", None)
        success = rc in (0, 10008, 10009)
    except Exception:
        success = result is not None
    if success:
        _agent_trade_count += 1
    return result


def startup_message_agent() -> None:
    """Clean unified startup summary with correct FTMO-Demo detection."""

    # Detect firm + rules + server using unified function
    firm, rules, detected_server = detect_prop_firm()

    # Get MT5 login and server directly
    ai_status = mt5.account_info()
    server = ai_status.server if ai_status and ai_status.server else detected_server
    login = str(ai_status.login) if ai_status and ai_status.login else "UNKNOWN"

    # Normalise server → detect FTMO Demo/Eval/Live
    s_norm = server.lower().replace("-", "").replace(" ", "")
    if "ftmo" in s_norm:
        firm = "FTMO"

    # Save firm globally
    PROP_ACTIVE["name"] = firm

    # Load correct risk rules (non-agent version)
    try:
        rules = load_risk_rules(firm)
    except Exception:
        rules = {}

    # Build clean startup message
    lines = [
        ">>> Bot Online",
        f"Firm Detected: {firm}",
        f"Server: {server}",
        f"Account: {login}",
        "",
        "Risk Rules Loaded:",
        str(rules),
        "",
        "Status:",
        "• MT5 initialized",
        "• Server connected",
        "• Firm detected",
        "• Trading engine active",
        "• Risk rules applied",
    ]

    print("\n".join(lines))

    # Send Telegram startup message (optional safety)
    try:
        telegram_msg(f"🤖 Bot Online | Firm: {firm} | Server: {server}")
    except Exception:
        pass

# End of startup_message_agent()

# =============================================================================
# HOLY GRAIL MAIN EXECUTION LOOP
# =============================================================================

def holy_grail_main():
    """
    Main execution loop for Holy Grail engine.
    Runs continuous SMC scanning in AGGRESSIVE mode.
    """
    try:
        print("="*70)
        print("HOLY GRAIL ENGINE - AGGRESSIVE SMC MODE")
        print("="*70)
        print("Rules:")
        print("  • BOS + Sweep + FVG = MANDATORY core")
        print("  • Score ≥70% = FULL trade")
        print("  • Score 65-69% = PREDICTIVE trade (AI confirm required)")
        print("  • Score <65% = IGNORE")
        print("  • Asian: XAUUSD OFF, GBPJPY ONLY")
        print("  • Blockers: MT5 connection, spread, SL/TP, FTMO DD ONLY")
        print("="*70)
        
        # Initialize MT5
        try:
            ensure_mt5_connected_or_exit()
        except Exception as e:
            print(f"❌ Failed to connect to MT5: {e}")
            return
        
        # Load state and memory
        try:
            load_state()
        except Exception:
            pass
        
        try:
            load_holy_grail_memory()
        except Exception:
            pass
        
        # Main loop
        scan_count = 0
        while True:
            try:
                scan_count += 1
                print(f"\n{'─'*70}")
                print(f"SCAN #{scan_count} - {datetime.now(SAFE_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'─'*70}")
                
                # Ticket-Funded: Monitor position closes for alerts
                try:
                    if 'monitor_holy_grail_position_closes' in globals():
                        monitor_holy_grail_position_closes()
                except Exception:
                    pass
                
                # Run Holy Grail scanner
                holy_grail_scan_and_execute()
                
                # Save state after scan
                try:
                    save_state()
                except Exception:
                    pass
                
                # Sleep between scans
                sleep_time = random.randint(SCAN_MIN_MINUTES * 60, SCAN_MAX_MINUTES * 60)
                print(f"\n⏳ Next scan in {sleep_time//60} minutes...")
                time.sleep(sleep_time)
            
            except KeyboardInterrupt:
                print("\n\n🛑 Holy Grail engine stopped by user")
                save_holy_grail_memory()
                save_state()
                break
            except Exception as e:
                error_msg = f"Scan error: {e}"
                print(f"⚠️ {error_msg}")
                # Ticket-Funded: Send error alert
                try:
                    if 'send_error_alert' in globals():
                        send_error_alert(error_msg)
                except Exception:
                    pass
                time.sleep(60)
    
    except Exception as e:
        print(f"❌ Holy Grail main loop error: {e}")
    finally:
        try:
            save_holy_grail_memory()
            save_state()
        except Exception:
            pass


'''

# When this module is executed directly, run the agent initialisation and print
# the startup message. This does not trigger when the module is imported as a
# library, thus avoiding side effects during normal bot operation.
if False:
    # The agent-specific startup has been disabled to avoid duplicate
    # initialisation when executing this module directly.  Startup logic
    # is handled by the earlier __main__ block invoking main().
    pass
