#!/usr/bin/env python3
"""
Octopus: Remote Signal Aggregator & Execution Engine.
Fetches signals from 'https://workspace-production-9fae.up.railway.app/predictions'
- REMOVED: HTML Scraping, Regex Parsing.
- ADDED: JSON API Integration.
- UPDATED: Sizing Formula (Equity * Leverage * Sum / TradedAssets).
- UPDATED: Execution Logic (Dynamic Remainder, 210s Duration, 10s Intervals).
- ADDED: Custom print function with 0.1s delay.
- UPDATED: Component-based Signal Calculation (Precise Timeframe Logic).
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

# --- Local Imports ---
try:
    from kraken_futures import KrakenFuturesApi
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}. Ensure 'kraken_futures.py' is in the directory.")
    sys.exit(1)

# --- Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Keys
KF_KEY = os.getenv("KRAKEN_FUTURES_KEY")
KF_SECRET = os.getenv("KRAKEN_FUTURES_SECRET")

# Global Settings
LEVERAGE = 0
SIGNAL_FEED_URL = "https://workspace-production-9fae.up.railway.app/predictions"

# Asset Mapping (Feed Symbol -> Kraken Futures Perpetual)
SYMBOL_MAP = {
    # --- Majors ---
    "BTCUSDT": "ff_xbtusd_260327",
    "ETHUSDT": "pf_ethusd",
    "SOLUSDT": "pf_solusd",
    "BNBUSDT": "pf_bnbusd",
    "XRPUSDT": "pf_xrpusd",
    "ADAUSDT": "pf_adausd",
    
    # --- Alts ---
    "DOGEUSDT": "pf_dogeusd",
    "AVAXUSDT": "pf_avaxusd",
    "DOTUSDT": "pf_dotusd",
    "LINKUSDT": "pf_linkusd",
    "TRXUSDT": "pf_trxusd",
    "BCHUSDT": "pf_bchusd",
    "XLMUSDT": "pf_xlmusd",
    "LTCUSDT": "pf_ltcusd",
    "SUIUSDT": "pf_suiusd",
    "HBARUSDT": "pf_hbarusd",
    "SHIBUSDT": "pf_shibusd", 
    "TONUSDT": "pf_tonusd",
    "UNIUSDT": "pf_uniusd",
    "ZECUSDT": "pf_zecusd",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    handlers=[logging.FileHandler("octopus.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Octopus")

# --- Signal Fetcher ---

class SignalFetcher:
    def __init__(self, url):
        self.url = url

    def fetch_signals(self) -> Tuple[Dict[str, int], int]:
        """
        Fetches JSON from the API and calculates the net vote based on candle closes.
        Logic:
        - 1d Close (00:00 UTC): Sum 5 (0-4)
        - 4h Close (04:00..): Sum 4 (0-3)
        - 1h Close (xx:00): Sum 3 (0-2)
        - 30m Close (xx:30): Sum 2 (0-1)
        - 15m Close (xx:15, xx:45): Sum 1 (0) -> Pure 15m signal
        """
        try:
            logger.info(f"Fetching signals from {self.url}...")
            resp = requests.get(self.url, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            asset_votes = {}
            traded_assets_count = len(data)

            # Determine Logic Slice based on Current Time
            now = datetime.now(timezone.utc)
            
            if now.minute == 0:
                if now.hour == 0:
                    # 1d Close: Include all 5 (0,1,2,3,4)
                    slice_limit = 5
                elif now.hour % 4 == 0:
                    # 4h Close: Include 4 (0,1,2,3)
                    slice_limit = 4
                else:
                    # 1h Close: Include 3 (0,1,2)
                    slice_limit = 3
            elif now.minute == 30:
                # 30m Close: Include 2 (0,1) -> 15m + 30m
                slice_limit = 2
            else:
                # 15m or 45m Close: Include 1 (0) -> 15m only
                slice_limit = 1
            
            logger.info(f"Time: {now.strftime('%H:%M')} UTC | Slice Limit: {slice_limit} (Components)")

            for asset_name, metrics in data.items():
                if asset_name not in SYMBOL_MAP:
                    continue
                
                # Get components list, default to empty
                comp = metrics.get("comp", [])
                
                if comp and isinstance(comp, list):
                    # Sum the relevant components based on the calculated slice_limit
                    # robustly handle cases where len(comp) < slice_limit
                    valid_slice = min(len(comp), slice_limit)
                    net_vote = sum(comp[:valid_slice])
                else:
                    # Fallback to pre-calculated sum if 'comp' is missing
                    net_vote = int(metrics.get("sum", 0))
                
                asset_votes[asset_name] = net_vote
            
            logger.info(f"Parsed {len(asset_votes)} active assets from a universe of {traded_assets_count}.")
            return asset_votes, traded_assets_count

        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return {}, 0

# --- Main Octopus Engine ---

class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.fetcher = SignalFetcher(SIGNAL_FEED_URL)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.instrument_specs = {}

    def _log(self, msg: str, level: str = "info"):
        """Custom print function with 0.1s delay."""
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        time.sleep(0.1)

    def initialize(self):
        self._log("Initializing Octopus (JSON API Mode)...")
        self._fetch_instrument_specs()
        
        # Connection Check
        self._log("Checking API Connection...")
        try:
            acc = self.kf.get_accounts()
            if "error" in acc:
                self._log(f"API Error: {acc}", level="error")
            else:
                self._log("API Connection Successful.")
        except Exception as e:
            self._log(f"API Connection Failed: {e}", level="error")

        self._log("Initialization Complete. Bot ready.")

    def _fetch_instrument_specs(self):
        try:
            url = "https://futures.kraken.com/derivatives/api/v3/instruments"
            resp = requests.get(url).json()
            if "instruments" in resp:
                for inst in resp["instruments"]:
                    sym = inst["symbol"].lower()
                    tick_size = float(inst.get("tickSize", 0.1))
                    precision = inst.get("contractValueTradePrecision")
                    size_step = 10 ** (-int(precision)) if precision is not None else 1.0
                    
                    self.instrument_specs[sym] = {
                        "sizeStep": size_step,
                        "tickSize": tick_size,
                        "contractSize": float(inst.get("contractSize", 1.0))
                    }
        except Exception as e:
            self._log(f"Error fetching specs: {e}", level="error")

    def _round_to_step(self, value: float, step: float) -> float:
        if step == 0: return value
        rounded = round(value / step) * step
        if isinstance(step, float) and "." in str(step):
            decimals = len(str(step).split(".")[1])
            rounded = round(rounded, decimals)
        elif isinstance(step, int) or step.is_integer():
            rounded = int(rounded)
        return rounded

    def _get_current_state(self, kf_symbol: str) -> Tuple[float, float]:
        """Helper to get current position size and mark price."""
        current_pos_size = 0.0
        mark_price = 0.0
        
        # 1. Get Position
        try:
            open_pos = self.kf.get_open_positions()
            if "openPositions" in open_pos:
                for p in open_pos["openPositions"]:
                    if p["symbol"].lower() == kf_symbol.lower():
                        size = float(p["size"])
                        if p["side"] == "short": size = -size
                        current_pos_size = size
                        break
        except Exception as e:
            self._log(f"[{kf_symbol}] Pos Fetch Error: {e}", level="error")
            
        # 2. Get Mark Price
        try:
            tickers = self.kf.get_tickers()
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
        except Exception as e:
            self._log(f"[{kf_symbol}] Ticker Fetch Error: {e}", level="error")
            
        return current_pos_size, mark_price

    def run(self):
        self._log("Bot started. Syncing with 15m intervals...")
        while True:
            now = datetime.now(timezone.utc)
            
            # Trigger every 15 minutes at second 30
            if now.minute % 15 == 0 and 30 <= now.second < 35:
                self._log(f"--- Trigger: {now.strftime('%H:%M:%S')} ---")
                self._process_signals()
                time.sleep(50) 
                
            time.sleep(1) 

    def _process_signals(self):
        # 1. Fetch Signals
        asset_votes, traded_assets_count = self.fetcher.fetch_signals()
        
        if traded_assets_count == 0:
            self._log("No assets found in feed. Skipping execution.", level="warning")
            return

        # 2. Get Account Equity
        try:
            acc = self.kf.get_accounts()
            if "flex" in acc.get("accounts", {}):
                equity = float(acc["accounts"]["flex"].get("marginEquity", 0))
            elif "accounts" in acc:
                first_acc = list(acc["accounts"].values())[0]
                equity = float(first_acc.get("marginEquity", 0))
            else:
                equity = 0
                
            if equity <= 0:
                self._log("Equity 0. Aborting.", level="error")
                return
        except Exception as e:
            self._log(f"Account fetch failed: {e}", level="error")
            return

        # 3. Calculate Unit Size
        unit_size_usd = (equity * LEVERAGE) / traded_assets_count
        self._log(f"Equity: ${equity:.2f} | Traded Assets: {traded_assets_count} | Unit Base: ${unit_size_usd:.2f}")

        # 4. Execute per Asset
        for asset, sum_val in asset_votes.items():
            target_usd = unit_size_usd * sum_val
            
            if sum_val != 0:
                self._log(f"[{asset}] Sum: {sum_val} -> Target Alloc: ${target_usd:.2f}")
            
            self.executor.submit(self._execute_single_asset_logic, asset, target_usd)

    def _execute_single_asset_logic(self, binance_asset: str, net_target_usd: float):
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol: return

        try:
            # Initial State Check to determine absolute target in contracts
            current_pos, mark_price = self._get_current_state(kf_symbol)
            if mark_price == 0:
                self._log(f"[{kf_symbol}] Mark price 0, skipping.", level="warning")
                return

            # Calculate Absolute Target in Contracts
            target_contracts = net_target_usd / mark_price
            
            self._log(f"[{kf_symbol}] Logic Start. Curr: {current_pos:.4f} -> Target: {target_contracts:.4f}")
            
            self._execute_smooth_order(kf_symbol, target_contracts)

        except Exception as e:
            self._log(f"[{kf_symbol}] Exec Error: {e}", level="error")

    def _execute_smooth_order(self, symbol: str, target_contracts: float):
        """
        Executes orders smoothly:
        1. Starts limit order 0.05% away.
        2. Updates 21 times (every 10s) = 210s total duration.
        3. CRITICAL: Re-calculates 'delta' (remaining needed) at every step.
        4. If not filled at end, sends Market Order for the EXACT remaining delta.
        """
        
        specs = self.instrument_specs.get(symbol.lower())
        size_inc = specs['sizeStep'] if specs else 0.001
        price_inc = specs['tickSize'] if specs else 0.01

        # Execution parameters
        # 21 steps * 10s = 210s total
        TOTAL_STEPS = 21
        STEP_INTERVAL = 10 
        
        START_OFFSET_PCT = 0.0005 # 0.05%
        DECAY_FACTOR = 0.90 # Move 10% closer each step
        
        order_id = None
        
        # Loop for smooth limit execution
        for i in range(TOTAL_STEPS):
            try:
                # 1. RE-CALCULATE REMAINDER (The "Remaining" Check)
                curr_pos, curr_mark = self._get_current_state(symbol)
                if curr_mark == 0: break

                delta = target_contracts - curr_pos
                
                # Check if we are "reasonably close" (within 1 size increment)
                if abs(delta) < size_inc:
                    self._log(f"[{symbol}] Target reached (Delta: {delta}). Stopping.")
                    if order_id: self.kf.cancel_order({"order_id": order_id, "symbol": symbol})
                    return

                # 2. Check Open Orders to detect fills
                # If order exists locally but not in API, it must have filled fully
                if order_id:
                    open_orders = self.kf.get_open_orders()
                    is_open = False
                    if "openOrders" in open_orders:
                        for o in open_orders["openOrders"]:
                            if o["order_id"] == order_id:
                                is_open = True
                                break
                    
                    if not is_open:
                        self._log(f"[{symbol}] Order {order_id} not found/filled. Recalculating.")
                        order_id = None
                        # Continue to next loop to re-measure 'delta' with new position
                        continue 

                # 3. Calculate Price & Size
                current_offset = START_OFFSET_PCT * (DECAY_FACTOR ** i)
                
                side = "buy" if delta > 0 else "sell"
                if side == "buy":
                    # Buy Limit below mark
                    limit_price = curr_mark * (1 - current_offset)
                else:
                    # Sell Limit above mark
                    limit_price = curr_mark * (1 + current_offset)

                limit_price = self._round_to_step(limit_price, price_inc)
                
                # STRICTLY USE REMAINDER AS SIZE
                order_size = self._round_to_step(abs(delta), size_inc)

                if order_size < size_inc:
                    continue

                # 4. Place or Edit Order
                if order_id is None:
                    # New Order
                    self._log(f"[{symbol}] Placing Limit {side.upper()} @ {limit_price} Size: {order_size} (Offset: {current_offset*100:.4f}%)")
                    resp = self.kf.send_order({
                        "orderType": "lmt",
                        "symbol": symbol,
                        "side": side,
                        "size": order_size,
                        "limitPrice": limit_price
                    })
                    if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                        order_id = resp["sendStatus"]["order_id"]
                    else:
                        self._log(f"[{symbol}] Limit Order Failed: {resp}", level="warning")
                else:
                    # Edit Order - Updates Size to match current remainder
                    self._log(f"[{symbol}] Updating Limit @ {limit_price} Size: {order_size} (Remaining)")
                    resp = self.kf.edit_order({
                        "orderId": order_id,
                        "limitPrice": limit_price,
                        "size": order_size,
                        "symbol": symbol
                    })
                    if "editStatus" in resp and "status" in resp["editStatus"]:
                         if resp["editStatus"]["status"] != "edited":
                             # Order likely filled or cancelled during edit
                             order_id = None

                # Wait between updates
                time.sleep(STEP_INTERVAL)

            except Exception as e:
                self._log(f"[{symbol}] Limit Loop Error: {e}", level="error")
                time.sleep(1)

        # --- End of Loop Cleanup ---
        if order_id:
            try:
                self.kf.cancel_order({"order_id": order_id, "symbol": symbol})
            except: pass

        # --- Final Market Fallback (Remainder Only) ---
        try:
            # RE-CALCULATE REMAINDER FINAL TIME
            curr_pos, _ = self._get_current_state(symbol)
            final_delta = target_contracts - curr_pos
            
            if abs(final_delta) >= size_inc:
                self._log(f"[{symbol}] Limit Loop Done (210s). MKT Execute Remaining: {final_delta:.4f}")
                side = "buy" if final_delta > 0 else "sell"
                final_size = self._round_to_step(abs(final_delta), size_inc)
                
                self.kf.send_order({
                    "orderType": "mkt",
                    "symbol": symbol,
                    "side": side,
                    "size": final_size
                })
            else:
                self._log(f"[{symbol}] Target Reached. No Market Order needed.")

        except Exception as e:
            self._log(f"[{symbol}] Market Fallback Error: {e}", level="error")

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()
