#!/usr/bin/env python3
"""
Octopus: NetSum Strategy (ETH Fixed Futures Edition)
Target: FF_ETHUSD_260130 | Source: Railway JSON
Logic: 1m Sync | 9-Step Dynamic Execution (8 LMT + 1 MKT)
Casing: Uppercase for Matching / Lowercase for Orders
"""

import os
import sys
import time
import logging
import requests
import threading
from datetime import datetime, timezone

# --- Local Imports ---
try:
    from kraken_futures import KrakenFuturesApi
except ImportError:
    print("CRITICAL: 'kraken_futures.py' not found.")
    sys.exit(1)

# --- Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KF_KEY = os.getenv("KRAKEN_FUTURES_KEY")
KF_SECRET = os.getenv("KRAKEN_FUTURES_SECRET")
DATA_URL = "https://web-production-73e1d.up.railway.app/"

# Strategy Constants (Strict Uppercase for Matching)
TARGET_SYMBOL = "FF_ETHUSD_260130"
LEVERAGE = 1.0
TIMEFRAME_MINUTES = 1
TRIGGER_OFFSET_SEC = 3
LOG_LOCK = threading.Lock()

# Execution Constants
MAX_STEPS = 8          # 8 Limit Order updates
STEP_INTERVAL = 5      # 5 seconds per step
INITIAL_OFFSET = 0.0002 # 0.02% (2 bps)
OFFSET_DECAY = 0.90    # Reduce offset by 10% per step

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("octopus_exec.log")]
)
logger = logging.getLogger("Octopus")

def octopus_log(msg, level="info"):
    with LOG_LOCK:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
        if level == "info": logger.info(msg)
        elif level == "error": logger.error(msg)
        elif level == "warning": logger.warning(msg)

class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.instrument_specs = {}

    def initialize(self):
        octopus_log("--- Initializing Octopus Engine (NetSum ETH) ---")
        self._fetch_instrument_specs()
        
        # Connection Test
        acc = self.kf.get_accounts()
        if "error" in acc: 
            octopus_log(f"API Error: {acc}", "error")
            sys.exit(1)
            
        if TARGET_SYMBOL not in self.instrument_specs:
            octopus_log(f"CRITICAL: {TARGET_SYMBOL} not found in specs.", "error")
            sys.exit(1)
        
        octopus_log("System Ready.")

    def _fetch_instrument_specs(self):
        """Fetches specs using UPPERCASE keys for storage."""
        try:
            url = "https://futures.kraken.com/derivatives/api/v3/instruments"
            resp = requests.get(url).json()
            for inst in resp.get("instruments", []):
                sym = inst["symbol"] # Keep Uppercase from API
                precision = inst.get("contractValueTradePrecision", 3)
                self.instrument_specs[sym] = {
                    "sizeStep": 10 ** (-int(precision)),
                    "tickSize": float(inst.get("tickSize", 0.1))
                }
        except Exception as e:
            octopus_log(f"Spec Fetch Error: {e}", "error")

    def _round_to_step(self, value, step):
        if step == 0: return value
        return round(round(value / step) * step, 8)

    def run(self):
        octopus_log(f"Bot Active. Sync: {TIMEFRAME_MINUTES}m @ :03s")
        while True:
            now = datetime.now(timezone.utc)
            if now.minute % TIMEFRAME_MINUTES == 0 and now.second == TRIGGER_OFFSET_SEC:
                octopus_log(f">>> TRIGGERED {now.strftime('%H:%M:%S')} <<<")
                self._process_cycle()
                time.sleep(2) # Prevent double trigger
            time.sleep(0.1)

    def _process_cycle(self):
        try:
            # 1. Fetch External Signal
            resp = requests.get(DATA_URL, timeout=5).json()
            net_sum = float(resp.get("netSum", 0))
            
            # 2. Fetch Equity
            acc = self.kf.get_accounts()
            equity = float(acc.get("accounts", {}).get("flex", {}).get("marginEquity", 0))
            
            if equity <= 0:
                octopus_log("Zero Equity. Skipping.", "warning")
                return

            octopus_log(f"Cycle Data | NetSum: {net_sum} | Equity: ${equity:.2f}")
            self._execute_dynamic_sequence(net_sum, equity)

        except Exception as e:
            octopus_log(f"Cycle Error: {e}", "error")

    def _get_current_state(self):
        """Helper to get current Mark Price and Position Size (Uppercase Matching)."""
        try:
            # Get Position
            pos_resp = self.kf.get_open_positions()
            curr_qty = 0.0
            for p in pos_resp.get("openPositions", []):
                # Strict Uppercase Match
                if p["symbol"] == TARGET_SYMBOL:
                    curr_qty = float(p["size"]) if p["side"] == "long" else -float(p["size"])
                    break
            
            # Get Mark Price
            tick_resp = self.kf.get_tickers()
            mark_price = 0.0
            for t in tick_resp.get("tickers", []):
                # Strict Uppercase Match
                if t["symbol"] == TARGET_SYMBOL:
                    mark_price = float(t["markPrice"])
                    break
            
            return curr_qty, mark_price
        except Exception as e:
            octopus_log(f"State Fetch Error: {e}", "error")
            return None, None

    def _execute_dynamic_sequence(self, net_sum, equity):
        """
        9-Step Execution: 8 Limit Updates + 1 Market Sweep.
        Strictly converts symbol to lowercase for API calls only.
        """
        specs = self.instrument_specs[TARGET_SYMBOL]
        order_id = None
        current_offset = INITIAL_OFFSET
        
        # Define lowercase symbol for execution calls
        exec_symbol = TARGET_SYMBOL.lower()

        # --- STEPS 1 to 8: Limit Order Updates ---
        for step in range(MAX_STEPS):
            # 1. Get Fresh State (Using Upper)
            curr_qty, mark_price = self._get_current_state()
            if mark_price is None or mark_price == 0:
                time.sleep(1)
                continue

            # 2. Recalculate Target Qty based on NEW Mark Price
            target_usd = (net_sum / 1440.0) * equity * LEVERAGE
            target_qty = target_usd / mark_price
            
            # 3. Calculate Delta
            delta = target_qty - curr_qty
            
            # KILLSWITCH: Stop if close enough
            if abs(delta) < specs["sizeStep"]:
                octopus_log(f"Target Reached (Delta {delta:.4f}). Stopping.")
                if order_id: self._cancel_ignore_error(order_id, exec_symbol)
                return

            # 4. Calculate Order Params
            side = "buy" if delta > 0 else "sell"
            abs_delta = abs(delta)
            size = self._round_to_step(abs_delta, specs["sizeStep"])
            
            # Price Logic: Start passive (0.02%), get aggressive
            price_mult = (1 - current_offset) if side == "buy" else (1 + current_offset)
            limit_price = self._round_to_step(mark_price * price_mult, specs["tickSize"])

            octopus_log(f"Step {step+1}/{MAX_STEPS} | Tgt: {target_qty:.3f} | Curr: {curr_qty:.3f} | Delta: {delta:.3f} | Off: {current_offset*100:.3f}%")

            # 5. Place or Edit Order (Using Lower)
            try:
                if not order_id:
                    res = self.kf.send_order({
                        "orderType": "lmt", "symbol": exec_symbol, "side": side, 
                        "size": size, "limitPrice": limit_price
                    })
                    if "sendStatus" in res:
                        order_id = res["sendStatus"]["order_id"]
                else:
                    self.kf.edit_order({
                        "orderId": order_id, "limitPrice": limit_price, 
                        "size": size, "symbol": exec_symbol
                    })
            except Exception as e:
                octopus_log(f"Order Update Failed (Filled/Gone): {e}", "warning")
                order_id = None 

            # 6. Decay Offset & Wait
            current_offset *= OFFSET_DECAY
            time.sleep(STEP_INTERVAL)

        # --- STEP 9: Market Sweep ---
        octopus_log("Maker sequence done. Executing Sweep.")
        
        if order_id: self._cancel_ignore_error(order_id, exec_symbol)
        time.sleep(0.5) 

        # Final Recalculation (Upper)
        curr_qty, mark_price = self._get_current_state()
        if mark_price:
            target_usd = (net_sum / 1440.0) * equity * LEVERAGE
            target_qty = target_usd / mark_price
            delta = target_qty - curr_qty
            
            if abs(delta) >= specs["sizeStep"]:
                side = "buy" if delta > 0 else "sell"
                size = self._round_to_step(abs(delta), specs["sizeStep"])
                octopus_log(f"SWEEPING MKT: {side.upper()} {size} (Delta: {delta:.4f})")
                try:
                    # Execute Sweep (Lower)
                    self.kf.send_order({
                        "orderType": "mkt", "symbol": exec_symbol, 
                        "side": side, "size": size
                    })
                except Exception as e:
                    octopus_log(f"Sweep Failed: {e}", "error")
            else:
                octopus_log("Sweep not needed (On Target).")

    def _cancel_ignore_error(self, order_id, symbol_lower):
        try: self.kf.cancel_order({"order_id": order_id, "symbol": symbol_lower})
        except: pass

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()
