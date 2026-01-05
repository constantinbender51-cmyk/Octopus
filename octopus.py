#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
"""

import os
import sys
import time
import json
import math
import base64
import logging
import threading
import requests
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

# --- Local Imports ---
try:
    from kraken_futures import KrakenFuturesApi
except ImportError:
    print("CRITICAL: 'kraken_futures.py' not found. Please ensure it is in the same directory.")
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
GITHUB_PAT = os.getenv("PAT")

# Global Settings
LEVERAGE = 2.0  # Global leverage setting
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# Asset Mapping (Binance USDT -> Kraken Futures Perpetual)
SYMBOL_MAP = {
    "BTCUSDT": "ff_xbtusd_261225",
    "ETHUSDT": "pf_ethusd",
    "SOLUSDT": "pf_solusd",
    "BNBUSDT": "pf_bnbusd",
    "XRPUSDT": "pf_xrpusd",
    "ADAUSDT": "pf_adausd",
    "DOGEUSDT": "pf_dogeusd",
    "AVAXUSDT": "pf_avaxusd",
    "DOTUSDT": "pf_dotusd",
    "LINKUSDT": "pf_linkusd",
}

# Reverse map for logging
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

# Tick Size Configuration
# Maps internal identifiers (lowercase) to tick sizes
TICK_SIZES = {
    "ada": 0.00001,
    "eth": 0.1,
    "sol": 0.01,
    "bnb": 0.01,
    "xrp": 0.00001,
    "doge": 0.000001,
    "avax": 0.001,
    "link": 0.001,
    "dot": 0.001,
    "xbt": 1,    # BTC usually XBT on Kraken
    "btc": 1,
}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("octopus.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Octopus")

# --- Helper Classes ---

class Strategy:
    """
    Represents a single loaded strategy (Asset + Timeframe).
    Holds the model logic (Probability Maps) and its current Virtual Position.
    """
    def __init__(self, asset: str, timeframe: str, config: dict):
        self.asset = asset
        self.timeframe = timeframe
        self.config = config
        self.bucket_size = config['bucket_size']
        self.seq_len = config['seq_len']
        self.model_type = config['model_type']
        
        # State
        self.virtual_position = 0.0  # The position this strategy *wants* to hold
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)
        self.all_vals = []
        self.all_changes = []
        
        # Identity
        self.id = f"{asset}_{timeframe}"

    def train(self, prices: List[float]):
        """Rebuilds the probability maps based on provided historical prices."""
        buckets = [self._get_bucket(p) for p in prices]
        
        if len(buckets) < self.seq_len + 10:
            return # Not enough data
            
        self.all_vals = list(set(buckets))
        self.all_changes = list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
        
        self.abs_map.clear()
        self.der_map.clear()
        
        for i in range(len(buckets) - self.seq_len):
            a_seq = tuple(buckets[i : i + self.seq_len])
            a_succ = buckets[i + self.seq_len]
            self.abs_map[a_seq][a_succ] += 1
            
            if i > 0:
                d_seq = tuple(buckets[j] - buckets[j-1] for j in range(i, i + self.seq_len))
                d_succ = buckets[i + self.seq_len] - buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def predict(self, recent_prices: List[float]) -> int:
        """Returns signal: 1 (Buy), -1 (Sell), 0 (Flat/Neutral)."""
        if len(recent_prices) < self.seq_len + 1:
            return 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        
        # Current Context
        curr_buckets = buckets[-self.seq_len:] # Last N buckets
        a_seq = tuple(curr_buckets)
        d_seq = tuple(curr_buckets[j] - curr_buckets[j-1] for j in range(1, len(curr_buckets)))
        
        last_val = curr_buckets[-1]
        
        # Get Prediction Target
        pred_bucket = last_val
        
        if self.model_type == "Absolute":
            if a_seq in self.abs_map:
                pred_bucket = self.abs_map[a_seq].most_common(1)[0][0]
        elif self.model_type == "Derivative":
            if d_seq in self.der_map:
                change = self.der_map[d_seq].most_common(1)[0][0]
                pred_bucket = last_val + change
        elif self.model_type == "Combined":
            # (Simplified logic for brevity, matching training logic)
            abs_cand = self.abs_map.get(a_seq, Counter())
            der_cand = self.der_map.get(d_seq, Counter())
            poss = set(abs_cand.keys())
            for c in der_cand.keys(): poss.add(last_val + c)
            
            best, max_s = last_val, -1
            for v in poss:
                s = abs_cand[v] + der_cand[v - last_val]
                if s > max_s: max_s, best = s, v
            pred_bucket = best

        # Signal Logic
        if pred_bucket > last_val: return 1
        elif pred_bucket < last_val: return -1
        else: return 0

    def _get_bucket(self, price: float) -> int:
        if price >= 0:
            return (int(price) // self.bucket_size) + 1
        else:
            return (int(price + 1) // self.bucket_size) - 1


class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.strategies: Dict[str, Strategy] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list) # Asset -> [(ts, price)]
        self.executor = ThreadPoolExecutor(max_workers=5) # Parallel execution
        self.total_strategies_count = 0

    # --- Initialization ---
    def initialize(self):
        logger.info("Initializing Octopus...")
        self._load_strategies_from_github()
        self._fetch_initial_data()
        self._train_all_strategies()
        logger.info("Initialization Complete. Entering Wait Loop.")

    def _load_strategies_from_github(self):
        """Downloads all JSON strategy files from the repo."""
        if not GITHUB_PAT:
            logger.error("No GitHub PAT found. Cannot load strategies.")
            return

        headers = {"Authorization": f"Bearer {GITHUB_PAT}"}
        try:
            resp = requests.get(GITHUB_API_URL, headers=headers)
            resp.raise_for_status()
            files = resp.json()
            
            count = 0
            for f in files:
                if f['name'].endswith(".json"):
                    # Download content
                    content_resp = requests.get(f['download_url'])
                    data = content_resp.json()
                    
                    # Parse filename: ASSET_TIMEFRAME.json
                    # But config is inside.
                    asset = data['asset']
                    tf = data['timeframe']
                    
                    # Pick best strategy from the union
                    best_strat = data['strategy_union'][0] # Top 1 is best
                    
                    s = Strategy(asset, tf, best_strat)
                    self.strategies[s.id] = s
                    count += 1
            
            self.total_strategies_count = count
            logger.info(f"Loaded {count} strategies from GitHub.")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")

    def _fetch_initial_data(self):
        """Fetches all 15m data since 2020 for all active assets from Binance."""
        active_assets = set(s.asset for s in self.strategies.values())
        logger.info(f"Fetching historical data for {len(active_assets)} assets (Since 2020)...")
        
        # 2020-01-01 00:00:00 UTC timestamp in milliseconds
        start_timestamp_2020 = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            try:
                # Binance requires Symbol, e.g. BTCUSDT
                url = "https://api.binance.com/api/v3/klines"
                
                all_candles = []
                current_start = start_timestamp_2020
                
                while True:
                    params = {
                        "symbol": asset, 
                        "interval": "15m", 
                        "limit": 1000,
                        "startTime": current_start
                    }
                    
                    r = requests.get(url, params=params)
                    data = r.json()
                    
                    if not data or not isinstance(data, list):
                        break
                        
                    all_candles.extend(data)
                    
                    # Check if we reached the end (fewer than limit returned)
                    if len(data) < 1000:
                        break
                        
                    # Update start time for next batch
                    # Index 6 is Close Time. We want next candle, so +1ms
                    last_close_time = int(data[-1][6])
                    current_start = last_close_time + 1
                    
                    # Rate limit safety
                    time.sleep(0.1)
                
                # Store (Time, Close) -> Index 6 is Close Time, Index 4 is Close Price
                self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"Loaded {len(all_candles)} candles for {asset} since 2020")
                
            except Exception as e:
                logger.error(f"Error fetching data for {asset}: {e}")

    def _train_all_strategies(self):
        """Resamples data and trains every strategy."""
        logger.info("Training strategies...")
        for s_id, strat in self.strategies.items():
            # 1. Get raw 15m data
            raw = self.price_history[strat.asset]
            if not raw: continue
            
            # 2. Resample
            prices = self._resample(raw, strat.timeframe)
            
            # 3. Train
            strat.train(prices)

    def _resample(self, raw_data: List[Tuple[int, float]], timeframe: str) -> List[float]:
        """Convert 15m raw data to target timeframe prices."""
        if timeframe == "15m":
            return [x[1] for x in raw_data]
            
        # Pandas for complex resampling
        df = pd.DataFrame(raw_data, columns=['ts', 'price'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        
        tf_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        target = tf_map.get(timeframe)
        
        if not target: return [x[1] for x in raw_data]
        
        # Resample logic: Take last close
        resampled = df['price'].resample(target).last().dropna()
        return resampled.tolist()

    def _get_tick_size(self, symbol: str) -> float:
        """Returns the tick size for a given Kraken symbol."""
        s_lower = symbol.lower()
        for key, tick in TICK_SIZES.items():
            if key in s_lower:
                return tick
        # Default fallback if not found (safer to be precise than loose)
        return 0.001

    def _format_price(self, price: float, symbol: str) -> float:
        """
        Rounds price to the nearest tick size and formats it correctly.
        Returns a float if the API handles it, but ensures the value is mathematically aligned.
        Kraken API often requires the JSON payload to have the number.
        However, the *value* must be quantized.
        """
        tick = self._get_tick_size(symbol)
        
        # Quantize to tick size
        # Logic: round(price / tick) * tick
        # Using Decimal to avoid floating point artifacts (e.g. 3000.100000002)
        
        d_price = Decimal(str(price))
        d_tick = Decimal(str(tick))
        
        quantized = (d_price / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick
        
        # Return as float for JSON serialization, but it's now a clean tick multiple
        # e.g. 50000.0 (float) or 2.343 (float)
        # Note: If Kraken strictly needs integer "50000" and not "50000.0" in JSON, 
        # the standard json serializer usually handles floats with .0.
        # However, the requirement "50000 not 50000.0" usually implies string formatting or int type.
        # Python's `json.dumps` turns int-like floats into floats (50000.0).
        # We might need to return int if it's a whole number tick.
        
        if tick >= 1:
            return int(quantized)
        else:
            return float(quantized)

    # --- Core Loop Logic ---

    def run(self):
        while True:
            now = datetime.now(timezone.utc)
            minute = now.minute
            hour = now.hour
            
            # 1. Update Data (Every 15 mins)
            if minute % 15 == 1: # Run at :01, :16, :31, :46
                logger.info(f"--- Trigger: {hour:02}:{minute:02} ---")
                self._update_all_data() # Append latest candle
                
                # 2. Determine which TFs to run
                tfs_to_run = []
                tfs_to_run.append("15m")
                
                if minute == 1 or minute == 31:
                    tfs_to_run.append("30m")
                
                if minute == 1:
                    tfs_to_run.append("60m")
                    if hour % 4 == 0: tfs_to_run.append("240m")
                    if hour == 0: tfs_to_run.append("1d")

                logger.info(f"Running strategies for: {tfs_to_run}")
                
                # 3. Execute Logic
                self._process_strategies(tfs_to_run)
                
                # Sleep to avoid double triggering in the same minute
                time.sleep(60)
            
            time.sleep(1)

    def _update_all_data(self):
        """Fetches just the last few candles to append."""
        active_assets = set(s.asset for s in self.strategies.values())
        for asset in active_assets:
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {"symbol": asset, "interval": "15m", "limit": 5} 
                r = requests.get(url, params=params)
                data = r.json()
                
                # Append new ones if timestamp > last stored
                last_stored_ts = self.price_history[asset][-1][0]
                for candle in data:
                    ts = int(candle[6])
                    price = float(candle[4])
                    if ts > last_stored_ts:
                        self.price_history[asset].append((ts, price))
                        
                # Trim list
                if len(self.price_history[asset]) > 200000:
                    self.price_history[asset] = self.price_history[asset][-200000:]
                    
            except Exception as e:
                logger.error(f"Update failed for {asset}: {e}")

    def _process_strategies(self, active_tfs: List[str]):
        """Generates signals and manages execution."""
        
        # 1. Get Capital
        try:
            acc = self.kf.get_accounts()
            # Navigate complex structure to find Equity
            if "flex" in acc.get("accounts", {}):
                equity = float(acc["accounts"]["flex"].get("marginEquity", 0))
            else:
                # Fallback to first account found
                first_acc = list(acc.get("accounts", {}).values())[0]
                equity = float(first_acc.get("marginEquity", 0))
                
            if equity <= 0:
                logger.error("Equity is 0 or negative. Aborting.")
                return
                
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")
            return

        # 2. Calculate Unit Size
        if self.total_strategies_count == 0: return
        unit_size_usd = (equity * LEVERAGE) / self.total_strategies_count
        logger.info(f"Equity: ${equity:.2f} | Unit Size: ${unit_size_usd:.2f}")

        # 3. Generate Signals & Update Virtual Positions
        active_assets = set()
        
        for s in self.strategies.values():
            if s.timeframe in active_tfs:
                active_assets.add(s.asset)
                
                # Get Data
                raw = self.price_history[s.asset]
                prices = self._resample(raw, s.timeframe)
                
                # Retrain with new data (fast)
                s.train(prices)
                
                # Predict
                sig = s.predict(prices) # 1, -1, 0
                
                # Update Virtual Position
                # 1 = Long 1 Unit, -1 = Short 1 Unit, 0 = Flat
                s.virtual_position = sig * unit_size_usd
                
                logger.info(f"Strategy {s.id}: Signal {sig} -> VirtPos ${s.virtual_position:.2f}")

        # 4. Aggregation & Execution (Per Asset)
        # Launch parallel execution for each asset
        for asset in active_assets:
            self.executor.submit(self._execute_asset_logic, asset)

    def _execute_asset_logic(self, binance_asset: str):
        """Calculates Net Target and executes Maker Order loop."""
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol:
            logger.warning(f"No Kraken mapping for {binance_asset}")
            return

        # A. Calculate Net Target (USD Value)
        net_target_usd = 0.0
        for s in self.strategies.values():
            if s.asset == binance_asset:
                net_target_usd += s.virtual_position

        # B. Get Current Position on Kraken
        # NOTE: Using lower() for reading/comparing data (all other requests)
        try:
            open_pos = self.kf.get_open_positions()
            current_pos_size = 0.0
            
            if "openPositions" in open_pos:
                for p in open_pos["openPositions"]:
                    if p["symbol"].lower() == kf_symbol.lower():
                        size = float(p["size"]) # Contracts
                        # Check direction
                        if p["side"] == "short": size = -size
                        current_pos_size = size
                        break
        except Exception as e:
            logger.error(f"[{kf_symbol}] Failed to get positions: {e}")
            return

        # C. Get Market Price for Conversion (USD -> Contracts)
        try:
            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0: raise ValueError("Mark price 0")
            
            # Convert Target USD to Contracts
            target_contracts = net_target_usd / mark_price
            
            # Rounding for contracts (Kraken rejects high precision on size too)
            # Assuming 4 decimals for contract size is safe generally, 
            # but usually size is also subject to lot size.
            target_contracts = round(target_contracts, 4)
            
            # Delta
            delta = target_contracts - current_pos_size
            
            # Min size threshold (approx $10)
            if abs(delta * mark_price) < 10:
                logger.info(f"[{kf_symbol}] Delta small (${delta*mark_price:.2f}). Skipping.")
                return

            logger.info(f"[{kf_symbol}] Net Target: {target_contracts} | Curr: {current_pos_size} | Delta: {delta}")
            
            # D. Execute Maker Loop
            self._run_maker_loop(kf_symbol, delta, mark_price)

        except Exception as e:
            logger.error(f"[{kf_symbol}] Execution Logic Failed: {e}")

    def _run_maker_loop(self, symbol: str, quantity: float, initial_mark: float):
        """
        Places a limit order and updates it every 30s to chase/decay towards mark.
        quantity: positive (buy) or negative (sell).
        """
        side = "buy" if quantity > 0 else "sell"
        abs_qty = abs(quantity)
        
        # Initial Offset (e.g., 0.5%)
        decay_steps = 10 # 5 minutes / 30s
        
        order_id = None
        
        for i in range(decay_steps):
            try:
                # 1. Get Fresh Mark Price (Read request: lowercase)
                tickers = self.kf.get_tickers()
                curr_mark = 0.0
                for t in tickers.get("tickers", []):
                    if t["symbol"].lower() == symbol.lower():
                        curr_mark = float(t["markPrice"])
                        break
                
                if curr_mark == 0: curr_mark = initial_mark
                
                # 2. Calculate Decay Price
                direction = 1 if side == "buy" else -1
                decay_factor = math.exp(-i * 0.5)
                offset = curr_mark * 0.01 * -direction * decay_factor
                
                raw_limit_price = curr_mark + offset
                
                # Format Price respecting Tick Size
                limit_price = self._format_price(raw_limit_price, symbol)
                
                logger.info(f"[{symbol}] Maker Iter {i}: {side.upper()} {abs_qty} @ {limit_price} (Mark: {curr_mark})")

                # 3. Place or Edit
                # NOTE: Order requests symbol must be ALL CAPS
                upper_symbol = symbol.upper()

                if order_id is None:
                    # Send New
                    resp = self.kf.send_order({
                        "orderType": "lmt",
                        "symbol": upper_symbol, 
                        "side": side,
                        "size": abs_qty,
                        "limitPrice": limit_price
                    })
                    if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                         order_id = resp["sendStatus"]["order_id"]
                    else:
                         logger.error(f"[{symbol}] Order fail: {resp}")
                         break # Fatal
                else:
                    # Edit (edit_order typically uses orderId, but if we sent symbol it would need to be caps)
                    # We only send orderId, price, size.
                    self.kf.edit_order({
                        "orderId": order_id,
                        "limitPrice": limit_price,
                        "size": abs_qty 
                    })

                # 4. Wait & Check Fill
                time.sleep(30)
                
                # Check status (Read request: lowercase or id based)
                status = self.kf.get_order(order_id)
                
            except Exception as e:
                logger.error(f"[{symbol}] Maker Loop Error: {e}")
                time.sleep(5)
        
        # Timeout - Cancel
        if order_id:
            try:
                logger.info(f"[{symbol}] Timeout. Cancelling.")
                self.kf.cancel_order({"orderId": order_id})
            except:
                pass

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()