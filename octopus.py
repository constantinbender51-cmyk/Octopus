#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
Updated to match 'Strategy Union' & 'Majority Vote' logic from Generator v58.
- Uses Fixed Bucket Size from JSON.
- Trains on 70% of history (matching optimizer split).
- SEQUENTIAL EXECUTION (No Threading) for easier debugging.
- Enhanced Logging for Order Status and Logic Gates.
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
import random
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
# ThreadPoolExecutor REMOVED for sequential debugging
from typing import Dict, List, Tuple, Any, Optional

# --- Local Imports ---
try:
    from kraken_futures import KrakenFuturesApi
    import stress_test
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}. Ensure 'kraken_futures.py' and 'stress_test.py' are in the directory.")
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
LEVERAGE = 2.0
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# Asset Mapping (Binance USDT -> Kraken Futures Perpetual)
SYMBOL_MAP = {
    "BTCUSDT": "ff_xbtusd_260327",
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("octopus.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Octopus")

# --- Strategy Logic Classes ---

class SubStrategy:
    """
    Represents a single model configuration (one item in 'strategy_union').
    """
    def __init__(self, config: dict):
        self.config = config
        self.bucket_count = config.get('bucket_count', 100)
        self.seq_len = config['seq_len']
        self.model_type = config['model_type']
        
        # Use FIXED bucket size from the optimization result
        self.bucket_size = config.get('bucket_size', 1.0)
        if self.bucket_size <= 0: self.bucket_size = 1.0
        
        # Runtime State
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)
        self.all_vals = []
        self.all_changes = []

    def _get_bucket(self, price: float) -> int:
        bs = self.bucket_size
        if bs <= 0: bs = 1e-9
        if price >= 0:
            return int(price // bs)
        else:
            return int(price // bs) - 1

    def train(self, prices: List[float]):
        """
        Trains the model using ONLY the first 70% of the provided data,
        matching the optimizer's training set logic.
        """
        if not prices: return

        # 1. Bucketize ALL prices first (to ensure consistency)
        buckets = [self._get_bucket(p) for p in prices]
        
        # 2. Apply 70% Split (Match app.py logic)
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        # Need enough data in the training set
        if len(train_buckets) < self.seq_len + 10:
            return 
            
        # 3. Build Maps on TRAIN set only
        self.all_vals = list(set(train_buckets))
        self.all_changes = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))
        if not self.all_vals: self.all_vals = [0]
        if not self.all_changes: self.all_changes = [0]
        
        self.abs_map.clear()
        self.der_map.clear()
        
        # Build Probability Maps
        for i in range(len(train_buckets) - self.seq_len):
            a_seq = tuple(train_buckets[i : i + self.seq_len])
            a_succ = train_buckets[i + self.seq_len]
            self.abs_map[a_seq][a_succ] += 1
            
            if self.seq_len > 1:
                # diffs within the sequence
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
                d_succ = train_buckets[i + self.seq_len] - train_buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def get_prediction_value(self, recent_prices: List[float]) -> int:
        """
        Returns the predicted BUCKET VALUE.
        Uses the maps built on the 70% train set, but queries with the LIVE recent sequence.
        """
        if len(recent_prices) < self.seq_len + 1:
            return self._get_bucket(recent_prices[-1]) if recent_prices else 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] # Need enough for derivative calc
        
        a_seq = tuple(window[1:]) 
        if self.seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        last_val = window[-1]
        
        # Prediction Logic
        if self.model_type == "Absolute":
            if a_seq in self.abs_map:
                return self.abs_map[a_seq].most_common(1)[0][0]
            return random.choice(self.all_vals) if self.all_vals else last_val
            
        elif self.model_type == "Derivative":
            if d_seq in self.der_map:
                change = self.der_map[d_seq].most_common(1)[0][0]
                return last_val + change
            change = random.choice(self.all_changes) if self.all_changes else 0
            return last_val + change
            
        elif self.model_type == "Combined":
            abs_cand = self.abs_map.get(a_seq, Counter())
            der_cand = self.der_map.get(d_seq, Counter())
            poss = set(abs_cand.keys())
            for c in der_cand.keys(): poss.add(last_val + c)
            
            if not poss: 
                return random.choice(self.all_vals) if self.all_vals else last_val
            
            best, max_s = last_val, -1
            for v in poss:
                s = abs_cand[v] + der_cand[v - last_val]
                if s > max_s: max_s, best = s, v
            return best
            
        return last_val

class EnsembleStrategy:
    """
    Holds the 'Strategy Union' for a specific Asset/Timeframe.
    Aggregates predictions using Majority Vote.
    """
    def __init__(self, asset: str, timeframe: str, config_list: List[dict]):
        self.asset = asset
        self.timeframe = timeframe
        self.id = f"{asset}_{timeframe}"
        self.virtual_position = 0.0
        
        # Initialize Sub-Strategies
        self.sub_strategies = [SubStrategy(cfg) for cfg in config_list]
        
    def train(self, prices: List[float]):
        """Trains all sub-strategies."""
        for strat in self.sub_strategies:
            strat.train(prices)
            
    def predict(self, recent_prices: List[float]) -> int:
        """
        Returns Aggregated Signal: 1 (Buy), -1 (Sell), 0 (Flat).
        Logic: Majority Vote.
        """
        if not recent_prices: return 0
        
        votes = []
        
        for strat in self.sub_strategies:
            # 1. Get Predicted Bucket
            pred_bucket = strat.get_prediction_value(recent_prices)
            
            # 2. Compare to Current Bucket
            current_bucket = strat._get_bucket(recent_prices[-1])
            
            diff = pred_bucket - current_bucket
            
            if diff > 0: votes.append(1)
            elif diff < 0: votes.append(-1)
            else: votes.append(0)
            
        # Majority Vote Logic
        up_votes = votes.count(1)
        down_votes = votes.count(-1)
        
        # logger.info(f"[{self.id}] Votes: +{up_votes} / -{down_votes} (Total {len(votes)})")
        
        if up_votes > down_votes:
            return 1
        elif down_votes > up_votes:
            return -1
        else:
            return 0

# --- Main Octopus Engine ---

class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.strategies: Dict[str, EnsembleStrategy] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        # Executor removed for sequential debugging
        self.total_strategies_count = 0
        self.instrument_specs = {}

    def initialize(self):
        logger.info("Initializing Octopus (Ensemble Version)...")
        self._fetch_instrument_specs()
        
        # Stress Test
        logger.info("Executing Startup Stress Test...")
        try:
            stress_test.run_stress_test(
                self.kf, SYMBOL_MAP, LEVERAGE, REPO_OWNER, REPO_NAME, GITHUB_PAT
            )
        except Exception as e:
            logger.error(f"Stress test failed/skipped: {e}")

        self._load_strategies_from_github()
        self._fetch_initial_data()
        self._train_all_strategies()
        logger.info("Initialization Complete.")

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
            logger.error(f"Error fetching specs: {e}")

    def _load_strategies_from_github(self):
        if not GITHUB_PAT:
            logger.error("No GitHub PAT found.")
            return

        headers = {"Authorization": f"Bearer {GITHUB_PAT}"}
        try:
            resp = requests.get(GITHUB_API_URL, headers=headers)
            resp.raise_for_status()
            files = resp.json()
            
            count = 0
            for f in files:
                if f['name'].endswith(".json"):
                    content_resp = requests.get(f['download_url'])
                    data = content_resp.json()
                    
                    asset = data.get('asset')
                    tf = data.get('timeframe')
                    
                    # Filtering: Combined Accuracy > 60%
                    acc = data.get('combined_accuracy', 0)
                    if acc < 60.0:
                        logger.warning(f"Skipping {asset} {tf} (Acc {acc:.2f}%)")
                        continue
                    
                    # Load the FULL strategy union
                    strategy_union = data.get('strategy_union', [])
                    if not strategy_union: continue

                    ens = EnsembleStrategy(asset, tf, strategy_union)
                    self.strategies[ens.id] = ens
                    count += 1
            
            self.total_strategies_count = count
            logger.info(f"Loaded {count} Ensemble Strategies.")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")

    def _fetch_initial_data(self):
        active_assets = set(s.asset for s in self.strategies.values())
        logger.info(f"Fetching history for {len(active_assets)} assets (Since 2020)...")
        start_timestamp_2020 = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            try:
                url = "https://api.binance.com/api/v3/klines"
                all_candles = []
                current_start = start_timestamp_2020
                
                while True:
                    params = {"symbol": asset, "interval": "15m", "limit": 1000, "startTime": current_start}
                    r = requests.get(url, params=params)
                    data = r.json()
                    if not data or not isinstance(data, list): break
                    all_candles.extend(data)
                    if len(data) < 1000: break
                    current_start = int(data[-1][6]) + 1
                    time.sleep(0.05)
                
                self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"Loaded {len(all_candles)} candles for {asset}")
            except Exception as e:
                logger.error(f"Data fetch error {asset}: {e}")

    def _train_all_strategies(self):
        logger.info("Training Ensemble Strategies...")
        for strat in self.strategies.values():
            raw = self.price_history[strat.asset]
            if not raw: continue
            prices = self._resample(raw, strat.timeframe)
            strat.train(prices)

    def _resample(self, raw_data: List[Tuple[int, float]], timeframe: str) -> List[float]:
        if timeframe == "15m": return [x[1] for x in raw_data]
        df = pd.DataFrame(raw_data, columns=['ts', 'price'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        tf_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        target = tf_map.get(timeframe)
        if not target: return [x[1] for x in raw_data]
        return df['price'].resample(target).last().dropna().tolist()

    def _round_to_step(self, value: float, step: float) -> float:
        if step == 0: return value
        rounded = round(value / step) * step
        if isinstance(step, float) and "." in str(step):
            decimals = len(str(step).split(".")[1])
            rounded = round(rounded, decimals)
        elif isinstance(step, int) or step.is_integer():
            rounded = int(rounded)
        return rounded

    def run(self):
        while True:
            now = datetime.now(timezone.utc)
            minute = now.minute
            hour = now.hour
            
            if minute % 15 == 1:
                logger.info(f"--- Trigger: {hour:02}:{minute:02} ---")
                self._update_all_data()
                
                tfs_to_run = ["15m"]
                if minute == 1 or minute == 31: tfs_to_run.append("30m")
                if minute == 1:
                    tfs_to_run.append("60m")
                    if hour % 4 == 0: tfs_to_run.append("240m")
                    if hour == 0: tfs_to_run.append("1d")

                self._process_strategies(tfs_to_run)
                time.sleep(60)
            time.sleep(1)

    def _update_all_data(self):
        now = datetime.now(timezone.utc)
        current_interval_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        limit_ts = int(current_interval_start.timestamp() * 1000)

        active_assets = set(s.asset for s in self.strategies.values())
        for asset in active_assets:
            try:
                params = {"symbol": asset, "interval": "15m", "limit": 5}
                r = requests.get("https://api.binance.com/api/v3/klines", params=params)
                data = r.json()
                last_stored_ts = self.price_history[asset][-1][0]
                
                for candle in data:
                    open_ts = int(candle[0])
                    close_ts = int(candle[6])
                    price = float(candle[4])
                    if close_ts > last_stored_ts and open_ts < limit_ts:
                        self.price_history[asset].append((close_ts, price))
                        
                if len(self.price_history[asset]) > 200000:
                    self.price_history[asset] = self.price_history[asset][-200000:]
            except Exception as e:
                logger.error(f"Update failed for {asset}: {e}")

    def _process_strategies(self, active_tfs: List[str]):
        try:
            acc = self.kf.get_accounts()
            # Handle flex/multi-collateral structure
            if "flex" in acc.get("accounts", {}):
                equity = float(acc["accounts"]["flex"].get("marginEquity", 0))
            elif "accounts" in acc:
                # Fallback to first available account
                first_acc = list(acc["accounts"].values())[0]
                equity = float(first_acc.get("marginEquity", 0))
            else:
                equity = 0
                
            if equity <= 0:
                logger.error("Equity 0. Aborting.")
                return
        except Exception as e:
            logger.error(f"Account fetch failed: {e}")
            return

        if self.total_strategies_count == 0: return
        unit_size_usd = (equity * LEVERAGE) / self.total_strategies_count
        logger.info(f"Equity: ${equity:.2f} | Unit: ${unit_size_usd:.2f}")

        active_assets = set()
        for s in self.strategies.values():
            if s.timeframe in active_tfs:
                active_assets.add(s.asset)
                raw = self.price_history[s.asset]
                prices = self._resample(raw, s.timeframe)
                s.train(prices)
                
                sig = s.predict(prices)
                s.virtual_position = sig * unit_size_usd
                logger.info(f"Strat {s.id}: Sig {sig} -> Pos ${s.virtual_position:.2f}")

        # SEQUENTIAL EXECUTION for better logging and debugging
        for asset in active_assets:
            self._execute_asset_logic(asset)

    def _execute_asset_logic(self, binance_asset: str):
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol: return

        net_target_usd = sum(s.virtual_position for s in self.strategies.values() if s.asset == binance_asset)

        try:
            open_pos = self.kf.get_open_positions()
            current_pos_size = 0.0
            if "openPositions" in open_pos:
                for p in open_pos["openPositions"]:
                    if p["symbol"].lower() == kf_symbol.lower():
                        size = float(p["size"])
                        if p["side"] == "short": size = -size
                        current_pos_size = size
                        break
            
            logger.info(f"[{kf_symbol}] Logic Triggered. Net Target: ${net_target_usd:.2f} | Current Pos: {current_pos_size}")

            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0: 
                logger.error(f"[{kf_symbol}] Mark price is 0. Aborting.")
                return
            
            target_contracts = net_target_usd / mark_price
            delta = target_contracts - current_pos_size
            
            logger.info(f"[{kf_symbol}] Mark: {mark_price} | Target Contracts: {target_contracts:.4f} | Delta: {delta:.4f}")
            
            specs = self.instrument_specs.get(kf_symbol.lower())
            size_increment = specs['sizeStep'] if specs else 0.001
            check_qty = self._round_to_step(abs(delta), size_increment)

            logger.info(f"[{kf_symbol}] Check Qty: {check_qty} | Min Step: {size_increment}")

            if check_qty < size_increment: 
                logger.info(f"[{kf_symbol}] Delta too small. Skipping.")
                return

            self._run_maker_loop(kf_symbol, delta, mark_price)

        except Exception as e:
            logger.error(f"[{kf_symbol}] Exec Error: {e}")

    def _run_maker_loop(self, symbol: str, quantity: float, initial_mark: float):
        side = "buy" if quantity > 0 else "sell"
        abs_qty = abs(quantity)
        decay_steps = 10
        order_id = None
        
        specs = self.instrument_specs.get(symbol.lower())
        size_inc = specs['sizeStep'] if specs else 0.001
        price_inc = specs['tickSize'] if specs else 0.01

        for i in range(decay_steps):
            try:
                tickers = self.kf.get_tickers()
                curr_mark = 0.0
                for t in tickers.get("tickers", []):
                    if t["symbol"].lower() == symbol.lower():
                        curr_mark = float(t["markPrice"])
                        break
                if curr_mark == 0: curr_mark = initial_mark
                
                direction = 1 if side == "buy" else -1
                decay = math.exp(-i * 0.5)
                offset = curr_mark * 0.01 * -direction * decay
                
                final_limit = self._round_to_step(curr_mark + offset, price_inc)
                final_size = self._round_to_step(abs_qty, size_inc)
                
                if order_id is None:
                    logger.info(f"[{symbol}] Placing {side.upper()} {final_size} @ {final_limit}...")
                    resp = self.kf.send_order({
                        "orderType": "lmt", "symbol": symbol, "side": side,
                        "size": final_size, "limitPrice": final_limit
                    })
                    logger.info(f"[{symbol}] Order Response: {resp}")
                    
                    if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                         order_id = resp["sendStatus"]["order_id"]
                    else: 
                        logger.warning(f"[{symbol}] Order failed or no ID returned.")
                        break
                else:
                    logger.info(f"[{symbol}] Editing Order {order_id} to {final_limit}...")
                    edit_resp = self.kf.edit_order({
                        "orderId": order_id, "limitPrice": final_limit,
                        "size": final_size, "symbol": symbol 
                    })
                    logger.info(f"[{symbol}] Edit Response: {edit_resp}")
                    
                time.sleep(30)
            except Exception as e:
                logger.error(f"[{symbol}] Maker Loop Error: {e}")
                time.sleep(5)
        
        if order_id:
            try:
                logger.info(f"[{symbol}] Cancelling Order {order_id}...")
                pb = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                self.kf.cancel_order({"order_id": order_id, "symbol": symbol, "processBefore": pb})
            except: pass

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()
