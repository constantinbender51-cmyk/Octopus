#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
Updated to match 'Strategy Union' & 'Majority Vote' logic from Generator v58.
- Uses Fixed Bucket Size from JSON.
- Trains on 70% of history (matching optimizer split).
- PARALLEL EXECUTION (ThreadPoolExecutor).
- Precise Timing (Execute at XX:XX:05).
- Dynamic Execution Window (Sprinter vs Marathoner).
- NO Retraining during live run.
- LOGGING: Enhanced decision logging.
- TRADE LOGGING: Local file (trade_log.txt) with strict format.
- STRICT ACCURACY: 
    - Moves < min bucket size = Flat (0).
    - Moves >= min bucket size in OPPOSITE direction = Incorrect (-1).
    - Moves >= min bucket size in CORRECT direction = Correct (1).
- RESET LOGIC: Resets virtual position on every execution.
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
from concurrent.futures import ThreadPoolExecutor
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
TRADE_LOG_FILE = "trade_log.txt"

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
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    handlers=[logging.FileHandler("octopus.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Octopus")

# --- Trade Logger Class ---

class TradeLogger:
    def __init__(self, filename=TRADE_LOG_FILE):
        self.filename = filename
        # Stores active trade state: {strat_id: {"signal": int, "price": float, "threshold": float}}
        self.pending_trades = {}
        self.lock = threading.Lock()

    def evaluate_and_log_close(self, strat_id: str, current_price: float, time_str: str):
        """
        Checks the outcome of the previous prediction and writes the Close log.
        Outcome: 1 (Correct), -1 (Incorrect), 0 (Flat/Small Move).
        """
        with self.lock:
            if strat_id not in self.pending_trades:
                return

            last = self.pending_trades[strat_id]
            last_signal = last["signal"]
            last_price = last["price"]
            threshold = last.get("threshold", 0.0)
            
            # Remove processed prediction immediately
            del self.pending_trades[strat_id]

            # Logic matching strict accuracy rules
            price_diff = current_price - last_price
            outcome = 0 # Default Flat

            if abs(price_diff) < threshold:
                outcome = 0 # Flat
            else:
                if last_signal == 1: # Predicted UP
                    outcome = 1 if price_diff > 0 else -1
                elif last_signal == -1: # Predicted DOWN
                    outcome = 1 if price_diff < 0 else -1
            
            # Write Close Log
            # Format: Close [strategy_name] 1/0/-1 [time]
            try:
                with open(self.filename, "a") as f:
                    f.write(f"Close [{strat_id}] {outcome} [{time_str}]\n")
            except Exception as e:
                logger.error(f"Failed to write close log: {e}")

    def log_signal_and_store(self, strat_id: str, signal: int, current_price: float, threshold: float, time_str: str):
        """
        Writes the Signal log (if non-zero) and stores state for next evaluation.
        """
        with self.lock:
            # Only log signals if they are actionable (non-zero)
            if signal != 0:
                # Format: Signal: 1/-1 [strategy_name] [time]
                try:
                    with open(self.filename, "a") as f:
                        f.write(f"Signal: {signal} [{strat_id}] [{time_str}]\n")
                except Exception as e:
                    logger.error(f"Failed to write signal log: {e}")
            
            # Store state for next cycle's evaluation
            self.pending_trades[strat_id] = {
                "signal": signal,
                "price": current_price,
                "threshold": threshold
            }

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
            self.abs_map[a_seq][train_buckets[i + self.seq_len]] += 1
            
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
        
        # Calculate Minimum Bucket Size for Performance Tracking
        if self.sub_strategies:
            self.min_bucket_size = min(s.bucket_size for s in self.sub_strategies)
        else:
            self.min_bucket_size = 0.0
        
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
        flat_votes = votes.count(0)
        
        signal = 0
        if up_votes > down_votes:
            signal = 1
        elif down_votes > up_votes:
            signal = -1
        else:
            signal = 0

        # --- LOG DECISION ---
        logger.info(f"[{self.id}] Decision: +{up_votes} / -{down_votes} / ={flat_votes} => Signal: {signal}")
        
        return signal

# --- Main Octopus Engine ---

class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.strategies: Dict[str, EnsembleStrategy] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        # Unbounded executor for max parallelism
        self.executor = ThreadPoolExecutor(max_workers=None)
        self.total_strategies_count = 0
        self.instrument_specs = {}
        # New Trade Logger
        self.trade_logger = TradeLogger()

    def initialize(self):
        logger.info("Initializing Octopus (Parallel Ensemble Version)...")
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
        logger.info("Initialization Complete. Strategies trained and ready.")

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
                if f['name'].endswith(".json") and f['name'] != "performance.json":
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
        """Fetches history sequentially during init to avoid rate limits before running."""
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
        logger.info("Training Ensemble Strategies (One-time Init)...")
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
        logger.info("Bot started. Waiting for next 15m mark + 5s...")
        while True:
            now = datetime.now(timezone.utc)
            
            # Precise Trigger: XX:00:05, XX:15:05, XX:30:05, XX:45:05
            if now.minute % 15 == 0 and 5 <= now.second < 10:
                logger.info(f"--- Trigger: {now.strftime('%H:%M:%S')} ---")
                
                # 1. Calculate active timeframes for this trigger
                tfs_to_run = ["15m"]
                if now.minute == 0 or now.minute == 30: tfs_to_run.append("30m")
                if now.minute == 0:
                    tfs_to_run.append("60m")
                    if now.hour % 4 == 0: tfs_to_run.append("240m")
                    if now.hour == 0: tfs_to_run.append("1d")
                
                # 2. Parallel Data Update
                self._update_all_data_parallel()
                
                # 3. Parallel Strategy Execution
                self._process_strategies_parallel(tfs_to_run)
                
                # Sleep to prevent re-triggering within the same minute
                time.sleep(50)
                
            time.sleep(0.1) # Fast polling for precision

    def _update_single_asset(self, asset: str, limit_ts: int):
        """Worker function to update a single asset."""
        try:
            params = {"symbol": asset, "interval": "15m", "limit": 5}
            r = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=5)
            data = r.json()
            if not isinstance(data, list): return

            last_stored_ts = self.price_history[asset][-1][0]
            
            for candle in data:
                open_ts = int(candle[0])
                close_ts = int(candle[6])
                price = float(candle[4])
                # Ensure we don't add future candles beyond the current trigger time
                if close_ts > last_stored_ts and open_ts < limit_ts:
                    self.price_history[asset].append((close_ts, price))
                    
            # Keep history manageable
            if len(self.price_history[asset]) > 5000:
                self.price_history[asset] = self.price_history[asset][-5000:]
        except Exception as e:
            logger.error(f"Update failed for {asset}: {e}")

    def _update_all_data_parallel(self):
        now = datetime.now(timezone.utc)
        current_interval_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        limit_ts = int(current_interval_start.timestamp() * 1000)

        active_assets = set(s.asset for s in self.strategies.values())
        futures = []
        for asset in active_assets:
            futures.append(self.executor.submit(self._update_single_asset, asset, limit_ts))
        
        # Wait for all updates to complete
        for f in futures:
            f.result()
        logger.info("Data update complete.")

    def _process_strategies_parallel(self, active_tfs: List[str]):
        try:
            acc = self.kf.get_accounts()
            # Handle flex/multi-collateral structure
            if "flex" in acc.get("accounts", {}):
                equity = float(acc["accounts"]["flex"].get("marginEquity", 0))
            elif "accounts" in acc:
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
        logger.info(f"Equity: ${equity:.2f} | Unit: ${unit_size_usd:.2f} | TFs: {active_tfs}")

        # 1. Determine Execution Mode (Sprinter vs Marathoner)
        # If any higher timeframe is active, use Marathoner settings
        is_marathon = any(tf in ["60m", "240m", "1d"] for tf in active_tfs)
        
        if is_marathon:
            # Marathoner: 5 mins, 10s interval, passive start, slow converge
            exec_duration = 300
            exec_interval = 10
            start_offset_bp = -5 
            step_bp = 0.5 
            mode_name = "MARATHONER"
        else:
            # Sprinter: 60s, 5s interval, at mark, fast converge
            exec_duration = 60
            exec_interval = 5
            start_offset_bp = 0 
            step_bp = 1.0 
            mode_name = "SPRINTER"

        logger.info(f"Execution Mode: {mode_name}")

        # 2. Calculate Signals Parallel (Fast)
        active_assets = set()
        
        def calc_signal(strat):
            if strat.timeframe in active_tfs:
                active_assets.add(strat.asset)
                raw = self.price_history[strat.asset]
                prices = self._resample(raw, strat.timeframe)
                current_price = prices[-1] if prices else 0.0
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # --- 1. EVALUATE PREVIOUS TRADE & LOG CLOSE ---
                self.trade_logger.evaluate_and_log_close(strat.id, current_price, time_str)
                
                # --- 2. RESET VIRTUAL POSITION (Strictly 1-candle trade) ---
                strat.virtual_position = 0.0
                
                # --- 3. PREDICT NEW SIGNAL ---
                sig = strat.predict(prices)
                
                # --- 4. LOG SIGNAL & STORE STATE ---
                self.trade_logger.log_signal_and_store(strat.id, sig, current_price, strat.min_bucket_size, time_str)

                strat.virtual_position = sig * unit_size_usd
                
                logger.info(f"Strat {strat.id}: Signal {sig} | Alloc: ${strat.virtual_position:.2f}")

        # Run signal calcs
        f_sigs = [self.executor.submit(calc_signal, s) for s in self.strategies.values()]
        for f in f_sigs: f.result()

        # 3. Execute Asset Logic Parallel (The Maker Loops)
        # We spawn a thread for each asset to run the maker loop simultaneously
        for asset in active_assets:
            self.executor.submit(
                self._execute_single_asset_logic, 
                asset, 
                exec_duration, 
                exec_interval, 
                start_offset_bp, 
                step_bp
            )

    def _execute_single_asset_logic(self, binance_asset: str, duration: int, interval: int, start_bp: float, step_bp: float):
        """
        Runs logic for a single asset.
        Calculates Net Target and enters Maker Loop if needed.
        """
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol: return

        net_target_usd = sum(s.virtual_position for s in self.strategies.values() if s.asset == binance_asset)

        try:
            # Check Current Position
            # Note: With high concurrency, this might occasionally fail on nonce/rate limit.
            # Ideally retry logic should be here, keeping it simple for now.
            open_pos = self.kf.get_open_positions()
            current_pos_size = 0.0
            if "openPositions" in open_pos:
                for p in open_pos["openPositions"]:
                    if p["symbol"].lower() == kf_symbol.lower():
                        size = float(p["size"])
                        if p["side"] == "short": size = -size
                        current_pos_size = size
                        break
            
            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0: return
            
            target_contracts = net_target_usd / mark_price
            delta = target_contracts - current_pos_size
            
            specs = self.instrument_specs.get(kf_symbol.lower())
            size_increment = specs['sizeStep'] if specs else 0.001
            check_qty = self._round_to_step(abs(delta), size_increment)

            if check_qty < size_increment: 
                logger.info(f"[{kf_symbol}] Delta {delta:.4f} too small (Min: {size_increment}). Holding.")
                return

            logger.info(f"[{kf_symbol}] Executing Delta: {delta:.4f} (Target: ${net_target_usd:.2f})")

            self._run_maker_loop(kf_symbol, delta, mark_price, duration, interval, start_bp, step_bp)

        except Exception as e:
            logger.error(f"[{kf_symbol}] Exec Error: {e}")

    def _run_maker_loop(self, symbol: str, quantity: float, initial_mark: float, 
                        max_duration: int, interval: int, start_offset_bp: float, step_bp: float):
        """
        Dynamic Maker Loop.
        Refreshes every 'interval' seconds.
        Aggressiveness increases by 'step_bp' basis points each step.
        """
        side = "buy" if quantity > 0 else "sell"
        abs_qty = abs(quantity)
        
        specs = self.instrument_specs.get(symbol.lower())
        size_inc = specs['sizeStep'] if specs else 0.001
        price_inc = specs['tickSize'] if specs else 0.01

        steps = max_duration // interval
        order_id = None
        
        direction = 1 if side == "buy" else -1

        for i in range(steps + 1):
            try:
                # Get fresh mark price
                tickers = self.kf.get_tickers()
                curr_mark = 0.0
                for t in tickers.get("tickers", []):
                    if t["symbol"].lower() == symbol.lower():
                        curr_mark = float(t["markPrice"])
                        break
                if curr_mark == 0: curr_mark = initial_mark
                
                # Calculate Price
                current_aggression_bp = start_offset_bp + (i * step_bp)
                
                pct_change = current_aggression_bp * 0.0001
                
                if side == "buy":
                    final_limit = curr_mark * (1 + pct_change)
                else:
                    final_limit = curr_mark * (1 - pct_change)

                final_limit = self._round_to_step(final_limit, price_inc)
                final_size = self._round_to_step(abs_qty, size_inc)
                
                if order_id is None:
                    # Place New
                    resp = self.kf.send_order({
                        "orderType": "lmt", "symbol": symbol, "side": side,
                        "size": final_size, "limitPrice": final_limit
                    })
                    if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                         order_id = resp["sendStatus"]["order_id"]
                         logger.info(f"[{symbol}] Order Placed @ {final_limit} ({current_aggression_bp}bp)")
                    else:
                        logger.warning(f"[{symbol}] Order Failed: {resp}")
                else:
                    # Edit Existing
                    self.kf.edit_order({
                        "orderId": order_id, "limitPrice": final_limit,
                        "size": final_size, "symbol": symbol 
                    })
                    logger.info(f"[{symbol}] Adjusted @ {final_limit} ({current_aggression_bp}bp)")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"[{symbol}] Maker Loop Error: {e}")
                time.sleep(1) # Short sleep on error
        
        # End of Loop Cleanup
        if order_id:
            try:
                self.kf.cancel_order({"order_id": order_id, "symbol": symbol})
            except: pass

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()
