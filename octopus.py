#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
Updated to match 'Strategy Union' & 'Majority Vote' logic from Generator v58.
- Uses Fixed Bucket Size from JSON.
- FIXED TRAINING HORIZON: "Now" for training defined as 2026-01-01.
- Trains on 70% of (2020 -> 2026-01-01).
- Verifies on 30% of (2020 -> 2026-01-01).
- PARALLEL EXECUTION (ThreadPoolExecutor).
- Precise Timing (Execute at XX:XX:05).
- Dynamic Execution Window (Sprinter vs Marathoner).
- NO Retraining during live run.
- LOGGING: Enhanced decision logging.
- PERFORMANCE TRACKING: Logs accuracy to GitHub (performance.json).
- STARTUP VALIDATION: Backtests the 30% holdout set to verify accuracy matches JSON specs.
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

# --- DEFINING 'NOW' FOR MODEL CONSISTENCY ---
# This ensures that regardless of the actual date, the model trains/verifies
# on the exact same dataset (2020-01-01 to 2026-01-01).
TRAINING_END_DATE = "2026-01-01"

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

# --- Performance Tracking Class ---

class PerformanceTracker:
    def __init__(self):
        self.stats = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
        self.last_predictions = {} # {strat_id: {"signal": int, "price": float, "ts": int, "threshold": float}}
        self.lock = threading.Lock()

    def evaluate(self, strat_id: str, current_price: float):
        """
        Compares the LAST prediction (if exists) against the CURRENT price.
        """
        with self.lock:
            if strat_id not in self.last_predictions:
                return

            last = self.last_predictions[strat_id]
            last_signal = last["signal"]
            last_price = last["price"]
            threshold = last.get("threshold", 0.0)
            
            del self.last_predictions[strat_id]

            if last_signal == 0:
                return 

            price_diff = current_price - last_price

            # 1. Check if outcome is FLAT (Small movement)
            if abs(price_diff) < threshold:
                return 

            # 2. Outcome is Directional (Big movement)
            is_correct = False
            
            if last_signal == 1: 
                if price_diff > 0: is_correct = True
            elif last_signal == -1: 
                if price_diff < 0: is_correct = True

            # Update Stats
            self.stats[strat_id]["total"] += 1
            if is_correct:
                self.stats[strat_id]["correct"] += 1
            
            total = self.stats[strat_id]["total"]
            corr = self.stats[strat_id]["correct"]
            self.stats[strat_id]["accuracy"] = round((corr / total) * 100, 2)

    def record_prediction(self, strat_id: str, signal: int, current_price: float, threshold: float):
        with self.lock:
            self.last_predictions[strat_id] = {
                "signal": signal,
                "price": current_price,
                "ts": int(time.time()),
                "threshold": threshold
            }

    def upload_to_github(self):
        if not GITHUB_PAT: return
        try:
            content_str = json.dumps(self.stats, indent=2)
            content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
            url = f"{GITHUB_API_URL}performance.json"
            headers = {"Authorization": f"Bearer {GITHUB_PAT}"}
            sha = None
            try:
                get_resp = requests.get(url, headers=headers)
                if get_resp.status_code == 200:
                    sha = get_resp.json().get("sha")
            except: pass

            data = {
                "message": f"Update performance stats {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64
            }
            if sha: data["sha"] = sha
            requests.put(url, headers=headers, json=data)
            logger.info("Performance stats uploaded to GitHub.")
        except Exception as e:
            logger.error(f"Failed to upload performance stats: {e}")

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
        self.bucket_size = config.get('bucket_size', 1.0)
        if self.bucket_size <= 0: self.bucket_size = 1e-9
        
        # Runtime State
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)
        self.all_vals = []
        self.all_changes = []
        self.trained_buckets = [] # For backtesting verification

    def _get_bucket(self, price: float) -> int:
        bs = self.bucket_size
        if price >= 0:
            return int(price // bs)
        else:
            return int(price // bs) - 1

    def train(self, prices: List[float]):
        """
        Trains the model using ONLY the first 70% of the provided data.
        NOTE: 'prices' passed here should already be truncated to TRAINING_END_DATE.
        """
        if not prices: return

        # 1. Bucketize ALL prices
        self.trained_buckets = [self._get_bucket(p) for p in prices]
        
        # 2. Apply 70% Split (Match app.py logic)
        split_idx = int(len(self.trained_buckets) * 0.7)
        train_buckets = self.trained_buckets[:split_idx]
        
        if len(train_buckets) < self.seq_len + 10: return 
            
        # 3. Build Maps on TRAIN set only
        self.all_vals = list(set(train_buckets))
        self.all_changes = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))
        if not self.all_vals: self.all_vals = [0]
        if not self.all_changes: self.all_changes = [0]
        
        self.abs_map.clear()
        self.der_map.clear()
        
        for i in range(len(train_buckets) - self.seq_len):
            a_seq = tuple(train_buckets[i : i + self.seq_len])
            self.abs_map[a_seq][train_buckets[i + self.seq_len]] += 1
            
            if self.seq_len > 1:
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
                d_succ = train_buckets[i + self.seq_len] - train_buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def get_prediction_value(self, recent_prices: List[float]) -> int:
        """Live prediction using recent price list."""
        if len(recent_prices) < self.seq_len + 1:
            return self._get_bucket(recent_prices[-1]) if recent_prices else 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] 
        
        return self._predict_internal(window)

    def _predict_internal(self, window_buckets: List[int]) -> int:
        """Internal helper for prediction based on bucket window."""
        a_seq = tuple(window_buckets[1:]) 
        last_val = window_buckets[-1]
        
        if self.seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()

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
    """
    def __init__(self, asset: str, timeframe: str, config_list: List[dict], expected_acc: float):
        self.asset = asset
        self.timeframe = timeframe
        self.id = f"{asset}_{timeframe}"
        self.virtual_position = 0.0
        self.expected_accuracy = expected_acc
        self.sub_strategies = [SubStrategy(cfg) for cfg in config_list]
        
        if self.sub_strategies:
            self.min_bucket_size = min(s.bucket_size for s in self.sub_strategies)
        else:
            self.min_bucket_size = 0.0
        
    def train(self, prices: List[float]):
        for strat in self.sub_strategies:
            strat.train(prices)
            
    def predict(self, recent_prices: List[float]) -> int:
        if not recent_prices: return 0
        votes = []
        for strat in self.sub_strategies:
            pred_bucket = strat.get_prediction_value(recent_prices)
            current_bucket = strat._get_bucket(recent_prices[-1])
            diff = pred_bucket - current_bucket
            if diff > 0: votes.append(1)
            elif diff < 0: votes.append(-1)
            else: votes.append(0)
            
        up_votes = votes.count(1)
        down_votes = votes.count(-1)
        
        signal = 0
        if up_votes > down_votes: signal = 1
        elif down_votes > up_votes: signal = -1
        
        logger.info(f"[{self.id}] Decision: +{up_votes} / -{down_votes} => Signal: {signal}")
        return signal

    def run_backtest_verification(self) -> Tuple[float, float, str]:
        """
        Replicates 'run_portfolio_analysis' on the 30% holdout set.
        Returns: (Calculated Accuracy, Expected Accuracy, Status Message)
        """
        if not self.sub_strategies: return 0.0, self.expected_accuracy, "No Sub-Strategies"
        
        # 1. Determine common test range
        # We assume all sub-strategies were fed the same 'prices' list, so lengths match.
        sample_strat = self.sub_strategies[0]
        total_len = len(sample_strat.trained_buckets)
        split_idx = int(total_len * 0.7)
        max_seq_len = max(s.seq_len for s in self.sub_strategies)
        
        start_test_idx = split_idx
        total_test_len = total_len - start_test_idx - max_seq_len
        
        if total_test_len < 10:
            return 0.0, self.expected_accuracy, "Insufficient Data for Backtest"

        unique_correct = 0
        unique_total = 0
        
        # 2. Scan the test set
        for i in range(total_test_len):
            curr_raw_idx = start_test_idx + i
            active_directions = []
            
            # Check each model
            for strat in self.sub_strategies:
                seq_len = strat.seq_len
                buckets = strat.trained_buckets
                
                window_input = buckets[curr_raw_idx : curr_raw_idx + seq_len + 1] # +1 for internal logic
                a_seq = window_input[:-1] # The sequence
                last_val = a_seq[-1]
                actual_val = buckets[curr_raw_idx + seq_len] # The truth
                pred_val = strat._predict_internal(window_input) # Uses window logic
                
                # Determine model's view of reality
                diff = actual_val - last_val
                model_actual_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
                
                pred_diff = pred_val - last_val
                
                if pred_diff != 0:
                    direction = 1 if pred_diff > 0 else -1
                    is_correct = (direction == model_actual_dir)
                    is_flat = (model_actual_dir == 0)
                    
                    active_directions.append({
                        "dir": direction,
                        "is_correct": is_correct,
                        "is_flat": is_flat
                    })
            
            # Aggregate
            if not active_directions: continue
            
            dirs = [x['dir'] for x in active_directions]
            up = dirs.count(1)
            down = dirs.count(-1)
            
            final_dir = 0
            if up > down: final_dir = 1
            elif down > up: final_dir = -1
            else: continue # Tie
            
            winning_voters = [x for x in active_directions if x['dir'] == final_dir]
            
            # Filter Logic from app.py
            if all(x['is_flat'] for x in winning_voters):
                continue
                
            unique_total += 1
            if any(x['is_correct'] for x in winning_voters):
                unique_correct += 1
                
        calculated_acc = (unique_correct / unique_total * 100) if unique_total > 0 else 0.0
        
        msg = f"Calc: {calculated_acc:.2f}% vs Spec: {self.expected_accuracy:.2f}% (Trades: {unique_total})"
        return calculated_acc, self.expected_accuracy, msg

# --- Main Octopus Engine ---

class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.strategies: Dict[str, EnsembleStrategy] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=None)
        self.total_strategies_count = 0
        self.instrument_specs = {}
        self.tracker = PerformanceTracker()

    def initialize(self):
        logger.info(f"Initializing Octopus with FIXED Training Horizon: {TRAINING_END_DATE}")
        self._fetch_instrument_specs()
        
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
        
        # --- NEW VALIDATION STEP ---
        logger.info("Running Startup Backtest Verification...")
        self.verify_strategies_on_startup()
        
        logger.info("Initialization Complete. Strategies trained and verified.")

    def verify_strategies_on_startup(self):
        """
        Runs the backtest logic on all strategies.
        If any strategy deviates > 5% from its spec, triggers ALARM (Exit).
        """
        alarm_triggered = False
        
        for s_id, strat in self.strategies.items():
            calc_acc, spec_acc, msg = strat.run_backtest_verification()
            diff = abs(calc_acc - spec_acc)
            
            if diff > 5.0:
                logger.error(f"[ALARM] Strategy {s_id} FAILED validation! {msg}")
                alarm_triggered = True
            else:
                logger.info(f"[PASS] Strategy {s_id}: {msg}")
                
        if alarm_triggered:
            logger.critical("One or more strategies failed accuracy validation (>5% deviation). Exiting safely.")
            sys.exit(1)

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
                    acc = data.get('combined_accuracy', 0) # Expected accuracy
                    
                    if acc < 60.0:
                        logger.warning(f"Skipping {asset} {tf} (Acc {acc:.2f}%)")
                        continue
                    
                    strategy_union = data.get('strategy_union', [])
                    if not strategy_union: continue

                    ens = EnsembleStrategy(asset, tf, strategy_union, acc)
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
        """
        Trains strategies on a FIXED dataset (2020-01-01 to TRAINING_END_DATE).
        This prevents data drift and ensures '70% split' is always the same.
        """
        logger.info("Training Ensemble Strategies (One-time Init)...")
        
        # Convert fixed end date to timestamp
        end_dt = datetime.strptime(TRAINING_END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ts = int(end_dt.timestamp() * 1000)
        
        for strat in self.strategies.values():
            raw = self.price_history[strat.asset]
            if not raw: continue
            
            # TRUNCATE DATA strictly for training
            training_data_raw = [x for x in raw if x[0] <= end_ts]
            
            # If we don't have enough data to reach the end date, warn but proceed with what we have
            if not training_data_raw:
                logger.warning(f"No data found for {strat.asset} before {TRAINING_END_DATE}")
                continue
                
            logger.info(f"Training {strat.id} on {len(training_data_raw)} candles (<= {TRAINING_END_DATE})")
            
            prices = self._resample(training_data_raw, strat.timeframe)
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
            if now.minute % 15 == 0 and 5 <= now.second < 10:
                logger.info(f"--- Trigger: {now.strftime('%H:%M:%S')} ---")
                
                tfs_to_run = ["15m"]
                if now.minute == 0 or now.minute == 30: tfs_to_run.append("30m")
                if now.minute == 0:
                    tfs_to_run.append("60m")
                    if now.hour % 4 == 0: tfs_to_run.append("240m")
                    if now.hour == 0: tfs_to_run.append("1d")
                
                self._update_all_data_parallel()
                self._process_strategies_parallel(tfs_to_run)
                self.executor.submit(self.tracker.upload_to_github)
                time.sleep(50)
            time.sleep(0.1)

    def _update_single_asset(self, asset: str, limit_ts: int):
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
                if close_ts > last_stored_ts and open_ts < limit_ts:
                    self.price_history[asset].append((close_ts, price))
            
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
        
        for f in futures: f.result()
        logger.info("Data update complete.")

    def _process_strategies_parallel(self, active_tfs: List[str]):
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
                logger.error("Equity 0. Aborting.")
                return
        except Exception as e:
            logger.error(f"Account fetch failed: {e}")
            return

        if self.total_strategies_count == 0: return
        unit_size_usd = (equity * LEVERAGE) / self.total_strategies_count
        logger.info(f"Equity: ${equity:.2f} | Unit: ${unit_size_usd:.2f} | TFs: {active_tfs}")

        is_marathon = any(tf in ["60m", "240m", "1d"] for tf in active_tfs)
        if is_marathon:
            exec_duration = 300
            exec_interval = 10
            start_offset_bp = -5 
            step_bp = 0.5 
            mode_name = "MARATHONER"
        else:
            exec_duration = 60
            exec_interval = 5
            start_offset_bp = 0 
            step_bp = 1.0 
            mode_name = "SPRINTER"

        logger.info(f"Execution Mode: {mode_name}")
        active_assets = set()
        
        def calc_signal(strat):
            if strat.timeframe in active_tfs:
                active_assets.add(strat.asset)
                raw = self.price_history[strat.asset]
                # RESAMPLE FULL HISTORY (Including live data > 2026-01-01)
                prices = self._resample(raw, strat.timeframe)
                current_price = prices[-1] if prices else 0.0

                self.tracker.evaluate(strat.id, current_price)
                strat.virtual_position = 0.0
                sig = strat.predict(prices)
                self.tracker.record_prediction(strat.id, sig, current_price, strat.min_bucket_size)

                strat.virtual_position = sig * unit_size_usd
                logger.info(f"Strat {strat.id}: Signal {sig} | Alloc: ${strat.virtual_position:.2f}")

        f_sigs = [self.executor.submit(calc_signal, s) for s in self.strategies.values()]
        for f in f_sigs: f.result()

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
        side = "buy" if quantity > 0 else "sell"
        abs_qty = abs(quantity)
        specs = self.instrument_specs.get(symbol.lower())
        size_inc = specs['sizeStep'] if specs else 0.001
        price_inc = specs['tickSize'] if specs else 0.01

        steps = max_duration // interval
        order_id = None
        
        for i in range(steps + 1):
            try:
                tickers = self.kf.get_tickers()
                curr_mark = 0.0
                for t in tickers.get("tickers", []):
                    if t["symbol"].lower() == symbol.lower():
                        curr_mark = float(t["markPrice"])
                        break
                if curr_mark == 0: curr_mark = initial_mark
                
                current_aggression_bp = start_offset_bp + (i * step_bp)
                pct_change = current_aggression_bp * 0.0001
                
                if side == "buy": final_limit = curr_mark * (1 + pct_change)
                else: final_limit = curr_mark * (1 - pct_change)

                final_limit = self._round_to_step(final_limit, price_inc)
                final_size = self._round_to_step(abs_qty, size_inc)
                
                if order_id is None:
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
                    self.kf.edit_order({
                        "orderId": order_id, "limitPrice": final_limit,
                        "size": final_size, "symbol": symbol 
                    })
                    logger.info(f"[{symbol}] Adjusted @ {final_limit} ({current_aggression_bp}bp)")
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"[{symbol}] Maker Loop Error: {e}")
                time.sleep(1)
        
        if order_id:
            try:
                self.kf.cancel_order({"order_id": order_id, "symbol": symbol})
            except: pass

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()
