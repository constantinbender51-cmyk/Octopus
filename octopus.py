#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
Updated to match 'Strategy Union' & 'Majority Vote' logic from Generator v58.
Includes STARTUP DIAGNOSTIC BACKTEST (2020-2026) followed by FRESH DATA LOAD for Live.
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
        self.last_predictions = {} 
        self.lock = threading.Lock()

    def evaluate(self, strat_id: str, current_price: float):
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

            # 2. Outcome is Directional
            is_correct = False
            
            if last_signal == 1: 
                if price_diff > 0: is_correct = True
                else: is_correct = False
            
            elif last_signal == -1: 
                if price_diff < 0: is_correct = True
                else: is_correct = False

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
        except Exception as e:
            logger.error(f"Failed to upload performance stats: {e}")

# --- Strategy Logic Classes ---

class SubStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.bucket_count = config.get('bucket_count', 100)
        self.seq_len = config['seq_len']
        self.model_type = config['model_type']
        self.bucket_size = config.get('bucket_size', 1.0)
        if self.bucket_size <= 0: self.bucket_size = 1.0
        
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

    def populate_maps(self, prices: List[float], train_limit_idx: int = None):
        """
        Populates the probability maps using the provided prices.
        If train_limit_idx is provided, ONLY uses data up to that index (for split verification).
        If None, uses ALL provided data (for live execution).
        """
        if not prices: return

        buckets = [self._get_bucket(p) for p in prices]
        
        # Determine the subset of data to learn from
        if train_limit_idx is not None:
            learn_buckets = buckets[:train_limit_idx]
        else:
            learn_buckets = buckets

        if len(learn_buckets) < self.seq_len + 10:
            return 
            
        self.all_vals = list(set(learn_buckets))
        self.all_changes = list(set(learn_buckets[j] - learn_buckets[j-1] for j in range(1, len(learn_buckets))))
        if not self.all_vals: self.all_vals = [0]
        if not self.all_changes: self.all_changes = [0]
        
        self.abs_map.clear()
        self.der_map.clear()
        
        for i in range(len(learn_buckets) - self.seq_len):
            a_seq = tuple(learn_buckets[i : i + self.seq_len])
            self.abs_map[a_seq][learn_buckets[i + self.seq_len]] += 1
            
            if self.seq_len > 1:
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
                d_succ = learn_buckets[i + self.seq_len] - learn_buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def get_prediction_value(self, recent_prices: List[float]) -> int:
        if len(recent_prices) < self.seq_len + 1:
            return self._get_bucket(recent_prices[-1]) if recent_prices else 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] 
        
        a_seq = tuple(window[1:]) 
        if self.seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        last_val = window[-1]
        
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
    def __init__(self, asset: str, timeframe: str, config_list: List[dict], expected_acc: float = 0.0, expected_trades: int = 0):
        self.asset = asset
        self.timeframe = timeframe
        self.id = f"{asset}_{timeframe}"
        self.virtual_position = 0.0
        
        # Validation benchmarks
        self.expected_accuracy = expected_acc
        self.expected_trades = expected_trades
        
        self.sub_strategies = [SubStrategy(cfg) for cfg in config_list]
        
        if self.sub_strategies:
            self.min_bucket_size = min(s.bucket_size for s in self.sub_strategies)
        else:
            self.min_bucket_size = 0.0
        
    def populate_maps(self, prices: List[float], train_limit_idx: int = None):
        for strat in self.sub_strategies:
            strat.populate_maps(prices, train_limit_idx)
            
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
        
        if up_votes > down_votes: return 1
        elif down_votes > up_votes: return -1
        return 0

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
        logger.info("Initializing Octopus (Backtest & Live Mode)...")
        self._fetch_instrument_specs()
        
        # 1. Startup Stress Test
        logger.info("Executing Startup Stress Test...")
        try:
            stress_test.run_stress_test(
                self.kf, SYMBOL_MAP, LEVERAGE, REPO_OWNER, REPO_NAME, GITHUB_PAT
            )
        except Exception as e:
            logger.error(f"Stress test failed/skipped: {e}")

        # 2. Load Strategy Configs
        self._load_strategies_from_github()
        
        # 3. VERIFICATION PHASE (2020-01-01 to 2026-01-01)
        logger.info("--- PHASE 1: DIAGNOSTIC BACKTEST (2020-2026) ---")
        self._fetch_verification_data()
        self.verify_strategies()
        
        # 4. LIVE PHASE (Fresh Data)
        logger.info("--- PHASE 2: PREPARING FOR LIVE EXECUTION ---")
        logger.info("Clearing historical verification data...")
        self.price_history.clear() # Dump the backtest data
        
        logger.info("Fetching fresh live data (Last 2000 candles)...")
        self._fetch_live_data() 
        
        logger.info("Populating strategy maps with fresh data...")
        self._populate_all_strategies_for_live()
        
        logger.info("Initialization Complete. Bot is ready.")

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
                    acc = data.get('combined_accuracy', 0)
                    trade_count = data.get('trade_count', 0)

                    if acc < 60.0:
                        continue
                    
                    strategy_union = data.get('strategy_union', [])
                    if not strategy_union: continue

                    ens = EnsembleStrategy(asset, tf, strategy_union, expected_acc=acc, expected_trades=trade_count)
                    self.strategies[ens.id] = ens
                    count += 1
            
            self.total_strategies_count = count
            logger.info(f"Loaded {count} Ensemble Strategies.")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")

    def _fetch_verification_data(self):
        """
        Fetches specific history window: 2020-01-01 to 2026-01-01
        Used strictly for verifying that the loaded strategies match their metadata.
        """
        start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        active_assets = set(s.asset for s in self.strategies.values())
        
        for asset in active_assets:
            try:
                url = "https://api.binance.com/api/v3/klines"
                all_candles = []
                current_start = start_ts
                
                while True:
                    if current_start >= end_ts: break
                    
                    params = {
                        "symbol": asset, 
                        "interval": "15m", 
                        "limit": 1000, 
                        "startTime": current_start,
                        "endTime": end_ts
                    }
                    r = requests.get(url, params=params)
                    data = r.json()
                    if not data or not isinstance(data, list) or len(data) == 0: break
                    
                    all_candles.extend(data)
                    
                    last_close = int(data[-1][6])
                    current_start = last_close + 1
                    time.sleep(0.05)
                
                self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"[Verification Data] {asset}: Loaded {len(all_candles)} candles (2020-2026).")
            except Exception as e:
                logger.error(f"Verification data fetch error {asset}: {e}")

    def _fetch_live_data(self):
        """
        Fetches the MOST RECENT data (Last 2000 15m candles).
        This ignores the 2020-2026 hard limit and gets 'Fresh' data for live execution.
        """
        active_assets = set(s.asset for s in self.strategies.values())
        
        for asset in active_assets:
            try:
                url = "https://api.binance.com/api/v3/klines"
                all_candles = []
                # Fetch last 2000 candles approx (2 calls of 1000)
                # We do this by not setting endTime, just standard backward fill or forward from a calculated start
                # Easier: Get latest, then get previous.
                
                # Method: standard walk forward from (Now - 2000 * 15min)
                start_ts = int((datetime.now(timezone.utc) - timedelta(minutes=15*2500)).timestamp() * 1000)
                
                current_start = start_ts
                while True:
                    params = {"symbol": asset, "interval": "15m", "limit": 1000, "startTime": current_start}
                    r = requests.get(url, params=params)
                    data = r.json()
                    if not data or not isinstance(data, list) or len(data) == 0: break
                    
                    all_candles.extend(data)
                    last_close = int(data[-1][6])
                    current_start = last_close + 1
                    
                    # Stop if we are at current time
                    if len(data) < 1000: break
                    time.sleep(0.05)
                
                self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"[Live Data] {asset}: Loaded {len(all_candles)} FRESH candles.")
            except Exception as e:
                logger.error(f"Live data fetch error {asset}: {e}")

    def _resample(self, raw_data: List[Tuple[int, float]], timeframe: str) -> List[float]:
        if timeframe == "15m": return [x[1] for x in raw_data]
        df = pd.DataFrame(raw_data, columns=['ts', 'price'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        tf_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        target = tf_map.get(timeframe)
        if not target: return [x[1] for x in raw_data]
        return df['price'].resample(target).last().dropna().tolist()

    # --- VERIFICATION LOGIC ---
    def verify_strategies(self):
        """
        Splits the 2020-2026 data: 70% Train (Populate Maps), 30% Test.
        """
        logger.info("Verifying strategies on 2020-2026 history (70/30 Split)...")
        
        for strat_id, strat in self.strategies.items():
            raw = self.price_history[strat.asset]
            if not raw: continue
            
            full_prices = self._resample(raw, strat.timeframe)
            if len(full_prices) < 100: continue

            # 70% Split Index
            split_idx = int(len(full_prices) * 0.7)
            
            # 1. Populate Maps using ONLY the first 70%
            # This simulates the state of the model at the start of the 'Test' phase
            strat.populate_maps(full_prices, train_limit_idx=split_idx)
            
            # 2. Walk-Forward Prediction on the remaining 30%
            correct = 0
            total_trades = 0
            
            # Predict for i, using info up to i-1
            for i in range(split_idx, len(full_prices)):
                current_price = full_prices[i]
                prev_price = full_prices[i-1]
                
                # Context window for prediction
                window_start = max(0, i - 50)
                recent_window = full_prices[window_start:i] 
                
                signal = strat.predict(recent_window)
                
                if signal == 0: continue
                
                diff = current_price - prev_price
                threshold = strat.min_bucket_size
                
                if abs(diff) < threshold:
                    continue # Flat/Ignored
                
                total_trades += 1
                is_correct = False
                if signal == 1 and diff > 0: is_correct = True
                elif signal == -1 and diff < 0: is_correct = True
                
                if is_correct: correct += 1
                
            calc_acc = (correct / total_trades * 100) if total_trades > 0 else 0.0
            exp_acc = strat.expected_accuracy
            exp_trades = strat.expected_trades
            
            logger.info(f"[{strat_id}] Test: Acc={calc_acc:.2f}% (Exp: {exp_acc}%) | Trades={total_trades} (Exp: {exp_trades})")
            
            fail = False
            if abs(calc_acc - exp_acc) > 5.0:
                logger.error(f"[{strat_id}] ACCURACY MISMATCH! {calc_acc:.2f} vs {exp_acc}")
                fail = True
            
            if exp_trades > 0:
                trade_dev = abs(total_trades - exp_trades) / exp_trades
                if trade_dev > 0.05:
                    logger.error(f"[{strat_id}] TRADE COUNT MISMATCH! {total_trades} vs {exp_trades}")
                    fail = True
            
            if fail:
                logger.critical(f"STOPPING: Strategy {strat_id} failed verification.")
                sys.exit(1)

        logger.info("--- VERIFICATION PASSED ---")

    def _populate_all_strategies_for_live(self):
        """
        Populates maps using ALL available fresh data (no split).
        """
        for strat in self.strategies.values():
            raw = self.price_history[strat.asset]
            if not raw: continue
            prices = self._resample(raw, strat.timeframe)
            # Use entire dataset for map population
            strat.populate_maps(prices, train_limit_idx=None)

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
            exec_duration, exec_interval, start_offset_bp, step_bp, mode_name = 300, 10, -5, 0.5, "MARATHONER"
        else:
            exec_duration, exec_interval, start_offset_bp, step_bp, mode_name = 60, 5, 0, 1.0, "SPRINTER"

        logger.info(f"Execution Mode: {mode_name}")
        active_assets = set()
        
        def calc_signal(strat):
            if strat.timeframe in active_tfs:
                active_assets.add(strat.asset)
                raw = self.price_history[strat.asset]
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
            self.executor.submit(self._execute_single_asset_logic, asset, exec_duration, exec_interval, start_offset_bp, step_bp)

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
            if check_qty < size_increment: return
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
                final_limit = curr_mark * (1 + pct_change) if side == "buy" else curr_mark * (1 - pct_change)
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
