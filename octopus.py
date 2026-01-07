#!/usr/bin/env python3
"""
Octopus v2: Multi-Strategy Aggregator with Fast Parallel Execution
- Parallel order execution across all assets
- Timeframe-based deadlines (15m: 2min, 4h: 15min, 1d: 30min)
- Adaptive maker loops with market order fallback
- Daily training schedule (no wasteful retraining)
- Trigger at T+5 seconds for minimal latency
"""

import os
import sys
import time
import json
import math
import logging
import threading
import asyncio
import requests
import pandas as pd
import numpy as np
import random
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

# --- Local Imports ---
try:
    from kraken_futures import KrakenFuturesApi
    import stress_test
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}")
    sys.exit(1)

# --- Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KF_KEY = os.getenv("KRAKEN_FUTURES_KEY")
KF_SECRET = os.getenv("KRAKEN_FUTURES_SECRET")
GITHUB_PAT = os.getenv("PAT")

LEVERAGE = 2.0
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

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

# --- Deadline Configuration ---
DEADLINE_CONFIG = {
    '15m': 120,    # 2 minutes
    '30m': 120,    # 2 minutes
    '60m': 300,    # 5 minutes
    '240m': 900,   # 15 minutes
    '1d': 1800,    # 30 minutes
}

# --- Edit Schedules by Timeframe ---
EDIT_SCHEDULES = {
    '15m': {'max_steps': 5, 'edit_interval': 20, 'base_offset_pct': 0.015, 'decay_rate': 1.5},
    '30m': {'max_steps': 5, 'edit_interval': 20, 'base_offset_pct': 0.015, 'decay_rate': 1.5},
    '60m': {'max_steps': 7, 'edit_interval': 40, 'base_offset_pct': 0.012, 'decay_rate': 1.2},
    '240m': {'max_steps': 10, 'edit_interval': 80, 'base_offset_pct': 0.010, 'decay_rate': 1.0},
    '1d': {'max_steps': 10, 'edit_interval': 120, 'base_offset_pct': 0.008, 'decay_rate': 0.8},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("octopus_v2.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OctopusV2")

# --- Order Tracker ---
class OrderTracker:
    def __init__(self, asset: str, timeframe: str, order_id: str, placed_at: float, deadline: float):
        self.asset = asset
        self.timeframe = timeframe
        self.order_id = order_id
        self.placed_at = placed_at
        self.deadline = deadline
        self.filled = False
        self.fill_time = None
        self.fill_price = None
    
    def time_remaining(self) -> float:
        return self.deadline - time.time()
    
    def urgency_level(self) -> str:
        if self.filled:
            return 'FILLED'
        remaining_pct = self.time_remaining() / (self.deadline - self.placed_at)
        if remaining_pct < 0:
            return 'EXPIRED'
        if remaining_pct < 0.1:
            return 'CRITICAL'
        if remaining_pct < 0.25:
            return 'HIGH'
        if remaining_pct < 0.5:
            return 'MEDIUM'
        return 'LOW'

# --- Execution Metrics ---
class ExecutionMetrics:
    def __init__(self):
        self.metrics = []
    
    def record(self, asset: str, timeframe: str, signal_time: float, 
               order_time: float, fill_time: float, fill_price: float, 
               target_price: float, went_market: bool):
        self.metrics.append({
            'asset': asset,
            'timeframe': timeframe,
            'signal_latency': order_time - signal_time,
            'fill_latency': fill_time - order_time,
            'total_latency': fill_time - signal_time,
            'slippage_pct': abs(fill_price - target_price) / target_price * 100,
            'deadline_met': (fill_time - signal_time) < DEADLINE_CONFIG[timeframe],
            'went_market': went_market,
            'timestamp': datetime.fromtimestamp(signal_time, tz=timezone.utc)
        })
    
    def summary(self, timeframe: str = None):
        relevant = self.metrics if not timeframe else [m for m in self.metrics if m['timeframe'] == timeframe]
        if not relevant:
            return "No data"
        
        return {
            'count': len(relevant),
            'avg_total_latency': np.mean([m['total_latency'] for m in relevant]),
            'deadline_hit_rate': np.mean([m['deadline_met'] for m in relevant]) * 100,
            'market_order_rate': np.mean([m['went_market'] for m in relevant]) * 100,
            'avg_slippage_bps': np.mean([m['slippage_pct'] for m in relevant]) * 100,
        }

# --- Strategy Classes (unchanged) ---
class SubStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.bucket_count = config.get('bucket_count', 100)
        self.seq_len = config['seq_len']
        self.model_type = config['model_type']
        self.bucket_size = config.get('bucket_size', 1.0)
        if self.bucket_size <= 0:
            self.bucket_size = 1.0
        
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)
        self.all_vals = []
        self.all_changes = []

    def _get_bucket(self, price: float) -> int:
        bs = self.bucket_size
        if bs <= 0:
            bs = 1e-9
        if price >= 0:
            return int(price // bs)
        else:
            return int(price // bs) - 1

    def train(self, prices: List[float]):
        if not prices:
            return

        buckets = [self._get_bucket(p) for p in prices]
        split_idx = int(len(buckets) * 0.7)
        train_buckets = buckets[:split_idx]
        
        if len(train_buckets) < self.seq_len + 10:
            return 
            
        self.all_vals = list(set(train_buckets))
        self.all_changes = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))
        if not self.all_vals:
            self.all_vals = [0]
        if not self.all_changes:
            self.all_changes = [0]
        
        self.abs_map.clear()
        self.der_map.clear()
        
        for i in range(len(train_buckets) - self.seq_len):
            a_seq = tuple(train_buckets[i : i + self.seq_len])
            a_succ = train_buckets[i + self.seq_len]
            self.abs_map[a_seq][a_succ] += 1
            
            if self.seq_len > 1:
                d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
                d_succ = train_buckets[i + self.seq_len] - train_buckets[i + self.seq_len - 1]
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
            for c in der_cand.keys():
                poss.add(last_val + c)
            
            if not poss: 
                return random.choice(self.all_vals) if self.all_vals else last_val
            
            best, max_s = last_val, -1
            for v in poss:
                s = abs_cand[v] + der_cand[v - last_val]
                if s > max_s:
                    max_s, best = s, v
            return best
            
        return last_val

class EnsembleStrategy:
    def __init__(self, asset: str, timeframe: str, config_list: List[dict]):
        self.asset = asset
        self.timeframe = timeframe
        self.id = f"{asset}_{timeframe}"
        self.virtual_position = 0.0
        self.sub_strategies = [SubStrategy(cfg) for cfg in config_list]
        
    def train(self, prices: List[float]):
        for strat in self.sub_strategies:
            strat.train(prices)
            
    def predict(self, recent_prices: List[float]) -> int:
        if not recent_prices:
            return 0
        
        votes = []
        for strat in self.sub_strategies:
            pred_bucket = strat.get_prediction_value(recent_prices)
            current_bucket = strat._get_bucket(recent_prices[-1])
            diff = pred_bucket - current_bucket
            
            if diff > 0:
                votes.append(1)
            elif diff < 0:
                votes.append(-1)
            else:
                votes.append(0)
        
        up_votes = votes.count(1)
        down_votes = votes.count(-1)
        
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
        self.total_strategies_count = 0
        self.instrument_specs = {}
        self.metrics = ExecutionMetrics()
        self.data_lock = threading.Lock()
        self.last_training = None

    def initialize(self):
        logger.info("Initializing Octopus V2 (Parallel Execution)...")
        self._fetch_instrument_specs()
        
        logger.info("Executing Startup Stress Test...")
        try:
            stress_test.run_stress_test(self.kf, SYMBOL_MAP, LEVERAGE, REPO_OWNER, REPO_NAME, GITHUB_PAT)
        except Exception as e:
            logger.error(f"Stress test failed: {e}")

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
                logger.info(f"Loaded specs for {len(self.instrument_specs)} instruments")
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
                    acc = data.get('combined_accuracy', 0)
                    
                    if acc < 60.0:
                        continue
                    
                    strategy_union = data.get('strategy_union', [])
                    if not strategy_union:
                        continue

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
                    if not data or not isinstance(data, list):
                        break
                    all_candles.extend(data)
                    if len(data) < 1000:
                        break
                    current_start = int(data[-1][6]) + 1
                    time.sleep(0.05)
                
                with self.data_lock:
                    self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"Loaded {len(all_candles)} candles for {asset}")
            except Exception as e:
                logger.error(f"Data fetch error {asset}: {e}")

    def _train_all_strategies(self):
        logger.info("Training Ensemble Strategies...")
        for strat in self.strategies.values():
            with self.data_lock:
                raw = self.price_history[strat.asset]
            if not raw:
                continue
            prices = self._resample(raw, strat.timeframe)
            strat.train(prices)
        self.last_training = datetime.now(timezone.utc)
        logger.info(f"Training complete at {self.last_training}")

    def _resample(self, raw_data: List[Tuple[int, float]], timeframe: str) -> List[float]:
        if timeframe == "15m":
            return [x[1] for x in raw_data]
        df = pd.DataFrame(raw_data, columns=['ts', 'price'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        tf_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        target = tf_map.get(timeframe)
        if not target:
            return [x[1] for x in raw_data]
        return df['price'].resample(target).last().dropna().tolist()

    def _round_to_step(self, value: float, step: float) -> float:
        if step == 0:
            return value
        rounded = round(value / step) * step
        if isinstance(step, float) and "." in str(step):
            decimals = len(str(step).split(".")[1])
            rounded = round(rounded, decimals)
        elif isinstance(step, int) or step.is_integer():
            rounded = int(rounded)
        return rounded

    # --- Background Data Updater ---
    def _continuous_data_update(self):
        """Runs continuously to keep price_history fresh"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                # Wait until next 15m mark
                next_update = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
                if next_update <= now:
                    next_update += timedelta(minutes=15)
                
                sleep_duration = (next_update - now).total_seconds()
                time.sleep(sleep_duration)
                
                # Update data at :00:00
                self._update_all_data()
                
            except Exception as e:
                logger.error(f"Data update error: {e}")
                time.sleep(60)

    def _update_all_data(self):
        """Fetch latest candles for all assets"""
        now = datetime.now(timezone.utc)
        current_interval_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        limit_ts = int(current_interval_start.timestamp() * 1000)

        active_assets = set(s.asset for s in self.strategies.values())
        for asset in active_assets:
            try:
                params = {"symbol": asset, "interval": "15m", "limit": 5}
                r = requests.get("https://api.binance.com/api/v3/klines", params=params)
                data = r.json()
                
                with self.data_lock:
                    if not self.price_history[asset]:
                        continue
                    last_stored_ts = self.price_history[asset][-1][0]
                    
                    for candle in data:
                        open_ts = int(candle[0])
                        close_ts = int(candle[6])
                        price = float(candle[4])
                        if close_ts > last_stored_ts and open_ts < limit_ts:
                            self.price_history[asset].append((close_ts, price))
                            
                    # Keep last 200k candles
                    if len(self.price_history[asset]) > 200000:
                        self.price_history[asset] = self.price_history[asset][-200000:]
            except Exception as e:
                logger.error(f"Update failed for {asset}: {e}")

    # --- Daily Training Scheduler ---
    def _daily_training_schedule(self):
        """Retrain models once per day at 00:05 UTC"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute == 5:
                    logger.info("=== DAILY TRAINING INITIATED ===")
                    self._train_all_strategies()
                    logger.info("=== DAILY TRAINING COMPLETE ===")
                    time.sleep(3600)  # Sleep 1 hour to avoid retrigger
                time.sleep(30)  # Check every 30s
            except Exception as e:
                logger.error(f"Training scheduler error: {e}")
                time.sleep(300)

    # --- Main Run Loop ---
    def run(self):
        # Start background threads
        data_updater = threading.Thread(target=self._continuous_data_update, daemon=True)
        data_updater.start()
        
        trainer = threading.Thread(target=self._daily_training_schedule, daemon=True)
        trainer.start()
        
        logger.info("Background threads started. Entering main loop...")
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                minute = now.minute
                second = now.second
                hour = now.hour
                
                # Trigger at :00:05 of each 15m interval
                if minute % 15 == 0 and second == 5:
                    logger.info(f"=== TRIGGER: {hour:02}:{minute:02}:{second:02} ===")
                    
                    # Determine active timeframes
                    tfs_to_run = ["15m"]
                    if minute == 0 or minute == 30:
                        tfs_to_run.append("30m")
                    if minute == 0:
                        tfs_to_run.append("60m")
                        if hour % 4 == 0:
                            tfs_to_run.append("240m")
                        if hour == 0:
                            tfs_to_run.append("1d")
                    
                    # Execute all assets in parallel
                    asyncio.run(self._execute_all_parallel(tfs_to_run))
                    
                    time.sleep(10)  # Prevent double-trigger
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1)

    # --- Parallel Execution ---
    async def _execute_all_parallel(self, active_tfs: List[str]):
        """Execute all assets simultaneously"""
        signal_time = time.time()
        
        # Get account equity
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

        if self.total_strategies_count == 0:
            return
            
        unit_size_usd = (equity * LEVERAGE) / self.total_strategies_count
        logger.info(f"Equity: ${equity:.2f} | Unit: ${unit_size_usd:.2f} | Timeframes: {active_tfs}")

        # Generate signals for all strategies
        active_assets = set()
        for strat in self.strategies.values():
            if strat.timeframe in active_tfs:
                active_assets.add(strat.asset)
                with self.data_lock:
                    raw = self.price_history[strat.asset]
                prices = self._resample(raw, strat.timeframe)
                
                sig = strat.predict(prices)
                strat.virtual_position = sig * unit_size_usd
                logger.info(f"Strat {strat.id}: Signal={sig} Pos=${strat.virtual_position:.2f}")

        # Execute all assets in parallel
        tasks = []
        for asset in active_assets:
            task = asyncio.create_task(self._execute_asset_parallel(asset, signal_time))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Asset execution failed: {result}")

    async def _execute_asset_parallel(self, binance_asset: str, signal_time: float):
        """Execute orders for a single asset (runs in parallel with others)"""
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol:
            return

        # Get current position
        try:
            open_pos = self.kf.get_open_positions()
            current_pos_size = 0.0
            if "openPositions" in open_pos:
                for p in open_pos["openPositions"]:
                    if p["symbol"].lower() == kf_symbol.lower():
                        size = float(p["size"])
                        if p["side"] == "short":
                            size = -size
                        current_pos_size = size
                        break
        except Exception as e:
            logger.error(f"[{kf_symbol}] Failed to get position: {e}")
            return

        # Get mark price
        try:
            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0:
                logger.error(f"[{kf_symbol}] Mark price is 0")
                return
        except Exception as e:
            logger.error(f"[{kf_symbol}] Failed to get mark price: {e}")
            return

        # Split orders by timeframe
        orders_by_tf = {}
        for strat in self.strategies.values():
            if strat.asset == binance_asset and strat.virtual_position != 0:
                if strat.timeframe not in orders_by_tf:
                    orders_by_tf[strat.timeframe] = 0
                orders_by_tf[strat.timeframe] += strat.virtual_position

        if not orders_by_tf:
            return

        # Execute each timeframe order in parallel
        tasks = []
        for tf, position_usd in orders_by_tf.items():
            target_contracts = position_usd / mark_price
            delta = target_contracts - current_pos_size
            
            specs = self.instrument_specs.get(kf_symbol.lower())
            size_increment = specs['sizeStep'] if specs else 0.001
            check_qty = self._round_to_step(abs(delta), size_increment)
            
            if check_qty < size_increment:
                continue
            
            deadline = signal_time + DEADLINE_CONFIG[tf]
            
            task = asyncio.create_task(
                self._run_maker_loop_async(
                    kf_symbol, delta, mark_price, deadline, tf, signal_time
                )
            )
            tasks.append(task)
            
            # Update current position for next calculation
            current_pos_size += delta

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_maker_loop_async(self, symbol: str, quantity: float, 
                                     initial_mark: float, deadline: float, 
                                     timeframe: str, signal_time: float):
        """Adaptive maker loop with deadline awareness"""
        side = "buy" if quantity > 0 else "sell"
        abs_qty = abs(quantity)
        direction = 1 if side == "buy" else -1
        
        schedule = EDIT_SCHEDULES.get(timeframe, EDIT_SCHEDULES['15m'])
        specs = self.instrument_specs.get(symbol.lower())
        size_inc = specs['sizeStep'] if specs else 0.001
        price_inc = specs['tickSize'] if specs else 0.01
        
        order_id = None
        order_placed_time = None
        went_market = False
        
        try:
            for step in range(schedule['max_steps']):
                # Check if deadline exceeded
                time_remaining = deadline - time.time()
                if time_remaining <= 0:
                    logger.warning(f"[{symbol}] {timeframe} deadline exceeded, going MARKET")
                    await self._place_market_order(symbol, abs_qty, side)
                    went_market = True
                    break
                
                # Calculate urgency
                elapsed = time.time() - signal_time
                total_time = deadline - signal_time
                urgency_pct = elapsed / total_time
                
                # Adaptive offset based on urgency
                if urgency_pct > 0.9:
                    # Critical: go to market
                    logger.warning(f"[{symbol}] {timeframe} 90% of deadline, going MARKET")
                    if order_id:
                        await self._cancel_order_async(symbol, order_id)
                    await self._place_market_order(symbol, abs_qty, side)
                    went_market = True
                    break
                elif urgency_pct > 0.75:
                    # High urgency: very aggressive pricing
                    base_offset_pct = 0.001
                    decay_rate = 3.0
                elif urgency_pct > 0.5:
                    # Medium urgency: moderately aggressive
                    base_offset_pct = 0.005
                    decay_rate = 2.0
                else:
                    # Low urgency: use schedule defaults
                    base_offset_pct = schedule['base_offset_pct']
                    decay_rate = schedule['decay_rate']
                
                # Get current mark price
                try:
                    tickers = self.kf.get_tickers()
                    curr_mark = initial_mark
                    for t in tickers.get("tickers", []):
                        if t["symbol"].lower() == symbol.lower():
                            curr_mark = float(t["markPrice"])
                            break
                except:
                    curr_mark = initial_mark
                
                # Calculate limit price with exponential decay
                decay = math.exp(-step * decay_rate)
                offset = curr_mark * base_offset_pct * -direction * decay
                final_limit = self._round_to_step(curr_mark + offset, price_inc)
                final_size = self._round_to_step(abs_qty, size_inc)
                
                # Place or edit order
                if order_id is None:
                    # Initial order placement
                    logger.info(f"[{symbol}] {timeframe} Placing {side.upper()} {final_size} @ {final_limit} (deadline in {time_remaining:.0f}s)")
                    try:
                        resp = self.kf.send_order({
                            "orderType": "lmt",
                            "symbol": symbol,
                            "side": side,
                            "size": final_size,
                            "limitPrice": final_limit
                        })
                        
                        if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                            order_id = resp["sendStatus"]["order_id"]
                            order_placed_time = time.time()
                            
                            # Check if immediately filled
                            if resp["sendStatus"].get("status") == "filled":
                                fill_price = final_limit  # Approximate
                                self.metrics.record(symbol, timeframe, signal_time, 
                                                  order_placed_time, time.time(), 
                                                  fill_price, curr_mark, went_market)
                                logger.info(f"[{symbol}] {timeframe} FILLED immediately @ {fill_price}")
                                return
                        else:
                            logger.warning(f"[{symbol}] Order response unclear: {resp}")
                            break
                    except Exception as e:
                        logger.error(f"[{symbol}] Order placement failed: {e}")
                        break
                else:
                    # Edit existing order
                    logger.info(f"[{symbol}] {timeframe} Editing to {final_limit} (step {step+1}/{schedule['max_steps']})")
                    try:
                        edit_resp = self.kf.edit_order({
                            "orderId": order_id,
                            "limitPrice": final_limit,
                            "size": final_size,
                            "symbol": symbol
                        })
                        
                        # Check edit status
                        if "editStatus" in edit_resp:
                            status = edit_resp["editStatus"].get("status")
                            
                            if status == "filled":
                                # Order filled during edit attempt
                                fill_price = final_limit  # Approximate
                                self.metrics.record(symbol, timeframe, signal_time,
                                                  order_placed_time, time.time(),
                                                  fill_price, curr_mark, went_market)
                                logger.info(f"[{symbol}] {timeframe} FILLED @ {fill_price} | Latency: {time.time() - signal_time:.1f}s")
                                return
                            elif status == "orderForEditNotFound":
                                # Order already filled or cancelled
                                logger.info(f"[{symbol}] {timeframe} Order not found (likely filled)")
                                return
                            elif status == "invalidPrice":
                                logger.warning(f"[{symbol}] Invalid price {final_limit}, skipping edit")
                                
                    except Exception as e:
                        logger.error(f"[{symbol}] Edit failed: {e}")
                
                # Wait before next edit
                wait_time = schedule['edit_interval']
                if urgency_pct > 0.75:
                    wait_time = min(10, wait_time)  # Faster edits when urgent
                
                await asyncio.sleep(wait_time)
            
            # Max steps reached - cancel and go market if configured
            if order_id and not went_market:
                if timeframe in ['15m', '30m', '60m']:
                    logger.warning(f"[{symbol}] {timeframe} Max steps reached, going MARKET")
                    await self._cancel_order_async(symbol, order_id)
                    await self._place_market_order(symbol, abs_qty, side)
                    went_market = True
                else:
                    logger.info(f"[{symbol}] {timeframe} Max steps reached, cancelling")
                    await self._cancel_order_async(symbol, order_id)
                    
        except Exception as e:
            logger.error(f"[{symbol}] Maker loop error: {e}")
            if order_id:
                try:
                    await self._cancel_order_async(symbol, order_id)
                except:
                    pass

    async def _place_market_order(self, symbol: str, quantity: float, side: str):
        """Place market order (fallback)"""
        try:
            resp = self.kf.send_order({
                "orderType": "mkt",
                "symbol": symbol,
                "side": side,
                "size": quantity
            })
            logger.info(f"[{symbol}] MARKET order executed: {resp}")
            return resp
        except Exception as e:
            logger.error(f"[{symbol}] Market order failed: {e}")
            return None

    async def _cancel_order_async(self, symbol: str, order_id: str):
        """Cancel order"""
        try:
            pb = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            resp = self.kf.cancel_order({
                "order_id": order_id,
                "symbol": symbol,
                "processBefore": pb
            })
            logger.info(f"[{symbol}] Cancelled order {order_id}")
            return resp
        except Exception as e:
            logger.error(f"[{symbol}] Cancel failed: {e}")
            return None

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()