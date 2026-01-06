#!/usr/bin/env python3
"""
Octopus: Multi-Strategy Aggregator & Execution Engine for Kraken Futures.
Updated to use Strategy Union (Voting Consensus) from top configurations.
DEBUG MODE: Uploading API logs to GitHub.
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
LEVERAGE = 2.0  # Global leverage setting
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

# Reverse map for logging
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}

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
    Represents a composite strategy (Asset + Timeframe) containing multiple sub-models (Top 3).
    Implements voting logic to determine the final signal.
    """
    def __init__(self, asset: str, timeframe: str, configs: list):
        self.asset = asset
        self.timeframe = timeframe
        self.configs = configs 
        self.virtual_position = 0.0
        self.models = [] 
        for c in configs:
            self.models.append({
                'config': c,
                'abs_map': defaultdict(Counter),
                'der_map': defaultdict(Counter),
                'all_vals': [],
                'all_changes': []
            })
        self.id = f"{asset}_{timeframe}"

    def train(self, prices: List[float]):
        for model in self.models:
            c = model['config']
            bs = c['bucket_size']
            sl = c['seq_len']
            buckets = [self._get_bucket(p, bs) for p in prices]
            if len(buckets) < sl + 10: continue
            model['all_vals'] = list(set(buckets))
            model['all_changes'] = list(set(buckets[j] - buckets[j-1] for j in range(1, len(buckets))))
            model['abs_map'].clear()
            model['der_map'].clear()
            for i in range(len(buckets) - sl):
                a_seq = tuple(buckets[i : i + sl])
                a_succ = buckets[i + sl]
                model['abs_map'][a_seq][a_succ] += 1
                if i > 0:
                    d_seq = tuple(buckets[j] - buckets[j-1] for j in range(i, i + sl))
                    d_succ = buckets[i + sl] - buckets[i + sl - 1]
                    model['der_map'][d_seq][d_succ] += 1

    def predict(self, recent_prices: List[float]) -> int:
        signals = []
        for model in self.models:
            c = model['config']
            bs = c['bucket_size']
            sl = c['seq_len']
            m_type = c['model_type']
            if len(recent_prices) < sl + 1:
                signals.append(0)
                continue
            buckets = [self._get_bucket(p, bs) for p in recent_prices]
            window = buckets[-(sl + 1):] 
            a_seq = tuple(window[1:]) 
            d_seq = tuple(window[j] - window[j-1] for j in range(1, len(window)))
            last_val = window[-1]
            pred_bucket = last_val
            
            if m_type == "Absolute":
                if a_seq in model['abs_map']:
                    pred_bucket = model['abs_map'][a_seq].most_common(1)[0][0]
            elif m_type == "Derivative":
                if d_seq in model['der_map']:
                    change = model['der_map'][d_seq].most_common(1)[0][0]
                    pred_bucket = last_val + change
            elif m_type == "Combined":
                abs_cand = model['abs_map'].get(a_seq, Counter())
                der_cand = model['der_map'].get(d_seq, Counter())
                poss = set(abs_cand.keys())
                for ch in der_cand.keys(): poss.add(last_val + ch)
                best, max_s = last_val, -1
                for v in poss:
                    s = abs_cand[v] + der_cand[v - last_val]
                    if s > max_s: max_s, best = s, v
                pred_bucket = best

            if pred_bucket > last_val: signals.append(1)
            elif pred_bucket < last_val: signals.append(-1)
            else: signals.append(0)

        has_up = 1 in signals
        has_down = -1 in signals
        if has_up and has_down: return 0 
        if has_up: return 1
        if has_down: return -1
        return 0

    def _get_bucket(self, price: float, size: float) -> int:
        if size <= 0: size = 1e-9
        if price >= 0: return int(price // size)
        else: return int(price // size) - 1


class Octopus:
    def __init__(self):
        self.kf = KrakenFuturesApi(KF_KEY, KF_SECRET)
        self.strategies: Dict[str, Strategy] = {}
        self.price_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.total_strategies_count = 0
        self.instrument_specs = {}
        
        # Lock for GitHub API uploads to prevent race conditions during threading
        self.gh_lock = threading.Lock()

    # --- Initialization ---
    def initialize(self):
        logger.info("Initializing Octopus...")
        self._fetch_instrument_specs()
        
        logger.info("Executing Startup Stress Test...")
        try:
            stress_test.run_stress_test(
                self.kf, SYMBOL_MAP, LEVERAGE, REPO_OWNER, REPO_NAME, GITHUB_PAT
            )
            logger.info("Stress Test Completed. Proceeding with Normal Boot.")
        except Exception as e:
            logger.error(f"Stress test failed or skipped: {e}")

        self._load_strategies_from_github()
        self._fetch_initial_data()
        self._train_all_strategies()
        logger.info("Initialization Complete. Entering Wait Loop.")

    def _fetch_instrument_specs(self):
        try:
            url = "https://futures.kraken.com/derivatives/api/v3/instruments"
            resp = requests.get(url).json()
            if "instruments" in resp:
                for inst in resp["instruments"]:
                    sym = inst["symbol"].lower()
                    self.instrument_specs[sym] = {
                        "lotSize": float(inst.get("lotSize", 1.0)),
                        "tickSize": float(inst.get("tickSize", 0.1)),
                        "contractSize": float(inst.get("contractSize", 1.0))
                    }
                logger.info(f"Loaded specs for {len(self.instrument_specs)} instruments.")
            else:
                logger.error("Failed to load instrument specs.")
        except Exception as e:
            logger.error(f"Error fetching instrument specs: {e}")

    def _load_strategies_from_github(self):
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
                    content_resp = requests.get(f['download_url'])
                    data = content_resp.json()
                    asset = data.get('asset')
                    tf = data.get('timeframe')
                    acc = data.get('combined_accuracy', 0)
                    if acc < 60.0:
                        logger.warning(f"Skipping strategy {asset} {tf} (Accuracy {acc:.2f}% < 60%)")
                        continue
                    
                    top_strats = data.get('strategy_union', [])
                    if not top_strats: continue

                    s = Strategy(asset, tf, top_strats)
                    self.strategies[s.id] = s
                    count += 1
            
            self.total_strategies_count = count
            logger.info(f"Loaded {count} composite strategies from GitHub (Filtered > 60%).")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")

    def _fetch_initial_data(self):
        active_assets = set(s.asset for s in self.strategies.values())
        logger.info(f"Fetching historical data for {len(active_assets)} assets (Since 2020)...")
        start_timestamp_2020 = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            try:
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
                    if not data or not isinstance(data, list): break
                    all_candles.extend(data)
                    if len(data) < 1000: break
                    last_close_time = int(data[-1][6])
                    current_start = last_close_time + 1
                    time.sleep(0.1)
                
                self.price_history[asset] = [(int(x[6]), float(x[4])) for x in all_candles]
                logger.info(f"Loaded {len(all_candles)} candles for {asset} since 2020")
                
            except Exception as e:
                logger.error(f"Error fetching data for {asset}: {e}")

    def _train_all_strategies(self):
        logger.info("Training strategies...")
        for s_id, strat in self.strategies.items():
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
        resampled = df['price'].resample(target).last().dropna()
        return resampled.tolist()

    def _round_to_step(self, value: float, step: float) -> float:
        if step == 0: return value
        rounded = round(value / step) * step
        if isinstance(step, float) and "." in str(step):
            decimals = len(str(step).split(".")[1])
            rounded = round(rounded, decimals)
        elif isinstance(step, int) or step.is_integer():
            rounded = int(rounded)
        return rounded

    # --- Debugging: GitHub Log Upload ---
    def _upload_api_log(self, action: str, payload: dict, response: Any):
        """Uploads API call details to orders_api.txt in the repo."""
        if not GITHUB_PAT: return
        
        # Prepare Log Entry
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = (
            f"--- {timestamp} ---\n"
            f"ACTION: {action}\n"
            f"PAYLOAD: {json.dumps(payload, default=str)}\n"
            f"RESPONSE: {json.dumps(response, default=str)}\n"
            f"--------------------------\n"
        )
        
        target_file = "orders_api.txt"
        url = f"{GITHUB_API_URL}{target_file}"
        headers = {"Authorization": f"Bearer {GITHUB_PAT}"}

        # Use lock to prevent race conditions from multiple threads
        with self.gh_lock:
            try:
                # 1. Get existing file (to get SHA and content)
                r = requests.get(url, headers=headers)
                sha = ""
                content = ""
                
                if r.status_code == 200:
                    data = r.json()
                    sha = data.get('sha')
                    content_b64 = data.get('content', '')
                    if content_b64:
                        content = base64.b64decode(content_b64).decode('utf-8', errors='ignore')

                # 2. Append new log
                # Limit file size to last ~100kb to prevent API errors on huge files
                if len(content) > 100000:
                    content = content[-100000:]
                    
                new_content = content + log_entry
                encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')

                # 3. Update file
                update_data = {
                    "message": f"Log API Call: {action}",
                    "content": encoded_content
                }
                if sha:
                    update_data["sha"] = sha
                
                put_resp = requests.put(url, headers=headers, json=update_data)
                if put_resp.status_code not in [200, 201]:
                    logger.error(f"Failed to upload log to GitHub: {put_resp.text}")

            except Exception as e:
                logger.error(f"Error logging to GitHub: {e}")

    # --- Core Loop Logic ---

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

                logger.info(f"Running strategies for: {tfs_to_run}")
                self._process_strategies(tfs_to_run)
                time.sleep(60)
            
            time.sleep(1)

    def _update_all_data(self):
        active_assets = set(s.asset for s in self.strategies.values())
        for asset in active_assets:
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {"symbol": asset, "interval": "15m", "limit": 5} 
                r = requests.get(url, params=params)
                data = r.json()
                last_stored_ts = self.price_history[asset][-1][0]
                for candle in data:
                    ts = int(candle[6])
                    price = float(candle[4])
                    if ts > last_stored_ts:
                        self.price_history[asset].append((ts, price))
                if len(self.price_history[asset]) > 200000:
                    self.price_history[asset] = self.price_history[asset][-200000:]
            except Exception as e:
                logger.error(f"Update failed for {asset}: {e}")

    def _process_strategies(self, active_tfs: List[str]):
        try:
            acc = self.kf.get_accounts()
            if "flex" in acc.get("accounts", {}):
                equity = float(acc["accounts"]["flex"].get("marginEquity", 0))
            else:
                first_acc = list(acc.get("accounts", {}).values())[0]
                equity = float(first_acc.get("marginEquity", 0))
            if equity <= 0: return
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")
            return

        if self.total_strategies_count == 0: return
        unit_size_usd = (equity * LEVERAGE) / self.total_strategies_count
        logger.info(f"Equity: ${equity:.2f} | Unit Size: ${unit_size_usd:.2f}")

        active_assets = set()
        for s in self.strategies.values():
            if s.timeframe in active_tfs:
                active_assets.add(s.asset)
                raw = self.price_history[s.asset]
                prices = self._resample(raw, s.timeframe)
                s.train(prices)
                sig = s.predict(prices)
                s.virtual_position = sig * unit_size_usd
                logger.info(f"Strategy {s.id}: Signal {sig} -> VirtPos ${s.virtual_position:.2f}")

        for asset in active_assets:
            self.executor.submit(self._execute_asset_logic, asset)

    def _execute_asset_logic(self, binance_asset: str):
        kf_symbol = SYMBOL_MAP.get(binance_asset)
        if not kf_symbol: return

        net_target_usd = 0.0
        for s in self.strategies.values():
            if s.asset == binance_asset:
                net_target_usd += s.virtual_position

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
        except Exception as e:
            logger.error(f"[{kf_symbol}] Failed to get positions: {e}")
            return

        try:
            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == kf_symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0: raise ValueError("Mark price 0")
            
            target_contracts = net_target_usd / mark_price
            delta = target_contracts - current_pos_size
            
            logger.info(f"[{kf_symbol}] Net Target: {target_contracts:.6f} | Curr: {current_pos_size} | Delta: {delta:.6f}")

            specs = self.instrument_specs.get(kf_symbol.lower())
            size_increment = specs['tickSize'] if specs else 0.001
            check_qty = self._round_to_step(abs(delta), size_increment)

            if check_qty < size_increment:
                logger.info(f"[{kf_symbol}] Delta rounds to 0 (Rounded: {check_qty} < SizeInc: {size_increment}). Skipping.")
                return

            self._run_maker_loop(kf_symbol, delta, mark_price)

        except Exception as e:
            logger.error(f"[{kf_symbol}] Execution Logic Failed: {e}")

    def _run_maker_loop(self, symbol: str, quantity: float, initial_mark: float):
        side = "buy" if quantity > 0 else "sell"
        abs_qty_raw = abs(quantity)
        decay_steps = 10 
        order_id = None
        
        specs = self.instrument_specs.get(symbol.lower())
        size_increment = specs['tickSize'] if specs else 0.001
        price_increment = specs['lotSize'] if specs else 0.01

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
                decay_factor = math.exp(-i * 0.5)
                offset = curr_mark * 0.01 * -direction * decay_factor
                raw_limit_price = curr_mark + offset
                
                final_limit_price = self._round_to_step(raw_limit_price, price_increment)
                final_size = self._round_to_step(abs_qty_raw, size_increment)
                
                logger.info(f"[{symbol}] Maker Iter {i}: {side.upper()} {final_size} @ {final_limit_price} (Mark: {curr_mark})")

                # --- DEBUGGING: SEND ORDER + LOG ---
                if order_id is None:
                    payload = {
                        "orderType": "lmt",
                        "symbol": symbol,
                        "side": side,
                        "size": final_size,
                        "limitPrice": final_limit_price
                    }
                    resp = self.kf.send_order(payload)
                    self._upload_api_log("SEND_ORDER", payload, resp) # <--- LOG TO GITHUB

                    if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                         order_id = resp["sendStatus"]["order_id"]
                    else:
                         logger.error(f"[{symbol}] Order fail: {resp}")
                         break 
                else:
                    payload = {
                        "orderId": order_id,
                        "limitPrice": final_limit_price,
                        "size": final_size,
                        "symbol": symbol 
                    }
                    resp = self.kf.edit_order(payload)
                    self._upload_api_log("EDIT_ORDER", payload, resp) # <--- LOG TO GITHUB

                time.sleep(30)
                
            except Exception as e:
                logger.error(f"[{symbol}] Maker Loop Error: {e}")
                time.sleep(5)
        
        # Timeout - Cancel
        if order_id:
            try:
                logger.info(f"[{symbol}] Timeout. Cancelling.")
                process_before = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                payload = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "processBefore": process_before
                }
                resp = self.kf.cancel_order(payload)
                self._upload_api_log("CANCEL_ORDER", payload, resp) # <--- LOG TO GITHUB

            except Exception as e:
                logger.error(f"[{symbol}] Cancel failed: {e}")

if __name__ == "__main__":
    bot = Octopus()
    bot.initialize()
    bot.run()