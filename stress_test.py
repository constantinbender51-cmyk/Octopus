import time
import json
import base64
import requests
import logging
from datetime import datetime, timezone, timedelta

# Configure Logger for Stress Test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StressTest")

def slow_print(msg):
    """Custom print with 0.1s delay for Railway logging limitations."""
    print(msg)
    time.sleep(0.1)

class StressTester:
    def __init__(self, api_interface, symbol_map, leverage, repo_owner, repo_name, pat):
        self.kf = api_interface
        self.symbol_map = symbol_map
        self.leverage = leverage
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.pat = pat
        self.logs = []
        self.equity = 0.0
        self.instrument_specs = {}

    def log(self, message):
        """Log to local stdout and append to internal log for upload."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        formatted_msg = f"[{timestamp}] {message}"
        slow_print(formatted_msg)
        self.logs.append(formatted_msg)

    def _fetch_instrument_specs(self):
        """Fetches lot size and tick size to ensure valid orders."""
        self.log("Fetching Instrument Specifications...")
        try:
            url = "https://futures.kraken.com/derivatives/api/v3/instruments"
            resp = requests.get(url, timeout=10).json()
            if "instruments" in resp:
                for inst in resp["instruments"]:
                    sym = inst["symbol"].lower()
                    self.instrument_specs[sym] = {
                        "lotSize": float(inst.get("lotSize", 1.0)),
                        "tickSize": float(inst.get("tickSize", 0.1)),
                        "contractSize": float(inst.get("contractSize", 1.0))
                    }
                self.log(f"Success: Loaded specs for {len(self.instrument_specs)} instruments.")
            else:
                self.log("Warning: Failed to parse instruments (no 'instruments' key).")
        except Exception as e:
            self.log(f"Error fetching specs: {e}")

    def run(self):
        self.log("--- STARTING STRESS TEST (Blind Execution Mode) ---")
        
        # 0. Load Specs
        self._fetch_instrument_specs()

        # 1. API Connectivity & Account Info
        self.log("1. Testing Account API & Fetching Equity...")
        try:
            accounts = self.kf.get_accounts()
            if "accounts" in accounts and "flex" in accounts["accounts"]:
                self.equity = float(accounts["accounts"]["flex"].get("marginEquity", 0))
            elif "accounts" in accounts:
                 first = list(accounts["accounts"].values())[0]
                 self.equity = float(first.get("marginEquity", 0))
            self.log(f"SUCCESS: Margin Equity retrieved: ${self.equity:.2f}")
        except Exception as e:
            self.log(f"CRITICAL: Failed to get accounts: {e}")
            return

        # 2. Fetch Open Data (Just logging, not using for logic)
        self.log("2. Logging Open Positions & Orders (Informational)...")
        try:
            open_positions = self.kf.get_open_positions()
            self.log(f"Open Positions: {len(open_positions.get('openPositions', []))}")
            open_orders = self.kf.get_open_orders()
            self.log(f"Open Orders: {len(open_orders.get('openOrders', []))}")
        except Exception as e:
            self.log(f"Error fetching open data: {e}")

        # 3. Order System Test
        self.log("3. Testing Order Execution (Place -> Edit -> Cancel)...")
        
        strat_count = len(self.symbol_map) 
        if strat_count == 0: strat_count = 1
        unit_size_usd = (self.equity * self.leverage) / strat_count
        self.log(f"Calculated Test Unit Size: ${unit_size_usd:.2f}")

        for binance_sym, kf_sym in self.symbol_map.items():
            self._test_symbol_execution(kf_sym, unit_size_usd)

        # 4. Upload Results
        self.log("4. Uploading Results to GitHub...")
        self._upload_to_github()
        self.log("--- STRESS TEST COMPLETE ---")

    def _test_symbol_execution(self, symbol, usd_size):
        self.log(f"--- Testing {symbol} ---")
        
        try:
            # A. Get Mark Price
            tickers = self.kf.get_tickers()
            mark_price = 0.0
            for t in tickers.get("tickers", []):
                if t["symbol"].lower() == symbol.lower():
                    mark_price = float(t["markPrice"])
                    break
            
            if mark_price == 0:
                self.log(f"SKIPPING: Could not get mark price for {symbol}")
                return

            # B. Calculate Size
            raw_size = usd_size / mark_price
            specs = self.instrument_specs.get(symbol.lower())
            
            if specs:
                lot_size = specs['lotSize']
                size = round(raw_size / lot_size) * lot_size
                if lot_size.is_integer():
                    size = int(size)
                else:
                    decimals = 0
                    if "." in str(lot_size):
                         decimals = len(str(lot_size).split(".")[1])
                    size = round(size, decimals)
            else:
                size = round(raw_size, 3)

            if size <= 0:
                self.log(f"SKIPPING: {symbol} (Size {size} too small)")
                return

            # C. Place 'Safe' Limit Order
            safe_limit_price = round(mark_price * 0.5, 2)
            self.log(f"Placing LIMIT BUY {size} @ {safe_limit_price}")
            
            order_payload = {
                "orderType": "lmt",
                "symbol": symbol,
                "side": "buy",
                "size": size,
                "limitPrice": safe_limit_price
            }
            
            resp = self.kf.send_order(order_payload)
            
            order_id = None
            if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                order_id = resp["sendStatus"]["order_id"]
                self.log(f"SUCCESS: Order Placed. ID: {order_id}")
            else:
                self.log(f"FAILURE: Order placement failed: {resp}")
                return

            # D. Edit Order (Blindly using ID)
            time.sleep(0.5) 
            new_price = round(safe_limit_price * 1.01, 2)
            self.log(f"Editing: Moving price to {new_price}...")
            
            edit_payload = {
                "orderId": order_id,
                "limitPrice": new_price,
                "size": size,
                "symbol": symbol 
            }
            
            edit_resp = self.kf.edit_order(edit_payload)
            
            if "editStatus" in edit_resp and edit_resp["editStatus"] == "edited":
                self.log("SUCCESS: Order Edited.")
            else:
                self.log(f"WARNING: Edit response uncertain: {edit_resp}")

            # E. Cancel Order (Blindly using ID)
            time.sleep(0.5)
            self.log(f"Cancelling Order {order_id}...")
            
            # --- FIX: Required Arguments for Cancellation ---
            # Format: 2023-11-08T19:56:35.441899Z
            process_before = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            cancel_payload = {
                "order_id": order_id, # User specified snake_case 'order_id'
                "symbol": symbol,
                "processBefore": process_before
            }
            
            try:
                cancel_resp = self.kf.cancel_order(cancel_payload)
                self.log(f"Cancel Response: {cancel_resp}")
            except Exception as e:
                self.log(f"Cancel Failed: {e}")

            # F. Cleanup Position (Just in case)
            self._check_and_close_position(symbol)

        except Exception as e:
            self.log(f"ERROR on {symbol}: {e}")

    def _check_and_close_position(self, symbol):
        """Checks for an open position and attempts to close it if found."""
        try:
            positions = self.kf.get_open_positions()
            size = 0.0
            if "openPositions" in positions:
                for p in positions["openPositions"]:
                    if p["symbol"].lower() == symbol.lower():
                        s = float(p["size"])
                        if p["side"] == "short": s = -s
                        size = s
                        break
            
            if size != 0:
                self.log(f"Open Position found for {symbol}: {size}. Closing...")
                side = "sell" if size > 0 else "buy"
                payload = {
                    "orderType": "mkt",
                    "symbol": symbol,
                    "side": side,
                    "size": abs(size),
                    "reduceOnly": True
                }
                self.kf.send_order(payload)
                
        except Exception as e:
            self.log(f"Error in Close Position logic: {e}")

    def _upload_to_github(self):
        if not self.pat:
            self.log("Skipping GitHub upload: No Token.")
            return

        file_path = "stress_test.txt"
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {self.pat}",
            "Accept": "application/vnd.github.v3+json"
        }

        content_str = "\n".join(self.logs)
        content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
        
        data = {
            "message": f"Stress Test Results {datetime.now().strftime('%Y-%m-%d')}",
            "content": content_b64
        }

        try:
            get_resp = requests.get(url, headers=headers)
            if get_resp.status_code == 200:
                data["sha"] = get_resp.json()["sha"]
                self.log("Existing file found. Updating...")
            else:
                self.log("No existing file. Creating new...")
        except:
            pass

        try:
            put_resp = requests.put(url, headers=headers, json=data)
            if put_resp.status_code in [200, 201]:
                self.log("SUCCESS: Results uploaded to GitHub.")
            else:
                self.log(f"FAILURE: Upload failed {put_resp.status_code} - {put_resp.text}")
        except Exception as e:
            self.log(f"FAILURE: Upload exception {e}")

def run_stress_test(kf_api, symbol_map, leverage, owner, repo, pat):
    tester = StressTester(kf_api, symbol_map, leverage, owner, repo, pat)
    tester.run()