import time
import json
import base64
import requests
import logging
from datetime import datetime, timezone

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

    def log(self, message):
        """Log to local stdout and append to internal log for upload."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        formatted_msg = f"[{timestamp}] {message}"
        slow_print(formatted_msg)
        self.logs.append(formatted_msg)

    def run(self):
        self.log("--- STARTING STRESS TEST ---")
        
        # 1. API Connectivity & Account Info
        self.log("1. Testing Account API & Fetching Equity...")
        try:
            accounts = self.kf.get_accounts()
            # self.log(f"Raw Account Data: {json.dumps(accounts)}")
            
            # Extract Margin Equity
            if "accounts" in accounts and "flex" in accounts["accounts"]:
                self.equity = float(accounts["accounts"]["flex"].get("marginEquity", 0))
            elif "accounts" in accounts:
                 # Fallback
                 first = list(accounts["accounts"].values())[0]
                 self.equity = float(first.get("marginEquity", 0))
            
            self.log(f"SUCCESS: Margin Equity retrieved: ${self.equity:.2f}")
            
        except Exception as e:
            self.log(f"CRITICAL: Failed to get accounts: {e}")
            return

        # 2. Fetch Open Data
        self.log("2. Fetching Open Positions & Orders...")
        try:
            open_positions = self.kf.get_open_positions()
            self.log(f"Open Positions Response: {len(open_positions.get('openPositions', []))} positions found.")
            
            open_orders = self.kf.get_open_orders()
            self.log(f"Open Orders Response: {len(open_orders.get('openOrders', []))} orders found.")
        except Exception as e:
            self.log(f"Error fetching open data: {e}")

        # 3. Order System Test
        self.log("3. Testing Order Execution (Place & Cancel) for all symbols...")
        
        # Calculate roughly the size we would use in live trading
        # (Equity * Leverage) / Count
        strat_count = len(self.symbol_map) # Approximation
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

            # B. Calculate Size in Contracts
            # Assuming linear for simplicity, check if inverse logic needed in real prod
            size = usd_size / mark_price
            if size < 0.0001: size = 0.0001 # Min size safety
            size = round(size, 4)

            # C. Place 'Safe' Limit Order
            # Place buy limit 50% BELOW market to ensure it is PLACED but NOT FILLED
            # This tests the API functionality without taking market risk during a test.
            safe_limit_price = round(mark_price * 0.5, 2)
            
            self.log(f"Placing LIMIT BUY {size} @ {safe_limit_price} (Mark: {mark_price})")
            
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
                status = resp["sendStatus"]["status"]
                self.log(f"API Response: Order Sent. ID: {order_id} | Status: {status}")
            else:
                self.log(f"FAILURE: Order placement failed: {resp}")
                return

            # D. Verify 'Placed' Status
            time.sleep(0.5) # Wait for engine
            check = self.kf.get_order(order_id)
            # Response handling for get_order depends on exact API return (list or dict)
            # Assuming standard behavior, often returns a list of orders queried
            found_status = "unknown"
            if isinstance(check, dict) and "orders" in check:
                 # Some endpoints return { orders: [ ... ] }
                 found_status = check["orders"][0].get("status")
            elif isinstance(check, list) and len(check) > 0:
                 found_status = check[0].get("status")
            
            self.log(f"Verification: Order Status is '{found_status}'")

            # E. Cancel Order ("Close an Order")
            self.log(f"Cancelling Order {order_id}...")
            cancel_resp = self.kf.cancel_order({"orderId": order_id})
            self.log(f"Cancel Response: {cancel_resp}")

            # F. Close Position (If exists)
            # We check if we accidentally have a position (or if one existed before test)
            # and simulate the "Close Position" logic
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
                # To close, we place a market order in opposite direction
                side = "sell" if size > 0 else "buy"
                payload = {
                    "orderType": "mkt",
                    "symbol": symbol,
                    "side": side,
                    "size": abs(size),
                    "reduceOnly": True
                }
                resp = self.kf.send_order(payload)
                self.log(f"Close Position Response: {resp}")
            else:
                self.log(f"No open position for {symbol} to close.")
                
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

        # Check if file exists to get SHA (needed for update)
        try:
            get_resp = requests.get(url, headers=headers)
            if get_resp.status_code == 200:
                data["sha"] = get_resp.json()["sha"]
                self.log("Existing file found. Updating...")
            else:
                self.log("No existing file. Creating new...")
        except:
            pass

        # PUT Request
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
