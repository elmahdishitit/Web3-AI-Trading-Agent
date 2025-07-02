import json
import sys
import argparse
import os
from web3 import Web3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Connect to BASE mainnet
w3 = Web3(Web3.HTTPProvider(config.BASE_RPC_URL))

# Check connection
if not w3.is_connected():
    raise Exception("Failed to connect to Base mainnet RPC")

# Define constants
USDC_DECIMALS = 6
ETH_DECIMALS = 18
Q96 = 2**96  # Q96 precision used by Uniswap v4

# Use minimal ABI with only the slot0 function
STATE_VIEW_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "poolId", "type": "bytes32"}],
        "name": "getSlot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint24", "name": "protocolFee", "type": "uint24"},
            {"internalType": "uint24", "name": "lpFee", "type": "uint24"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Convert sqrtPriceX96 to price
def sqrt_price_x96_to_price(sqrt_price_x96, token0_decimals, token1_decimals):
    """
    Convert sqrtPriceX96 to human-readable price
    For an ETH-USDC pool where token0 is ETH and token1 is USDC
    """
    # sqrtPriceX96 is the sqrt(price) * 2^96 for token1/token0
    price = (sqrt_price_x96 / Q96) ** 2
    
    # Adjust for decimals
    decimal_adjustment = 10 ** (token1_decimals - token0_decimals)
    price = price / decimal_adjustment
    
    return price

# Convert tick to price
def tick_to_price(tick, token0_decimals, token1_decimals):
    """
    Convert tick to price (token1/token0)
    """
    # Tick is a log base 1.0001 of the price (token1/token0)
    price = 1.0001 ** tick
    
    # Adjust for decimals
    decimal_adjustment = 10 ** (token1_decimals - token0_decimals)
    price = price / decimal_adjustment
    
    return price

# Safe contract call with fallback
def safe_contract_call(contract_call, default_value=None, error_msg=None):
    """Make a contract call with error handling and return a default value if it fails"""
    try:
        return contract_call()
    except Exception as e:
        error_detail = str(e)
        if error_msg:
            print(f"Warning: {error_msg}: {error_detail}")
        return default_value

# Main function to get pool data
def get_pool_data():
    # Create contract instances
    try:
        state_view = w3.eth.contract(address=config.UNISWAP_V4_STATE_VIEW, abi=STATE_VIEW_ABI)
    except Exception as e:
        return {"error": f"Failed to create contract instances: {str(e)}. Check your contract addresses."}
    
    # Convert the pool ID from hex string to bytes
    try:
        pool_id = Web3.to_bytes(hexstr=config.UNISWAP_V4_ETH_USDC_POOL)
    except Exception as e:
        return {"error": f"Invalid pool ID format: {str(e)}"}
    
    # Initialize result structure with just pool_id and price_data
    result = {
        "pool_id": config.UNISWAP_V4_ETH_USDC_POOL,
        "price_data": {}
    }
    
    # Try to get slot0 data
    slot0_data = safe_contract_call(
        lambda: state_view.functions.getSlot0(pool_id).call(),
        None,
        "Failed to get slot0 data"
    )
    
    if slot0_data is None:
        return {"error": "Failed to fetch essential pool data (slot0)"}
    
    # Unpack slot0 data
    sqrt_price_x96, tick, _, _ = slot0_data
    
    # Calculate prices
    price_from_sqrt = sqrt_price_x96_to_price(sqrt_price_x96, ETH_DECIMALS, USDC_DECIMALS)
    price_from_tick = tick_to_price(tick, ETH_DECIMALS, USDC_DECIMALS)
    
    # Update result with price data
    result["price_data"] = {
        "eth_price_in_usdc_from_sqrt": float(price_from_sqrt),
        "eth_price_in_usdc_from_tick": float(price_from_tick),
        "raw_sqrt_price_x96": sqrt_price_x96,
        "tick": tick
    }
    
    return result

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch Uniswap v4 ETH-USDC pool price from Base mainnet")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug information")
    args = parser.parse_args()
    
    print("Fetching Uniswap v4 ETH-USDC pool data from Base mainnet...")
    pool_data = get_pool_data()
    
    # Handle errors
    if "error" in pool_data:
        print(f"Error: {pool_data['error']}")
        if args.debug:
            print("\nDebug Tips:")
            print("1. Check your config.py file for correct contract addresses")
            print("2. Verify that ETH-USDC pool exists on Base mainnet")
            print("3. Ensure your RPC endpoint for Base is working")
            print(f"   Current RPC URL: {config.BASE_RPC_URL}")
        sys.exit(1)
    
    # Display only the minimal output
    print(json.dumps(pool_data, indent=2)) 