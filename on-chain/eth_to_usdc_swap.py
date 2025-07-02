import os
import sys
from decimal import Decimal, getcontext
from web3 import Web3
from web3.types import Wei
from eth_abi import encode
from uniswap_universal_router_decoder import RouterCodec
import json

# Add the parent directory to the Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_RPC_URL, 
    PRIVATE_KEY, 
    UNISWAP_V4_UNIVERSAL_ROUTER,
    BASE_USDC_ADDRESS,
    ETH_ADDRESS,
    WETH_ADDRESS,
    UNISWAP_V4_ETH_USDC_POOL,
    UNISWAP_V4_POSITION_MANAGER,
    UNISWAP_V4_STATE_VIEW,
    UNISWAP_V4_POOL_MANAGER
)

# Set high precision for Decimal calculations
getcontext().prec = 40

# Define token decimals
ETH_DECIMALS = 18
USDC_DECIMALS = 6

# ABI for the position manager (minimal interface for what we need)
POSITION_MANAGER_ABI = [
    {
        "inputs": [{"internalType": "bytes25", "name": "id", "type": "bytes25"}],
        "name": "poolKeys",
        "outputs": [
            {"internalType": "address", "name": "currency0", "type": "address"},
            {"internalType": "address", "name": "currency1", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
            {"internalType": "int24", "name": "tickSpacing", "type": "int24"},
            {"internalType": "address", "name": "hooks", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Basic ABI for the pool manager to get swap events
POOL_MANAGER_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
            {"indexed": False, "internalType": "int128", "name": "amount0", "type": "int128"},
            {"indexed": False, "internalType": "int128", "name": "amount1", "type": "int128"},
            {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
            {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
            {"indexed": False, "internalType": "uint24", "name": "fee", "type": "uint24"}
        ],
        "name": "Swap",
        "type": "event"
    }
]

# ABI for the state view contract
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

def calculate_price_from_sqrt_price_x96(sqrt_price_x96):
    """
    Calculate price from sqrtPriceX96 value using Uniswap V4's formula
    Price = (sqrtPriceX96 / 2^96) ^ 2
    For ETH/USDC pair, need to account for decimal differences (ETH: 18, USDC: 6)
    """
    try:
        # Convert to Decimal for precise calculation
        sqrt_price = Decimal(sqrt_price_x96)
        two_96 = Decimal(2) ** Decimal(96)
        
        # Calculate sqrtPrice / 2^96
        sqrt_price_adjusted = sqrt_price / two_96
        
        # Square it to get the price
        price = sqrt_price_adjusted * sqrt_price_adjusted
        
        # Convert to proper decimals (ETH/USDC)
        # ETH (18 decimals) to USDC (6 decimals) = multiply by 10^12
        price = price * Decimal(10 ** 12)
        
        return float(price)
    except Exception as e:
        print(f"Error calculating price: {e}")
        return None

def tick_to_price(tick):
    """
    Convert tick to price using the formula:
    price = 1.0001^tick
    For ETH/USDC, we need to account for decimal differences
    """
    try:
        tick_multiplier = Decimal('1.0001') ** Decimal(tick)
        # Adjust for decimals (ETH 18 - USDC 6 = 12)
        return float(tick_multiplier * Decimal(10 ** 12))
    except Exception as e:
        print(f"Error calculating price from tick: {e}")
        return None

def calculate_amounts_from_liquidity(liquidity, sqrt_price_x96, current_tick):
    """
    Calculate token amounts from liquidity using Uniswap V4 formulas
    """
    try:
        liquidity = Decimal(liquidity)
        sqrt_price = Decimal(sqrt_price_x96)
        two_96 = Decimal(2) ** Decimal(96)
        
        # Calculate ETH amount (token0)
        eth_amount = (liquidity * two_96) / sqrt_price / Decimal(10 ** 18)
        
        # Calculate USDC amount (token1)
        usdc_amount = (liquidity * sqrt_price) / two_96 / Decimal(10 ** 6)
        
        return float(eth_amount), float(usdc_amount)
    except Exception as e:
        print(f"Error calculating amounts: {e}")
        return 0, 0

def fetch_pool_stats(w3, pool_id, pool_manager_address, minutes=10):
    """
    Fetch detailed statistics for the pool similar to fetch_pool_stats.py
    """
    try:
        # Create pool manager contract instance
        pool_manager = w3.eth.contract(
            address=w3.to_checksum_address(pool_manager_address),
            abi=POOL_MANAGER_ABI
        )
        
        # Calculate block range (approximately 2 seconds per block on Base)
        current_block = w3.eth.block_number
        blocks_ago = minutes * 60 // 2  # n minutes in blocks
        start_block = max(0, current_block - blocks_ago)
        
        # Get all swap events from the pool manager
        events = w3.eth.get_logs({
            'address': w3.to_checksum_address(pool_manager_address),
            'fromBlock': start_block,
            'toBlock': current_block,
            'topics': [
                w3.keccak(text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").hex(),
                "0x" + pool_id[2:].ljust(64, '0')  # First indexed param is poolId, need to pad to 32 bytes
            ]
        })
        
        # Parse events
        swaps = []
        for event in events:
            try:
                parsed_event = pool_manager.events.Swap().process_log(event)
                swaps.append(parsed_event)
            except:
                continue
        
        if not swaps:
            return {
                "success": False,
                "error": "No swap events found for this pool"
            }
        
        # Calculate volumes and fees
        volume_eth = 0
        volume_usdc = 0
        fees_eth = 0
        fees_usdc = 0
        
        for swap in swaps:
            try:
                # Get amounts from the swap event
                amount0 = abs(int(swap['args']['amount0'])) / 1e18  # ETH has 18 decimals
                amount1 = abs(int(swap['args']['amount1'])) / 1e6   # USDC has 6 decimals
                
                # Add to volume
                volume_eth += amount0
                volume_usdc += amount1
                
                # Calculate fees (fee is usually in the event)
                fee_rate = int(swap['args']['fee']) / 1_000_000  # Fee is in hundredths of a bip
                fees_eth += amount0 * fee_rate
                fees_usdc += amount1 * fee_rate
            except:
                continue
                
        # Get latest state from the last swap
        last_swap = swaps[-1]
        liquidity = int(last_swap['args']['liquidity'])
        sqrt_price_x96 = int(last_swap['args']['sqrtPriceX96'])
        tick = int(last_swap['args']['tick'])
        
        # Calculate ETH price
        eth_price = calculate_price_from_sqrt_price_x96(sqrt_price_x96)
        if not eth_price:
            eth_price = tick_to_price(tick)
        
        # Calculate amounts in the pool
        eth_amount, usdc_amount = calculate_amounts_from_liquidity(liquidity, sqrt_price_x96, tick)
        
        # Calculate TVL
        tvl = (eth_amount * eth_price) + usdc_amount
        
        # Calculate average price from volume
        avg_price = volume_usdc / volume_eth if volume_eth > 0 else 0
        
        return {
            "success": True,
            "pool_id": pool_id,
            "last_swap": {
                "tick": tick,
                "sqrtPriceX96": sqrt_price_x96,
                "liquidity": liquidity,
                "eth_price": eth_price
            },
            "pool_balances": {
                "eth": eth_amount,
                "usdc": usdc_amount,
                "tvl": tvl
            },
            "volume": {
                "eth": volume_eth,
                "usdc": volume_usdc,
                "avg_price": avg_price
            },
            "fees": {
                "eth": fees_eth,
                "usdc": fees_usdc
            },
            "stats": {
                "num_swaps": len(swaps),
                "time_period_minutes": minutes
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Connect to BASE
w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
if not w3.is_connected():
    raise Exception("Failed to connect to BASE network")

# Setup account
account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"Account: {account.address}")

# Check balances before swap
eth_balance = w3.eth.get_balance(account.address)
print(f"ETH balance: {eth_balance / 10**18:.8f} ETH")

# For USDC balance, we need the contract
usdc_abi = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]

# Convert addresses to checksum format
USDC_ADDRESS_CS = Web3.to_checksum_address(BASE_USDC_ADDRESS)
ETH_ADDRESS_CS = Web3.to_checksum_address(ETH_ADDRESS)
UNIVERSAL_ROUTER_CS = Web3.to_checksum_address(UNISWAP_V4_UNIVERSAL_ROUTER)
POSITION_MANAGER_CS = Web3.to_checksum_address(UNISWAP_V4_POSITION_MANAGER)
STATE_VIEW_CS = Web3.to_checksum_address(UNISWAP_V4_STATE_VIEW)
POOL_MANAGER_CS = Web3.to_checksum_address(UNISWAP_V4_POOL_MANAGER)

usdc_contract = w3.eth.contract(address=USDC_ADDRESS_CS, abi=usdc_abi)
usdc_balance = usdc_contract.functions.balanceOf(account.address).call()
print(f"USDC balance: {usdc_balance / 10**6:.6f} USDC")  # USDC has 6 decimals

print("-" * 50)

# Setup Universal Router Codec
codec = RouterCodec(w3)

# Amount to swap (0.00001 ETH)
amount_in = Wei(int(0.00001 * 10**18))
print(f"Swapping {amount_in / 10**18:.8f} ETH to USDC via Universal Router")

# Create the correct v4 pool key for ETH/USDC
eth_usdc_pool_key = codec.encode.v4_pool_key(
    ETH_ADDRESS_CS,          # Numerically smaller
    USDC_ADDRESS_CS,         # Numerically larger
    500,                     # 0.05% fee tier
    10                       # Tick spacing
)

# Print the pool key parameters for reference
print(f"Pool key parameters: {eth_usdc_pool_key}")

# Extract pool key parameters
token0 = eth_usdc_pool_key['currency_0']
token1 = eth_usdc_pool_key['currency_1']
fee = eth_usdc_pool_key['fee']
tick_spacing = eth_usdc_pool_key['tick_spacing']
hooks = eth_usdc_pool_key['hooks']

# UniswapV4 Pool ID calculation
pool_init_code = encode(
    ['address', 'address', 'uint24', 'int24', 'address'],
    [token0, token1, fee, tick_spacing, hooks]
)

# Calculate the pool ID (keccak256 hash of the encoded parameters)
# Relevant Uniswap V4 pool ID calculation logic in PoolId.sol:
# function toId(PoolKey memory poolKey) internal pure returns (PoolId poolId) {
#     assembly {
#         poolId := keccak256(poolKey, 0xa0)
#     }
# }
calculated_pool_id = Web3.solidity_keccak(['bytes'], [pool_init_code]).hex()

# Print the calculated pool ID
print(f"Calculated Pool ID: 0x{calculated_pool_id}")

# Query the position manager to get the actual pool parameters
try:
    # Convert pool ID to bytes25 by taking first 25 bytes
    pool_id_bytes25 = bytes.fromhex(calculated_pool_id)[:25]
    position_manager = w3.eth.contract(address=POSITION_MANAGER_CS, abi=POSITION_MANAGER_ABI)
    
    # Get actual pool parameters from the chain
    pool_data = position_manager.functions.poolKeys(pool_id_bytes25).call()
    
    print("\nPool data retrieved from the chain:")
    print(f"  Currency0: {pool_data[0]}")
    print(f"  Currency1: {pool_data[1]}")
    print(f"  Fee: {pool_data[2] / 10000}%")
    print(f"  Tick Spacing: {pool_data[3]}")
    print(f"  Hooks: {pool_data[4]}")
    
except Exception as e:
    print(f"Error retrieving pool data: {e}")

# Use the calculated pool ID for the transaction
pool_id = f"0x{calculated_pool_id}"
print(f"\nUsing pool ID for transaction: {pool_id}")
print(f"Uniswap info: https://info.uniswap.org/#/base/pools/{pool_id}")

# Build the transaction
try:
    # We'll use swap_exact_in_single for a direct ETH to USDC swap
    transaction_params = (
        codec
        .encode
        .chain()
        .v4_swap()
        .swap_exact_in_single(
            pool_key=eth_usdc_pool_key,
            zero_for_one=True,  # ETH is token0, USDC is token1
            amount_in=amount_in,
            amount_out_min=Wei(0),  # Setting min amount to 0 - in production set a reasonable slippage
        )
        .take_all(USDC_ADDRESS_CS, Wei(0))
        .settle_all(ETH_ADDRESS_CS, amount_in)
        .build_v4_swap()
        .build_transaction(
            account.address,
            amount_in,
            ur_address=UNIVERSAL_ROUTER_CS,
            block_identifier=w3.eth.block_number
        )
    )
    
    # Make sure we have all required transaction parameters
    if 'chainId' not in transaction_params:
        transaction_params['chainId'] = w3.eth.chain_id
    
    if 'nonce' not in transaction_params:
        transaction_params['nonce'] = w3.eth.get_transaction_count(account.address)
    
    if 'gasPrice' not in transaction_params and 'maxFeePerGas' not in transaction_params:
        transaction_params['gasPrice'] = w3.eth.gas_price
    
    # Sign and send the transaction
    try:
        signed_txn = w3.eth.account.sign_transaction(transaction_params, account.key)
        
        # Get the raw transaction bytes
        if hasattr(signed_txn, 'rawTransaction'):
            raw_tx = signed_txn.rawTransaction
        elif hasattr(signed_txn, 'raw_transaction'):
            raw_tx = signed_txn.raw_transaction
        else:
            raise Exception("Could not find raw transaction data")
        
        # Send transaction
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        print(f"Transaction sent: {w3.to_hex(tx_hash)}")
        print("Waiting for confirmation...")
        
        # Wait for the transaction to be mined
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt['status'] == 1:
            print("Transaction successful!")
        else:
            print("Transaction failed!")
            print(receipt)
        
        print("-" * 50)
        
        # Check balances after swap
        eth_balance_after = w3.eth.get_balance(account.address)
        usdc_balance_after = usdc_contract.functions.balanceOf(account.address).call()
        
        eth_spent = (eth_balance - eth_balance_after) / 10**18
        usdc_received = (usdc_balance_after - usdc_balance) / 10**6
        
        print(f"ETH balance: {eth_balance_after / 10**18:.8f} ETH")
        print(f"USDC balance: {usdc_balance_after / 10**6:.6f} USDC")
        print("-" * 50)
        
        # Get transaction summary
        print(f"Summary:")
        print(f"ETH spent (swap only): {amount_in / 10**18:.8f} ETH")
        print(f"USDC received: {usdc_received:.6f} USDC")
        
        # Calculate pure swap rate (using original amount_in)
        pure_swap_rate = (usdc_received / (amount_in / 10**18))
        
        # Calculate effective rate including gas
        effective_rate = usdc_received / eth_spent
        
        print(f"Pure swap rate: 1 ETH = {pure_swap_rate:.2f} USDC")
        
        # Fetch pool data using the simpler approach
        print("\nFetching pool data...")
        try:
            # Create state view contract instance
            state_view = w3.eth.contract(address=STATE_VIEW_CS, abi=STATE_VIEW_ABI)
            
            # Get slot0 data
            slot0_data = state_view.functions.getSlot0(pool_id).call()
            sqrt_price_x96, tick, protocol_fee, lp_fee = slot0_data
            
            # Calculate prices
            # For ETH/USDC: token0 (ETH) has 18 decimals, token1 (USDC) has 6 decimals
            price_from_sqrt = (sqrt_price_x96 / (2**96)) ** 2 / (10 ** (USDC_DECIMALS - ETH_DECIMALS))
            price_from_tick = 1.0001 ** tick / (10 ** (USDC_DECIMALS - ETH_DECIMALS))
            
            # Format output similar to fetch_pool_data.py
            pool_data = {
                "pool_id": pool_id,
                "price_data": {
                    "eth_price_in_usdc_from_sqrt": float(price_from_sqrt),
                    "eth_price_in_usdc_from_tick": float(price_from_tick),
                    "raw_sqrt_price_x96": sqrt_price_x96,
                    "tick": tick,
                    "protocol_fee": protocol_fee,
                    "lp_fee": lp_fee
                }
            }
            
            print(json.dumps(pool_data, indent=2))
            
        except Exception as e:
            print(f"Error fetching pool data: {e}")
    
    except Exception as e:
        print(f"Error during transaction: {e}")
    
except Exception as e:
    print(f"Error: {e}") 