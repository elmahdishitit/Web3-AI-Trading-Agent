import os
import sys
from decimal import Decimal, getcontext
from web3 import Web3
from web3.types import Wei
from eth_abi import encode
from uniswap_universal_router_decoder import RouterCodec
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOCAL_RPC_URL, 
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

# Permit2 constants
PERMIT2_ADDRESS = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
MAX_UINT160 = 2**160 - 1

# ABI for USDC contract (ERC20)
USDC_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]

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

# Permit2 ABI
PERMIT2_ABI = [
    {
        "inputs": [
            {"name": "token", "type": "address"},
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"}
        ],
        "name": "approve",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "", "type": "address"},
            {"name": "", "type": "address"},
            {"name": "", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"},
            {"name": "nonce", "type": "uint48"}
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
        logger.error(f"Error calculating price: {e}")
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
        logger.error(f"Error calculating price from tick: {e}")
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
        logger.error(f"Error calculating amounts: {e}")
        return 0, 0

def setup_permit2_allowance(w3, account, token_address, spender_address, amount):
    """
    Set up Permit2 allowance for a token
    """
    permit2_contract = w3.eth.contract(address=PERMIT2_ADDRESS, abi=PERMIT2_ABI)
    
    # Check current allowance
    try:
        current_allowance = permit2_contract.functions.allowance(
            account.address, token_address, spender_address
        ).call()
        logger.info(f"Current Permit2 allowance: {current_allowance}")
        
        # If allowance is sufficient and not expired, return
        if current_allowance[0] >= amount and current_allowance[1] > int(time.time()):
            logger.info("Sufficient Permit2 allowance already exists")
            return True
            
    except Exception as e:
        logger.warning(f"Error checking allowance: {e}")
    
    # Set new allowance
    expiration = int(time.time()) + 3600  # 1 hour from now
    
    try:
        logger.info(f"Setting Permit2 allowance...")
        
        # Build transaction
        tx_data = permit2_contract.functions.approve(
            token_address,
            spender_address,
            MAX_UINT160,
            expiration
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'chainId': w3.eth.chain_id
        })
        
        # Sign and send
        signed_tx = account.sign_transaction(tx_data)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            logger.info(f"Permit2 allowance set successfully: {tx_hash.hex()}")
            return True
        else:
            logger.error(f"Permit2 allowance transaction failed")
            return False
            
    except Exception as e:
        logger.error(f"Error setting Permit2 allowance: {e}")
        return False

# Connect to local Foundry node instead of BASE
w3 = Web3(Web3.HTTPProvider(LOCAL_RPC_URL))
if not w3.is_connected():
    raise Exception(f"Failed to connect to local Foundry node at {LOCAL_RPC_URL}")

# Log connection details
logger.info(f"Connected to local Foundry node: {LOCAL_RPC_URL}")
logger.info(f"Current block number: {w3.eth.block_number}")

# Setup account
account = w3.eth.account.from_key(PRIVATE_KEY)
logger.info(f"Account: {account.address}")

# Convert addresses to checksum format
USDC_ADDRESS_CS = Web3.to_checksum_address(BASE_USDC_ADDRESS)
ETH_ADDRESS_CS = Web3.to_checksum_address(ETH_ADDRESS)
UNIVERSAL_ROUTER_CS = Web3.to_checksum_address(UNISWAP_V4_UNIVERSAL_ROUTER)
POSITION_MANAGER_CS = Web3.to_checksum_address(UNISWAP_V4_POSITION_MANAGER)
STATE_VIEW_CS = Web3.to_checksum_address(UNISWAP_V4_STATE_VIEW)
POOL_MANAGER_CS = Web3.to_checksum_address(UNISWAP_V4_POOL_MANAGER)

# Check balances before swap
eth_balance = w3.eth.get_balance(account.address)
logger.info(f"ETH balance: {eth_balance / 10**18:.8f} ETH")

# Setup USDC contract
usdc_contract = w3.eth.contract(address=USDC_ADDRESS_CS, abi=USDC_ABI)
usdc_balance = usdc_contract.functions.balanceOf(account.address).call()
logger.info(f"USDC balance: {usdc_balance / 10**6:.6f} USDC")

logger.info("-" * 50)

# Setup Universal Router Codec
codec = RouterCodec(w3)

# Amount to swap (0.01 USDC)
amount_in = Wei(int(0.01 * 10**6))  # USDC has 6 decimals
logger.info(f"Swapping {amount_in / 10**6:.6f} USDC to ETH via Universal Router")

# Step 1: Approve Permit2 to spend USDC
logger.info("Step 1: Setting USDC allowance for Permit2...")

# Check current ERC20 allowance for Permit2
current_allowance = usdc_contract.functions.allowance(account.address, PERMIT2_ADDRESS).call()
logger.info(f"Current ERC20 allowance for Permit2: {current_allowance / 10**6:.6f} USDC")

if current_allowance < 10**6 * 1000:  # 1000 USDC worth of allowance
    logger.info("Setting ERC20 allowance for Permit2...")
    
    # Build approval transaction
    approve_tx = usdc_contract.functions.approve(
        PERMIT2_ADDRESS,
        2**256 - 1  # Max approval
    ).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 100000,
        'gasPrice': w3.eth.gas_price,
        'chainId': w3.eth.chain_id
    })
    
    # Sign and send
    signed_tx = account.sign_transaction(approve_tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    if receipt.status == 1:
        logger.info(f"ERC20 approval successful: {tx_hash.hex()}")
    else:
        logger.error("ERC20 approval failed")
        raise Exception("ERC20 approval failed")
else:
    logger.info("Sufficient ERC20 allowance already exists for Permit2")

# Step 2: Set Permit2 allowance for Universal Router
logger.info("Step 2: Setting Permit2 allowance for Universal Router...")

if not setup_permit2_allowance(w3, account, Web3.to_checksum_address(BASE_USDC_ADDRESS), Web3.to_checksum_address(UNISWAP_V4_UNIVERSAL_ROUTER), amount_in):
    logger.error("Failed to set Permit2 allowance")
    raise Exception("Failed to set Permit2 allowance")

# Create the correct v4 pool key for ETH/USDC
eth_usdc_pool_key = codec.encode.v4_pool_key(
    ETH_ADDRESS_CS,          # Numerically smaller
    USDC_ADDRESS_CS,         # Numerically larger
    500,                     # 0.05% fee tier
    10                       # Tick spacing
)

# Print the pool key parameters for reference
logger.info(f"Pool key parameters: {eth_usdc_pool_key}")

# Extract pool key parameters
token0 = eth_usdc_pool_key['currency_0']
token1 = eth_usdc_pool_key['currency_1']
fee = eth_usdc_pool_key['fee']
tick_spacing = eth_usdc_pool_key['tick_spacing']
hooks = eth_usdc_pool_key['hooks']

# Correct UniswapV4 Pool ID calculation based on PoolId.sol
# Encode the pool parameters
pool_init_code = encode(
    ['address', 'address', 'uint24', 'int24', 'address'],
    [token0, token1, fee, tick_spacing, hooks]
)

# Calculate the pool ID (keccak256 hash of the encoded parameters)
calculated_pool_id = Web3.solidity_keccak(['bytes'], [pool_init_code]).hex()

# Print the calculated pool ID
logger.info(f"Calculated Pool ID: 0x{calculated_pool_id}")

# Query the position manager to get the actual pool parameters
try:
    # Convert pool ID to bytes25 by taking first 25 bytes
    pool_id_bytes25 = bytes.fromhex(calculated_pool_id)[:25]
    position_manager = w3.eth.contract(address=POSITION_MANAGER_CS, abi=POSITION_MANAGER_ABI)
    
    # Get actual pool parameters from the chain
    pool_data = position_manager.functions.poolKeys(pool_id_bytes25).call()
    
    logger.info("\nPool data retrieved from the chain:")
    logger.info(f"  Currency0: {pool_data[0]}")
    logger.info(f"  Currency1: {pool_data[1]}")
    logger.info(f"  Fee: {pool_data[2] / 10000}%")
    logger.info(f"  Tick Spacing: {pool_data[3]}")
    logger.info(f"  Hooks: {pool_data[4]}")
    
except Exception as e:
    logger.error(f"Error retrieving pool data: {e}")

# Use the calculated pool ID for the transaction
pool_id = f"0x{calculated_pool_id}"
logger.info(f"\nUsing pool ID for transaction: {pool_id}")
logger.info(f"Uniswap info: https://info.uniswap.org/#/base/pools/{pool_id}")

# Get current tick from state view to help set bounds
try:
    state_view = w3.eth.contract(address=STATE_VIEW_CS, abi=STATE_VIEW_ABI)
    slot0_data = state_view.functions.getSlot0(pool_id).call()
    sqrt_price_x96, current_tick, protocol_fee, lp_fee = slot0_data
    logger.info(f"Current pool tick: {current_tick}")
    
    # Set tick bounds with a buffer to avoid PastOrFutureTickOrder errors
    tick_lower = current_tick - 5000
    tick_upper = current_tick + 5000
    logger.info(f"Using tick range: {tick_lower} to {tick_upper}")
except Exception as e:
    logger.warning(f"Could not get current tick, using default range: {e}")
    tick_lower = -887272
    tick_upper = 887272

# Build the transaction
try:
    # Convert amount_in to proper Wei if it isn't already
    if not isinstance(amount_in, int):
        amount_in = int(amount_in)
    amount_in = Wei(amount_in)
    
    logger.info(f"Building transaction with amount_in: {amount_in} wei ({amount_in / 10**6:.6f} USDC)")
    
    # We'll use swap_exact_in_single for a direct USDC to ETH swap
    transaction_params = (
        codec
        .encode
        .chain()
        .v4_swap()
        .swap_exact_in_single(
            pool_key=eth_usdc_pool_key,
            zero_for_one=False,  # USDC to ETH (token1 to token0)
            amount_in=amount_in,
            amount_out_min=Wei(0),  # Setting min amount to 0 - in production set a reasonable slippage
        )
        .take_all(ETH_ADDRESS_CS, Wei(0))
        .settle_all(USDC_ADDRESS_CS, amount_in)
        .build_v4_swap()
        .build_transaction(
            account.address,
            Wei(0),  # No ETH value needed for USDC -> ETH swap
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
    
    # Set higher gas limit for local Foundry node
    transaction_params['gas'] = 500000
    
    logger.info(f"Transaction parameters prepared. Gas limit: {transaction_params['gas']}")
    
    # Sign and send the transaction
    signed_txn = w3.eth.account.sign_transaction(transaction_params, account.key)
    
    # Get the raw transaction bytes (handle both attribute names)
    raw_tx = getattr(signed_txn, 'rawTransaction', None) or getattr(signed_txn, 'raw_transaction', None)
    if not raw_tx:
        raise Exception("Could not find raw transaction data")
    
    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    logger.info(f"Transaction sent: {w3.to_hex(tx_hash)}")
    logger.info("Waiting for confirmation...")
    
    # Wait for the transaction to be mined
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    if receipt['status'] == 1:
        logger.info("Transaction successful!")
    else:
        logger.error("Transaction failed!")
        logger.error(receipt)
    
    logger.info("-" * 50)
    
    # Check balances after swap
    eth_balance_after = w3.eth.get_balance(account.address)
    usdc_balance_after = usdc_contract.functions.balanceOf(account.address).call()
    
    eth_balance_diff = (eth_balance_after - eth_balance) / 10**18
    usdc_spent = (usdc_balance - usdc_balance_after) / 10**6
    
    logger.info(f"ETH balance: {eth_balance_after / 10**18:.8f} ETH")
    logger.info(f"USDC balance: {usdc_balance_after / 10**6:.6f} USDC")
    logger.info("-" * 50)
    
    # Get transaction summary
    logger.info(f"Summary:")
    logger.info(f"USDC spent: {usdc_spent:.6f} USDC")
    
    # Calculate gas cost properly (handle both legacy and EIP-1559 transactions)
    gas_used = receipt['gasUsed']
    if 'gasPrice' in transaction_params:
        gas_price = transaction_params['gasPrice']
    elif 'maxFeePerGas' in transaction_params:
        gas_price = transaction_params['maxFeePerGas']
    else:
        gas_price = w3.eth.gas_price
    
    gas_cost = (gas_used * gas_price) / 10**18
    
    # For USDC -> ETH swap, eth_balance_diff should be positive (we received ETH)
    # If it's negative, it means gas costs exceeded the swap amount, which is normal for small swaps
    if eth_balance_diff > 0:
        pure_swap_amount = eth_balance_diff
        logger.info(f"ETH received (net): {pure_swap_amount:.8f} ETH")
    else:
        # Calculate what would have been received before gas costs
        pure_swap_amount = abs(eth_balance_diff) + gas_cost
        logger.info(f"ETH received (gross, before gas): {pure_swap_amount:.8f} ETH")
        logger.info(f"Gas cost: {gas_cost:.8f} ETH")
        logger.info(f"Net ETH change: {eth_balance_diff:.8f} ETH")
    
    # Calculate pure swap rate
    pure_swap_rate = pure_swap_amount / usdc_spent if usdc_spent > 0 else 0
    
    logger.info(f"Pure swap rate: 1 USDC = {pure_swap_rate:.8f} ETH")
    
    # Fetch pool data using the state view contract
    logger.info("\nFetching pool data...")
    try:
        # Create state view contract instance
        state_view = w3.eth.contract(address=STATE_VIEW_CS, abi=STATE_VIEW_ABI)
        
        # Get slot0 data
        slot0_data = state_view.functions.getSlot0(pool_id).call()
        sqrt_price_x96, tick, protocol_fee, lp_fee = slot0_data
        
        # Calculate prices
        # For ETH/USDC: token0 (ETH) has 18 decimals, token1 (USDC) has 6 decimals
        price_from_sqrt = (sqrt_price_x96 / (2**96)) ** 2 * (10 ** (ETH_DECIMALS - USDC_DECIMALS))
        price_from_tick = 1.0001 ** tick * (10 ** (ETH_DECIMALS - USDC_DECIMALS))
        
        # Format output similar to eth_to_usdc_swap.py
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
        logger.error(f"Error fetching pool data: {e}")
    
except Exception as e:
    logger.error(f"Error: {e}")

# If run as a script
if __name__ == "__main__":
    pass  # All code is at module level 