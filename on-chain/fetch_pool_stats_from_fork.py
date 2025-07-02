from web3 import Web3
import json
from decimal import Decimal, getcontext

# Make sure you have Foundry Anvil running on localhost:8545
LOCAL_RPC_URL = "http://127.0.0.1:8545"
UNISWAP_V4_POOL_MANAGER = "0x498581ff718922c3f8e6a244956af099b2652b2b"
UNISWAP_V4_POSITION_MANAGER = "0x7C5f5A4bBd8fD63184577525326123B519429bDc"
UNISWAP_V4_ETH_USDC_POOL = "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a"
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # 6 decimals
ETH_ADDRESS = "0x0000000000000000000000000000000000000000"  # ETH native token address (18 decimals)

# Set Decimal precision
getcontext().prec = 40

# Known aggregator addresses on BASE
KNOWN_AGGREGATORS = {
    "0x6fF5693b99212Da76ad316178A184AB56D299b43": "Odos Router",
    "0x5C9bdC801a600c006c388FC032dCb27355154cC9": "1inch Router",
}

# Load ABIs
def load_abi(filename):
    with open(filename, 'r') as f:
        return json.load(f)

POOL_MANAGER_ABI = load_abi('pool_manager.abi')
POSITION_MANAGER_ABI = load_abi('position_manager.abi')

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
        # ETH (18 decimals) to USDC (6 decimals) = divide by 10^12
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

def format_amount(amount, decimals):
    """Format token amount with proper decimals"""
    return amount / (10 ** decimals)

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

def main():
    # Connect to local your Foundry Anvil fork instead of BASE network
    w3 = Web3(Web3.HTTPProvider(LOCAL_RPC_URL))
    
    # Verify connection
    if not w3.is_connected():
        raise Exception("Failed to connect to local Anvil fork")
    
    print(f"Connected to local Anvil fork. Current block: {w3.eth.block_number}")
    
    # Create contract instances
    pool_manager = w3.eth.contract(
        address=w3.to_checksum_address(UNISWAP_V4_POOL_MANAGER),
        abi=POOL_MANAGER_ABI
    )
    
    position_manager = w3.eth.contract(
        address=w3.to_checksum_address(UNISWAP_V4_POSITION_MANAGER),
        abi=POSITION_MANAGER_ABI
    )
    
    try:
        # Convert pool ID to bytes32 (the full ID)
        pool_id = UNISWAP_V4_ETH_USDC_POOL
        # Convert to bytes25 for position manager
        pool_id_bytes25 = bytes.fromhex(UNISWAP_V4_ETH_USDC_POOL[2:])[:25]
        
        # Get current block
        current_block = w3.eth.block_number
        
        # Calculate block from 10 minutes ago (assuming 2s block time on Base)
        blocks_ago = 10 * 60 // 2  # 10 minutes in blocks
        start_block = current_block - blocks_ago
        
        print(f"Fetching pool data from blocks {start_block} to {current_block}")
        
        # First get pool information from position manager
        pool_data = position_manager.functions.poolKeys(pool_id_bytes25).call()
        currency0, currency1, fee, tick_spacing, hooks = pool_data
        
        print(f"Successfully retrieved pool keys")
        print(f"Pool Manager address: {UNISWAP_V4_POOL_MANAGER}")
        
        # Get all events from the pool manager
        print("\nFetching all pool manager events...")
        all_logs = w3.eth.get_logs({
            'address': w3.to_checksum_address(UNISWAP_V4_POOL_MANAGER),
            'fromBlock': start_block,
            'toBlock': current_block
        })
        
        print(f"Found {len(all_logs)} total events")
        
        # Parse all events to find swaps
        swaps = []
        aggregator_volumes = {}
        
        for log in all_logs:
            try:
                # Try to parse as Swap event
                parsed_log = pool_manager.events.Swap().process_log(log)
                if parsed_log['args']['id'].hex() == UNISWAP_V4_ETH_USDC_POOL[2:]:
                    swaps.append(parsed_log)
                    
                    # Track aggregator volumes
                    sender = parsed_log['args']['sender']
                    if sender in KNOWN_AGGREGATORS:
                        if sender not in aggregator_volumes:
                            aggregator_volumes[sender] = {'eth': 0, 'usdc': 0}
                        amount0 = abs(int(parsed_log['args']['amount0'])) / 1e18
                        amount1 = abs(int(parsed_log['args']['amount1'])) / 1e6
                        aggregator_volumes[sender]['eth'] += amount0
                        aggregator_volumes[sender]['usdc'] += amount1
                    
                    # Print details of first few swaps for debugging
                    if len(swaps) <= 3:
                        print(f"\nSwap {len(swaps)} details:")
                        print(f"Sender: {parsed_log['args']['sender']} {KNOWN_AGGREGATORS.get(parsed_log['args']['sender'], '')}")
                        print(f"Amount0: {parsed_log['args']['amount0'] / 1e18:.6f} ETH")
                        print(f"Amount1: {parsed_log['args']['amount1'] / 1e6:.2f} USDC")
                        print(f"SqrtPrice: {parsed_log['args']['sqrtPriceX96']}")
                        print(f"Liquidity: {parsed_log['args']['liquidity']}")
                        print(f"Tick: {parsed_log['args']['tick']}")
            except Exception as e:
                # Not a swap event or failed to parse
                continue
        
        print(f"\nFound {len(swaps)} swaps for our pool")
        
        # Calculate volume and fees
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
                
                # Calculate fees (fee is in hundredths of a bip, i.e. 0.0001%)
                fee_rate = fee / 1_000_000  # Use the pool's fee rate
                fees_eth += amount0 * fee_rate
                fees_usdc += amount1 * fee_rate
            except Exception as e:
                print(f"Error processing swap: {e}")
                continue
        
        # Calculate TVL and current price from the last swap
        if swaps:
            last_swap = swaps[-1]
            liquidity = int(last_swap['args']['liquidity'])
            sqrt_price_x96 = int(last_swap['args']['sqrtPriceX96'])
            tick = int(last_swap['args']['tick'])
            
            # Calculate price from sqrtPriceX96
            eth_price = calculate_price_from_sqrt_price_x96(sqrt_price_x96)
            if not eth_price:
                # Fallback to tick-based price if sqrtPrice calculation fails
                eth_price = tick_to_price(tick)
                print(f"Using tick-based price calculation: ${eth_price:,.2f}")
            
            # Calculate amounts from liquidity using improved formulas
            eth_amount, usdc_amount = calculate_amounts_from_liquidity(liquidity, sqrt_price_x96, tick)
            
            # Calculate TVL using current ETH price
            tvl = (eth_amount * eth_price) + usdc_amount
            
            print(f"\nLast swap details:")
            print(f"Tick: {tick}")
            print(f"SqrtPrice: {sqrt_price_x96}")
            print(f"Liquidity: {liquidity}")
            print(f"Calculated ETH price: ${eth_price:,.2f}")
            
            # Verify price calculation by comparing amount ratios
            if volume_eth > 0 and volume_usdc > 0:
                avg_price = volume_usdc / volume_eth
                print(f"Average price from recent swaps: ${avg_price:,.2f}")
        else:
            eth_price = 0
            eth_amount = 0
            usdc_amount = 0
            tvl = 0
        
        # Print results
        print("\nUniswap V4 ETH/USDC Pool Stats (Last 10 minutes)")
        print("-" * 50)
        print(f"Pool ID: {UNISWAP_V4_ETH_USDC_POOL}")
        print(f"Currency0: {currency0} (ETH)")
        print(f"Currency1: {currency1} (USDC)")
        print(f"Pool Fee: {fee/10000:.4f}%")  # Fee is in hundredths of a bip
        
        print(f"\nPool Balances:")
        print(f"ETH: {eth_amount:.4f}")
        print(f"USDC: {usdc_amount:.2f}")
        print(f"TVL: ${tvl:,.2f}")
        
        print(f"\n10min Volume:")
        print(f"ETH: {volume_eth:.4f}")
        print(f"USDC: ${volume_usdc:,.2f}")
        
        if aggregator_volumes:
            print(f"\nVolume by Aggregator:")
            for addr, vol in aggregator_volumes.items():
                name = KNOWN_AGGREGATORS[addr]
                print(f"{name}:")
                print(f"  ETH: {vol['eth']:.4f}")
                print(f"  USDC: ${vol['usdc']:,.2f}")
        
        print(f"\nFees Collected (10min):")
        print(f"ETH: {fees_eth:.6f}")
        print(f"USDC: ${fees_usdc:.2f}")
        
        print(f"\nETH/USDC Price: ${eth_price:,.2f}")
        print(f"Number of swaps in last 10min: {len(swaps)}")
        
    except Exception as e:
        print(f"Error fetching pool data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 