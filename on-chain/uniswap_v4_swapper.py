"""
Uniswap v4 Swapper
-----------------
A reusable module for executing swaps on Uniswap v4 on BASE
"""

from decimal import Decimal, getcontext
from web3 import Web3
from web3.types import Wei
from eth_abi import encode
from uniswap_universal_router_decoder import RouterCodec
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# ERC20 ABI for token operations
ERC20_ABI = [
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
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
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

class UniswapV4Swapper:
    """
    A class to handle Uniswap v4 swaps on Base
    """
    
    def __init__(self, w3, private_key, config):
        """
        Initialize the swapper
        
        Args:
            w3: Web3 instance
            private_key: Private key for signing transactions
            config: Configuration dictionary with contract addresses
        """
        self.w3 = w3
        self.account = w3.eth.account.from_key(private_key)
        self.address = self.account.address
        
        # Store contract addresses
        self.usdc_address = config['BASE_USDC_ADDRESS']
        self.eth_address = config['ETH_ADDRESS']
        self.weth_address = config.get('WETH_ADDRESS', None)
        self.universal_router = config['UNISWAP_V4_UNIVERSAL_ROUTER']
        self.position_manager = config['UNISWAP_V4_POSITION_MANAGER']
        self.state_view = config['UNISWAP_V4_STATE_VIEW']
        self.pool_manager = config['UNISWAP_V4_POOL_MANAGER']
        
        # Convert addresses to checksum format
        self.usdc_address_cs = Web3.to_checksum_address(self.usdc_address)
        self.eth_address_cs = Web3.to_checksum_address(self.eth_address)
        self.universal_router_cs = Web3.to_checksum_address(self.universal_router)
        self.position_manager_cs = Web3.to_checksum_address(self.position_manager)
        self.state_view_cs = Web3.to_checksum_address(self.state_view)
        self.pool_manager_cs = Web3.to_checksum_address(self.pool_manager)
        
        # Initialize contracts
        self.usdc_contract = w3.eth.contract(address=self.usdc_address_cs, abi=ERC20_ABI)
        
        # Setup Universal Router Codec
        self.codec = RouterCodec(w3)
        
        # Create the v4 pool key for ETH/USDC
        self.eth_usdc_pool_key = self.codec.encode.v4_pool_key(
            self.eth_address_cs,     # Numerically smaller
            self.usdc_address_cs,    # Numerically larger
            500,                     # 0.05% fee tier
            10                       # Tick spacing
        )
        
        # Calculate pool ID
        self.pool_id = self._calculate_pool_id()
        logger.info(f"Initialized UniswapV4Swapper for account: {self.address}")
        logger.info(f"Using pool ID: {self.pool_id}")
    
    def _calculate_pool_id(self):
        """Calculate the Uniswap v4 pool ID for ETH/USDC"""
        # Extract pool key parameters
        token0 = self.eth_usdc_pool_key['currency_0']
        token1 = self.eth_usdc_pool_key['currency_1']
        fee = self.eth_usdc_pool_key['fee']
        tick_spacing = self.eth_usdc_pool_key['tick_spacing']
        hooks = self.eth_usdc_pool_key['hooks']
        
        # Encode the pool parameters
        pool_init_code = encode(
            ['address', 'address', 'uint24', 'int24', 'address'],
            [token0, token1, fee, tick_spacing, hooks]
        )
        
        # Calculate the pool ID (keccak256 hash of the encoded parameters)
        calculated_pool_id = Web3.solidity_keccak(['bytes'], [pool_init_code]).hex()
        return f"0x{calculated_pool_id}"
    
    def get_eth_price(self):
        """Get the current ETH price in USDC from the pool"""
        try:
            # Create state view contract instance
            state_view = self.w3.eth.contract(address=self.state_view_cs, abi=STATE_VIEW_ABI)
            
            # Get slot0 data
            try:
                # Try to call getSlot0 with pool_id directly (newer contract design)
                slot0_data = state_view.functions.getSlot0(self.pool_id).call()
            except Exception as e:
                logger.error(f"Error calling getSlot0 directly: {e}")
                # Try to call with bytes32 pool ID
                pool_id_bytes32 = Web3.to_bytes(hexstr=self.pool_id)
                try:
                    slot0_data = state_view.functions.getSlot0(pool_id_bytes32).call()
                except Exception as e2:
                    logger.error(f"Error calling getSlot0 with bytes32: {e2}")
                    # Fallback to hard-coded price for testing
                    logger.warning(f"Using fallback price for testing")
                    return 1800.0
            
            sqrt_price_x96, tick, protocol_fee, lp_fee = slot0_data
            
            # Calculate price from sqrt_price_x96
            price = calculate_price_from_sqrt_price_x96(sqrt_price_x96)
            
            # If that fails, try calculating from tick
            if not price:
                price = tick_to_price(tick)
            
            return price
        except Exception as e:
            logger.error(f"Error getting ETH price: {e}")
            return None
    
    def get_balances(self):
        """Get ETH and USDC balances for the account"""
        try:
            eth_balance = self.w3.eth.get_balance(self.address)
            usdc_balance = self.usdc_contract.functions.balanceOf(self.address).call()
            
            return {
                "ETH": eth_balance / 10**18,
                "USDC": usdc_balance / 10**6
            }
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return {"ETH": 0, "USDC": 0}
    
    def swap_eth_to_usdc(self, eth_amount, slippage=0.01):
        """
        Swap ETH to USDC
        
        Args:
            eth_amount: Amount of ETH to swap
            slippage: Slippage tolerance (default 1%)
            
        Returns:
            dict: Transaction details
        """
        try:
            # Get balances before swap
            balances_before = self.get_balances()
            
            # Convert ETH amount to Wei
            amount_in = Wei(int(eth_amount * 10**18))
            logger.info(f"Swapping {eth_amount:.8f} ETH to USDC")
            
            # Get current ETH price to estimate output
            eth_price = self.get_eth_price()
            if not eth_price:
                logger.error("Failed to get ETH price, cannot estimate output")
                return {"success": False, "error": "Failed to get ETH price"}
            
            # Estimate USDC output
            estimated_usdc_out = eth_amount * eth_price
            min_usdc_out = estimated_usdc_out * (1 - slippage)
            
            # Build the transaction
            transaction_params = (
                self.codec
                .encode
                .chain()
                .v4_swap()
                .swap_exact_in_single(
                    pool_key=self.eth_usdc_pool_key,
                    zero_for_one=True,  # ETH is token0, USDC is token1
                    amount_in=amount_in,
                    amount_out_min=Wei(int(min_usdc_out * 10**6)),  # USDC has 6 decimals
                )
                .take_all(self.usdc_address_cs, Wei(0))
                .settle_all(self.eth_address_cs, amount_in)
                .build_v4_swap()
                .build_transaction(
                    self.address,
                    amount_in,
                    ur_address=self.universal_router_cs,
                    block_identifier=self.w3.eth.block_number
                )
            )
            
            # Make sure we have all required transaction parameters
            if 'chainId' not in transaction_params:
                transaction_params['chainId'] = self.w3.eth.chain_id
            
            if 'nonce' not in transaction_params:
                transaction_params['nonce'] = self.w3.eth.get_transaction_count(self.address)
            
            if 'gasPrice' not in transaction_params and 'maxFeePerGas' not in transaction_params:
                transaction_params['gasPrice'] = self.w3.eth.gas_price
            
            # Sign and send the transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction_params, self.account.key)
            
            # Get the raw transaction bytes
            if hasattr(signed_txn, 'rawTransaction'):
                raw_tx = signed_txn.rawTransaction
            elif hasattr(signed_txn, 'raw_transaction'):
                raw_tx = signed_txn.raw_transaction
            else:
                raise Exception("Could not find raw transaction data")
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            logger.info(f"Transaction sent: {self.w3.to_hex(tx_hash)}")
            
            # Wait for the transaction to be mined
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if transaction was successful
            if receipt['status'] == 1:
                logger.info("Transaction successful!")
                
                # Get balances after swap
                balances_after = self.get_balances()
                
                # Calculate amounts
                eth_spent = balances_before["ETH"] - balances_after["ETH"]
                usdc_received = balances_after["USDC"] - balances_before["USDC"]
                
                return {
                    "success": True,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "eth_spent": eth_spent,
                    "usdc_received": usdc_received,
                    "effective_price": usdc_received / eth_spent if eth_spent > 0 else 0,
                    "receipt": receipt
                }
            else:
                logger.error(f"Transaction failed: {receipt}")
                return {
                    "success": False,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "error": "Transaction failed",
                    "receipt": receipt
                }
                
        except Exception as e:
            logger.error(f"Error in swap_eth_to_usdc: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def swap_usdc_to_eth(self, usdc_amount, slippage=0.01):
        """
        Swap USDC to ETH
        
        Args:
            usdc_amount: Amount of USDC to swap
            slippage: Slippage tolerance (default 1%)
            
        Returns:
            dict: Transaction details
        """
        try:
            # Get balances before swap
            balances_before = self.get_balances()
            usdc_balance = balances_before['USDC']
            
            # Make sure we don't try to swap more than we have
            if usdc_amount > usdc_balance * 0.99:
                logger.warning(f"Requested USDC amount ({usdc_amount}) higher than 99% of balance ({usdc_balance})")
                usdc_amount = usdc_balance * 0.95  # Use 95% of balance at most
                logger.info(f"Reduced USDC swap amount to {usdc_amount:.6f}")
            
            # Convert USDC amount to smallest unit (6 decimals)
            amount_in = int(usdc_amount * 10**6)
            logger.info(f"Swapping {usdc_amount:.6f} USDC to ETH")
            
            # Ensure minimum USDC amount (at least 0.01 USDC)
            if amount_in < 10000:  # 0.01 USDC in smallest units
                logger.warning(f"USDC amount too small: {usdc_amount:.6f} USDC")
                return {"success": False, "error": "USDC amount too small to swap"}
            
            # Get current ETH price to estimate output
            eth_price = self.get_eth_price()
            if not eth_price:
                logger.error("Failed to get ETH price, cannot estimate output")
                return {"success": False, "error": "Failed to get ETH price"}
            
            # Estimate ETH output
            estimated_eth_out = usdc_amount / eth_price
            min_eth_out = estimated_eth_out * (1 - slippage)
            min_eth_out_wei = Wei(int(min_eth_out * 10**18))
            
            # Check and set allowance for USDC if needed
            try:
                current_allowance = self.usdc_contract.functions.allowance(self.address, self.universal_router_cs).call()
                if current_allowance < amount_in:
                    logger.info(f"Setting USDC allowance for Universal Router. Current: {current_allowance}, Needed: {amount_in}")
                    
                    # First, reset allowance to 0 if it's low but not zero
                    if 0 < current_allowance < amount_in:
                        try:
                            logger.info("Resetting existing allowance to 0 first")
                            reset_approve_txn = self.usdc_contract.functions.approve(
                                self.universal_router_cs,
                                0  # Reset to zero
                            ).build_transaction({
                                'from': self.address,
                                'nonce': self.w3.eth.get_transaction_count(self.address),
                                'gas': 100000,
                                'gasPrice': self.w3.eth.gas_price
                            })
                            
                            # Sign and send reset transaction
                            signed_reset_txn = self.w3.eth.account.sign_transaction(reset_approve_txn, self.account.key)
                            reset_tx_hash = self.w3.eth.send_raw_transaction(signed_reset_txn.rawTransaction)
                            logger.info(f"Reset allowance transaction sent: {self.w3.to_hex(reset_tx_hash)}")
                            reset_receipt = self.w3.eth.wait_for_transaction_receipt(reset_tx_hash)
                            if reset_receipt['status'] != 1:
                                logger.warning("Reset allowance transaction failed, continuing with new approval anyway")
                        except Exception as e:
                            logger.warning(f"Error resetting allowance: {e}, continuing with new approval")
                            
                    # Set allowance - use exact amount instead of unlimited approval
                    approve_txn = self.usdc_contract.functions.approve(
                        self.universal_router_cs,
                        amount_in  # Using exact amount for better security
                    ).build_transaction({
                        'from': self.address,
                        'nonce': self.w3.eth.get_transaction_count(self.address),
                        'gas': 100000,
                        'gasPrice': self.w3.eth.gas_price
                    })
                    
                    # Sign and send approval transaction
                    signed_txn = self.w3.eth.account.sign_transaction(approve_txn, self.account.key)
                    
                    # Get the raw transaction bytes (handle both attribute names)
                    raw_tx = getattr(signed_txn, 'rawTransaction', None) or getattr(signed_txn, 'raw_transaction', None)
                    if not raw_tx:
                        raise Exception("Could not find raw transaction data")
                    
                    # Send transaction
                    tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
                    logger.info(f"Approval transaction sent: {self.w3.to_hex(tx_hash)}")
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    if receipt['status'] != 1:
                        raise Exception("Approval transaction failed!")
                    logger.info("Approval successful!")
                    
                    # Verify new allowance
                    new_allowance = self.usdc_contract.functions.allowance(self.address, self.universal_router_cs).call()
                    logger.info(f"New allowance set: {new_allowance} (needed: {amount_in})")
                    if new_allowance < amount_in:
                        raise Exception(f"Allowance ({new_allowance}) still less than required ({amount_in})")
            except Exception as e:
                logger.error(f"Error setting USDC allowance: {e}")
                return {"success": False, "error": f"Failed to set USDC allowance: {str(e)}"}
            
            # Get current tick from state view to help set bounds
            try:
                state_view = self.w3.eth.contract(address=self.state_view_cs, abi=STATE_VIEW_ABI)
                slot0_data = state_view.functions.getSlot0(self.pool_id).call()
                sqrt_price_x96, current_tick, protocol_fee, lp_fee = slot0_data
                logger.info(f"Current pool tick: {current_tick}")
                
                # Set tick bounds with a buffer to avoid PastOrFutureTickOrder errors
                tick_lower = current_tick - 5000
                tick_upper = current_tick + 5000
                logger.info(f"Using tick range: {tick_lower} to {tick_upper}")
            except Exception as e:
                logger.warning(f"Could not get current tick, using default range: {e}")
                # Default tick range if we can't get current tick
                current_tick = -200000  # Approximate tick for ETH
                tick_lower = current_tick - 5000
                tick_upper = current_tick + 5000
            
            # Build the transaction
            try:
                transaction_params = (
                    self.codec
                    .encode
                    .chain()
                    .v4_swap()
                    .swap_exact_in_single(
                        pool_key=self.eth_usdc_pool_key,
                        zero_for_one=False,  # USDC to ETH (token1 to token0)
                        amount_in=Wei(amount_in),
                        amount_out_min=Wei(0),
                    )
                    .take_all(self.eth_address_cs, Wei(0))
                    .settle_all(self.usdc_address_cs, Wei(amount_in))
                    .build_v4_swap()
                    .build_transaction(
                        self.address,
                        Wei(0),  # No ETH value needed for USDC -> ETH swap
                        ur_address=self.universal_router_cs,
                        block_identifier=self.w3.eth.block_number
                    )
                )
                
                # Make sure we have all required transaction parameters
                if 'chainId' not in transaction_params:
                    transaction_params['chainId'] = self.w3.eth.chain_id
                
                if 'nonce' not in transaction_params:
                    transaction_params['nonce'] = self.w3.eth.get_transaction_count(self.address)
                
                if 'gasPrice' not in transaction_params and 'maxFeePerGas' not in transaction_params:
                    transaction_params['gasPrice'] = self.w3.eth.gas_price
                
                # Set higher gas limit for local Foundry node
                transaction_params['gas'] = 500000
                
                # Sign and send the transaction
                signed_txn = self.w3.eth.account.sign_transaction(transaction_params, self.account.key)
                
                # Get the raw transaction bytes (handle both attribute names)
                raw_tx = getattr(signed_txn, 'rawTransaction', None) or getattr(signed_txn, 'raw_transaction', None)
                if not raw_tx:
                    raise Exception("Could not find raw transaction data")
                
                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
                logger.info(f"Transaction sent: {self.w3.to_hex(tx_hash)}")
                logger.info("Waiting for confirmation...")
                
                # Wait for the transaction to be mined
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                if receipt['status'] == 1:
                    logger.info("Transaction successful!")
                else:
                    logger.error("Transaction failed!")
                    logger.error(receipt)
                    return {
                        "success": False,
                        "tx_hash": self.w3.to_hex(tx_hash),
                        "error": "Transaction failed",
                        "receipt": receipt
                    }
                
                # Check balances after swap
                balances_after = self.get_balances()
                
                # Calculate amounts
                usdc_spent = balances_before["USDC"] - balances_after["USDC"]
                eth_received = balances_after["ETH"] - balances_before["ETH"]
                
                # Calculate gas cost properly
                gas_used = receipt['gasUsed']
                if 'gasPrice' in transaction_params:
                    gas_price = transaction_params['gasPrice']
                elif 'maxFeePerGas' in transaction_params:
                    gas_price = transaction_params['maxFeePerGas']
                else:
                    gas_price = self.w3.eth.gas_price
                
                gas_cost = (gas_used * gas_price) / 10**18
                
                # Calculate the pure swap amount (ETH received before gas costs)
                pure_swap_amount = eth_received + gas_cost
                
                # Calculate pure swap rate
                pure_swap_rate = pure_swap_amount / usdc_spent if usdc_spent > 0 else 0
                
                logger.info(f"USDC spent: {usdc_spent:.6f} USDC")
                logger.info(f"ETH received (after gas): {eth_received:.8f} ETH")
                logger.info(f"ETH received (swap only): {pure_swap_amount:.8f} ETH")
                logger.info(f"Pure swap rate: 1 USDC = {pure_swap_rate:.8f} ETH")
                
                return {
                    "success": True,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "usdc_spent": usdc_spent,
                    "eth_received": eth_received,
                    "pure_eth_received": pure_swap_amount,
                    "gas_cost_eth": gas_cost,
                    "effective_price": usdc_spent / eth_received if eth_received > 0 else 0,
                    "receipt": receipt
                }
                
            except Exception as e:
                logger.error(f"Error executing swap: {e}")
                return {"success": False, "error": str(e)}
                
        except Exception as e:
            logger.error(f"Error in swap_usdc_to_eth: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)} 