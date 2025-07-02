import os
import sys
from datetime import datetime
import time
import pandas as pd
from web3 import Web3
from web3.exceptions import TransactionNotFound, BlockNotFound
from requests.exceptions import ConnectionError

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import config
from concurrent.futures import ThreadPoolExecutor, as_completed

# Maximum retries for web3 operations
MAX_RETRIES = 5
# Base delay between retries (will be multiplied by retry attempt number)
BASE_RETRY_DELAY = 5
# Maximum delay between retries
MAX_RETRY_DELAY = 60

class UniswapV4DataCollector:
    def __init__(self):
        self.setup_web3_connection()
        # Convert address to checksum format
        self.pool_manager_address = Web3.to_checksum_address(config.UNISWAP_V4_POOL_MANAGER)
        self.universal_router_address = Web3.to_checksum_address(config.UNISWAP_V4_UNIVERSAL_ROUTER)
        self.target_pool_id = config.UNISWAP_V4_ETH_USDC_POOL.replace('0x', '')
        
        # Uniswap v4 PoolManager ABI for Swap event
        self.pool_manager_abi = [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "PoolId", "name": "id", "type": "bytes32"},
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
        
        self.setup_contract()

    def setup_web3_connection(self):
        """Initialize Web3 connection with retry mechanism and RPC failover."""
        retry_count = 0
        rpc_index = 0
        
        while True:
            try:
                # Rotate through available RPC endpoints
                rpc_url = config.BASE_RPC_URLS[rpc_index % len(config.BASE_RPC_URLS)]
                self.w3 = Web3(Web3.HTTPProvider(rpc_url))
                
                # Test connection
                self.w3.eth.block_number
                print(f"Successfully connected to Base network using {rpc_url}")
                break
            except Exception as e:
                retry_count += 1
                # Try next RPC endpoint
                rpc_index += 1
                
                if retry_count > MAX_RETRIES * len(config.BASE_RPC_URLS):
                    raise Exception(f"Failed to connect to any Base network RPC after {MAX_RETRIES * len(config.BASE_RPC_URLS)} attempts")
                
                delay = min(BASE_RETRY_DELAY * (retry_count % MAX_RETRIES + 1), MAX_RETRY_DELAY)
                print(f"Connection to {rpc_url} failed: {str(e)}")
                print(f"Trying next endpoint in {delay} seconds... (Attempt {retry_count})")
                time.sleep(delay)

    def setup_contract(self):
        """Initialize contract with retry mechanism."""
        retry_count = 0
        while True:
            try:
                self.contract = self.w3.eth.contract(
                    address=self.pool_manager_address,
                    abi=self.pool_manager_abi
                )
                break
            except Exception as e:
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    raise Exception(f"Failed to setup contract after {MAX_RETRIES} attempts")
                delay = min(BASE_RETRY_DELAY * retry_count, MAX_RETRY_DELAY)
                print(f"Contract setup failed. Retrying in {delay} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                time.sleep(delay)

    def get_last_processed_block(self, output_file: str) -> int:
        """Get the last successfully processed block from the CSV file."""
        try:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                df = pd.read_csv(output_file)
                if not df.empty:
                    return df['block_number'].max()
        except Exception as e:
            print(f"Error reading last processed block: {e}")
        return None

    def collect_swap_events(
        self, 
        start_block: int, 
        end_block: int, 
        batch_size: int = None
    ) -> pd.DataFrame:
        """Collect swap events with retry mechanism."""
        all_events = []
        current_block = start_block
        
        if batch_size is None:
            batch_size = int(config.BATCH_SIZE or 1000)

        event_signature = "0x" + Web3.keccak(
            text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)"
        ).hex()
        pool_id_bytes = bytes.fromhex(self.target_pool_id)
        pool_id_topic = '0x' + pool_id_bytes.hex().zfill(64)

        while current_block < end_block:
            retry_count = 0
            success = False
            
            while not success and retry_count < MAX_RETRIES:
                try:
                    batch_end = min(current_block + batch_size, end_block)
                    
                    event_filter = {
                        'address': self.pool_manager_address,
                        'fromBlock': current_block,
                        'toBlock': batch_end,
                        'topics': [event_signature, pool_id_topic]
                    }
                    
                    events = self.w3.eth.get_logs(event_filter)
                    event_count = 0
                    
                    for event in events:
                        try:
                            processed_event = self.contract.events.Swap().process_log(event)
                            
                            if processed_event.args.id.hex() == self.target_pool_id:
                                tx = self.w3.eth.get_transaction(processed_event.transactionHash.hex())
                                tx_receipt = self.w3.eth.get_transaction_receipt(processed_event.transactionHash.hex())
                                
                                if processed_event.args.sender.lower() != self.universal_router_address.lower():
                                    continue
                                
                                is_contract = len(self.w3.eth.get_code(tx['from'])) > 0
                                
                                event_data = {
                                    'block_number': processed_event.blockNumber,
                                    'transaction_hash': processed_event.transactionHash.hex(),
                                    'pool_id': processed_event.args.id.hex(),
                                    'router_address': processed_event.args.sender,
                                    'original_sender': tx['from'],
                                    'is_contract_sender': is_contract,
                                    'amount0': str(processed_event.args.amount0),
                                    'amount1': str(processed_event.args.amount1),
                                    'sqrt_price_x96': str(processed_event.args.sqrtPriceX96),
                                    'liquidity': str(processed_event.args.liquidity),
                                    'tick': processed_event.args.tick,
                                    'fee': processed_event.args.fee,
                                    'timestamp': self.w3.eth.get_block(processed_event.blockNumber)['timestamp']
                                }
                                
                                all_events.append(event_data)
                                event_count += 1
                        except Exception as e:
                            print(f"Error processing event: {e}")
                            continue
                    
                    print(f"Processed blocks {current_block} to {batch_end}, found {event_count} events")
                    success = True
                    current_block = batch_end + 1
                    
                except (ConnectionError, TransactionNotFound, BlockNotFound) as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        print(f"Failed to process blocks {current_block} to {batch_end} after {MAX_RETRIES} attempts")
                        # Save the current progress
                        if all_events:
                            return pd.DataFrame(all_events)
                        return pd.DataFrame()
                    
                    delay = min(BASE_RETRY_DELAY * retry_count, MAX_RETRY_DELAY)
                    print(f"Error: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    time.sleep(delay)
                    
                    # Reconnect to the network
                    self.setup_web3_connection()
                    self.setup_contract()

        return pd.DataFrame(all_events) if all_events else pd.DataFrame()

    def collect_uniswap_v4_data(
        self,
        start_block: int = None,
    ) -> pd.DataFrame:
        """Main data collection pipeline with resume capability."""
        if start_block is None:
            start_block = int(config.START_BLOCK or 0)
        
        batch_size = int(config.BATCH_SIZE or 1000)

        # Create output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('data/raw', exist_ok=True)
        output_file = f'data/raw/swap_events_{timestamp}.csv'

        # Check for last processed block
        last_block = self.get_last_processed_block(output_file)
        if last_block is not None:
            print(f"Resuming collection from block {last_block + 1}")
            start_block = last_block + 1

        # Configure pandas to not use scientific notation
        pd.set_option('display.float_format', lambda x: f'{x:.0f}' if abs(x) >= 1 else f'{x:g}')
        
        return self._collect_and_save(start_block, batch_size, output_file)

    def _collect_and_save(self, current_block: int, batch_size: int, output_file: str) -> pd.DataFrame:
        """Continuously collect data with error handling and resume capability."""
        blocks_per_chunk = int(600 / config.BASE_BLOCK_TIME)  # 600 seconds = 10 minutes
        max_workers = 8  # Number of parallel threads
        
        try:
            while True:
                try:
                    latest_block = self.w3.eth.block_number
                    
                    if current_block >= latest_block:
                        print("Waiting for new blocks...")
                        time.sleep(30)
                        continue

                    # Calculate how many chunks to process in parallel
                    total_blocks_to_process = min(blocks_per_chunk * max_workers, latest_block - current_block)
                    blocks_per_parallel_chunk = total_blocks_to_process // max_workers
                    
                    if blocks_per_parallel_chunk < batch_size:
                        blocks_per_parallel_chunk = min(batch_size, latest_block - current_block)
                        max_workers = max(1, (latest_block - current_block) // blocks_per_parallel_chunk)
                    
                    print(f"\nProcessing {total_blocks_to_process} blocks using {max_workers} workers")
                    print(f"Each worker will process chunks of {blocks_per_parallel_chunk} blocks in batches of {batch_size}")
                    
                    # Create block ranges for parallel processing
                    block_ranges = []
                    for i in range(max_workers):
                        start = current_block + (i * blocks_per_parallel_chunk)
                        end = min(start + blocks_per_parallel_chunk, latest_block)
                        if start < end:
                            block_ranges.append((start, end))
                    
                    if not block_ranges:
                        print("No valid block ranges to process")
                        time.sleep(30)
                        continue
                    
                    # Process block ranges in parallel
                    all_swap_dfs = []
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_range = {
                            executor.submit(self.collect_swap_events, start, end, batch_size): (start, end) 
                            for start, end in block_ranges
                        }
                        
                        for future in as_completed(future_to_range):
                            block_range = future_to_range[future]
                            try:
                                swap_df = future.result()
                                if len(swap_df) > 0:
                                    all_swap_dfs.append(swap_df)
                                print(f"Completed blocks {block_range[0]} to {block_range[1]}, found {len(swap_df)} events")
                            except Exception as e:
                                print(f"Error processing blocks {block_range[0]} to {block_range[1]}: {e}")
                    
                    # Combine results and save
                    if all_swap_dfs:
                        combined_df = pd.concat(all_swap_dfs, ignore_index=True)
                        file_exists = os.path.isfile(output_file)
                        combined_df.to_csv(output_file, mode='a', header=not file_exists, index=False, float_format='%.0f')
                        
                        print(f"\nChunk Summary:")
                        print(f"Events collected in this round: {len(combined_df)}")
                        
                        if file_exists:
                            total_events = len(pd.read_csv(output_file))
                            print(f"Total events collected so far: {total_events}")
                    
                    # Update current block for next iteration
                    current_block = max(end for _, end in block_ranges) + 1
                    
                except (ConnectionError, TransactionNotFound, BlockNotFound) as e:
                    print(f"Network error: {e}")
                    print("Attempting to reconnect...")
                    time.sleep(BASE_RETRY_DELAY)
                    self.setup_web3_connection()
                    self.setup_contract()
                    continue
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    print("Continuing with next chunk...")
                    # Move forward by one chunk to avoid getting stuck
                    current_block = current_block + blocks_per_chunk
                    continue
                
        except KeyboardInterrupt:
            print("\nData collection stopped by user.")
            if os.path.isfile(output_file):
                final_df = pd.read_csv(output_file)
                print(f"\nFinal Collection Summary:")
                print(f"Total events collected: {len(final_df)}")
                return final_df
            return pd.DataFrame()

if __name__ == '__main__':
    while True:
        try:
            collector = UniswapV4DataCollector()
            collector.collect_uniswap_v4_data(
                start_block=config.START_BLOCK
            )
        except Exception as e:
            print(f"Fatal error: {e}")
            print("Restarting data collection in 60 seconds...")
            time.sleep(60)
            continue 