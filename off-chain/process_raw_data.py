import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import logging
import psutil
import os
import json

class BlockchainGANDataset(Dataset):
    """Dataset for blockchain GAN training with conditional inputs"""
    
    def __init__(self, sequences: np.ndarray, conditions: np.ndarray):
        """Initialize dataset with sequences and conditions"""
        # Convert to float32 for PyTorch compatibility
        self.sequences = torch.FloatTensor(sequences.astype(np.float32))
        self.conditions = torch.FloatTensor(conditions.astype(np.float32))
        
        # Log shapes for debugging
        logging.debug(f"Dataset initialized with sequences shape: {self.sequences.shape}")
        logging.debug(f"Dataset initialized with conditions shape: {self.conditions.shape}")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its condition by index"""
        return self.sequences[idx], self.conditions[idx]

class BlockchainDataRepresentation:
    """Handles data representation for GAN training on raw Uniswap V4 blockchain data"""
    
    def __init__(self, sequence_length: int = 50, condition_window: int = 20, batch_size: int = 32):
        """
        Args:
            sequence_length: Number of swap events in each training sequence
            condition_window: Number of recent events to use for market condition
            batch_size: Batch size for training
        """
        self.sequence_length = sequence_length
        self.condition_window = condition_window
        self.batch_size = batch_size
        self.numeric_columns = [
            'amount0', 'amount1', 'sqrt_price_x96', 'liquidity', 'tick',
            'fee', 'timestamp'
        ]
        self.categorical_columns = [
            'pool_id', 'router_address', 'original_sender', 'is_contract_sender'
        ]
        
    def load_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types and validate data"""
        df = df.copy()
        
        # Convert numeric columns except sqrt_price_x96 first
        numeric_cols = [c for c in self.numeric_columns if c != 'sqrt_price_x96']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for NaN values after conversion
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logging.warning(f"Found {nan_count} NaN values in column {col}")
        
        # Special handling for sqrt_price_x96 - preserve the original uint160 format
        if 'sqrt_price_x96' in df.columns:
            # Keep sqrt_price_x96 as string to preserve the exact uint160 precision
            df['sqrt_price_x96'] = df['sqrt_price_x96'].astype(str)
            
            # Log some sample values to verify we're preserving the original data
            logging.info(f"Original sqrt_price_x96 (first 3 values): {df['sqrt_price_x96'].head(3).tolist()}")
        
        # Convert boolean columns
        if 'is_contract_sender' in df.columns:
            df['is_contract_sender'] = df['is_contract_sender'].astype(bool)
        
        # Sort by timestamp to ensure temporal order
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
        
    def calculate_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional market data features based on swap events"""
        df = df.copy()
        
        # Calculate price from tick using Uniswap V3 formula: price = 1.0001^tick
        # For ETH/USDC: token0 (ETH) has 18 decimals, token1 (USDC) has 6 decimals
        TOKEN0_DECIMALS = 18  # ETH
        TOKEN1_DECIMALS = 6   # USDC
        
        # Calculate price from tick using the exact formula
        df['price'] = df['tick'].apply(
            lambda tick: (1.0001 ** tick) * (10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))
        )
        
        # Convert token amounts to native token units
        # ETH amount (amount0) - convert from wei to ETH
        df['amount0_eth'] = df['amount0'] / (10 ** TOKEN0_DECIMALS)
        
        # USDC amount (amount1) - convert from microdollars to dollars
        df['amount1_usdc'] = df['amount1'] / (10 ** TOKEN1_DECIMALS)
        
        # Calculate transaction volume in USD
        df['volume'] = df['amount1_usdc'].abs()
        
        # Calculate price volatility (standard deviation over rolling window)
        price_window = 3  # number of transactions to look at
        if len(df) >= price_window:
            df['volatility'] = df['price'].rolling(window=price_window, min_periods=1).std().fillna(0)
        else:
            df['volatility'] = 0
            
        # Moving average of volume
        volume_window = min(5, len(df))
        df['volume_ma'] = df['volume'].rolling(window=volume_window, min_periods=1).mean().fillna(0)
        
        # Tick change between transactions
        df['tick_change'] = df['tick'].diff().fillna(0)
        
        # Time between transactions
        df['time_diff'] = df['timestamp'].diff().fillna(0)
        
        # Liquidity utilization ratio
        if 'liquidity' in df.columns:
            # Safe handling of large numbers
            df['liquidity_usage'] = df['volume'] / (df['liquidity'].astype(float) + 1)  # add 1 to avoid division by zero
        else:
            df['liquidity_usage'] = 0
        
        return df
        
    def create_condition_vector(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create a condition vector from market data for the GAN"""
        # Extract key market conditions
        # Use only numeric columns that are relevant for market conditions
        
        # If empty dataframe, return zeros
        if len(market_data) == 0:
            return np.zeros(5)
            
        # Calculate average price
        avg_price = market_data['price'].mean()
        
        # Calculate price volatility
        price_volatility = market_data['price'].std() if len(market_data) > 1 else 0
        
        # Calculate average volume
        avg_volume = market_data['volume'].mean()
        
        # Calculate average tick change (market direction)
        avg_tick_change = market_data['tick_change'].mean() if 'tick_change' in market_data.columns else 0
        
        # Calculate time between transactions
        avg_time_diff = market_data['time_diff'].mean() if 'time_diff' in market_data.columns else 0
        
        # Create condition vector
        condition = np.array([
            avg_price,
            price_volatility,
            avg_volume,
            avg_tick_change,
            avg_time_diff
        ], dtype=np.float32)
        
        return condition
        
    def prepare_blockchain_data(self, df: pd.DataFrame) -> Tuple[BlockchainGANDataset, BlockchainGANDataset]:
        """Prepare blockchain data for GAN training by creating sequences and conditions"""
        # Add market conditions
        df_with_conditions = self.calculate_market_conditions(df)
        
        # Select only numeric columns for sequence data
        # Exclude sqrt_price_x96 since it's kept as a string
        numeric_cols = [
            'price', 'amount0_eth', 'amount1_usdc', 'tick', 'volume', 
            'volatility', 'volume_ma', 'tick_change', 'time_diff', 'liquidity_usage'
        ]
        
        # Ensure all selected columns exist
        numeric_cols = [col for col in numeric_cols if col in df_with_conditions.columns]
        
        # Log the columns we're using
        logging.info(f"Using columns for sequence data: {numeric_cols}")
        
        # Calculate normalization parameters and normalize the features
        # We track mean and std for later denormalization
        normalization_params = {}
        df_normalized = df_with_conditions.copy()
        
        for col in numeric_cols:
            # Special handling for volume and amount columns which may have high outliers
            if col in ['volume', 'volume_ma', 'amount0_eth', 'amount1_usdc']:
                # Log transform for heavy-tailed distributions
                # Add a small epsilon to avoid log(0)
                epsilon = 1e-8
                # For columns that can be negative (like amount1_usdc), handle differently
                if df_with_conditions[col].min() < 0:
                    # For columns with negative values, use signed log transform
                    # log(|x| + 1) * sign(x)
                    sign = np.sign(df_with_conditions[col])
                    log_transform = np.log1p(np.abs(df_with_conditions[col]) + epsilon) * sign
                    mean = log_transform.mean()
                    std = log_transform.std()
                    # Store transformation info
                    normalization_params[col] = {
                        'mean': mean, 
                        'std': std, 
                        'transform': 'signed_log'
                    }
                    # Normalize
                    df_normalized[col] = (log_transform - mean) / std
                else:
                    # For positive-only columns like volume
                    log_transform = np.log1p(df_with_conditions[col] + epsilon)
                    mean = log_transform.mean()
                    std = log_transform.std()
                    # Store transformation info
                    normalization_params[col] = {
                        'mean': mean, 
                        'std': std, 
                        'transform': 'log'
                    }
                    # Normalize
                    df_normalized[col] = (log_transform - mean) / std
            else:
                # Standard Z-score normalization for other columns
                mean = df_with_conditions[col].mean()
                std = df_with_conditions[col].std()
                # Avoid division by zero
                if std == 0:
                    std = 1.0
                    
                # Store normalization parameters
                normalization_params[col] = {'mean': mean, 'std': std, 'transform': 'standard'}
                
                # Z-score normalize: (x - mean) / std
                df_normalized[col] = (df_with_conditions[col] - mean) / std
            
            # Log normalization parameters
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            logging.info(f"Normalized {col}: mean=0, std=1, min={min_val:.4f}, max={max_val:.4f}")
        
        # Save normalization parameters to a file for reuse during generation
        norm_params_path = "off-chain/models/normalization_params.json"
        os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)
        with open(norm_params_path, 'w') as f:
            json.dump(normalization_params, f)
        logging.info(f"Saved normalization parameters to {norm_params_path}")
        
        # Create sequences
        sequences = []
        for i in range(len(df_normalized) - self.sequence_length + 1):
            seq = df_normalized[numeric_cols].iloc[i:i+self.sequence_length].values
            sequences.append(seq)
        
        # Create condition vectors
        conditions = []
        for i in range(len(df_with_conditions) - self.sequence_length + 1):
            # Use the last condition_window transactions before each sequence as condition
            start_idx = max(0, i - self.condition_window)
            condition_data = df_with_conditions.iloc[start_idx:i]
            if len(condition_data) > 0:
                condition = self.create_condition_vector(condition_data)
                conditions.append(condition)
            else:
                # If no prior data, use zeros as condition
                condition = np.zeros(5)  # 5 features in condition vector
                conditions.append(condition)
        
        # Convert to numpy arrays
        sequences_np = np.array(sequences)
        conditions_np = np.array(conditions)
        
        # Log shapes
        logging.info(f"Sequences shape: {sequences_np.shape}")
        logging.info(f"Conditions shape: {conditions_np.shape}")
        
        # Split into training and validation sets (80/20 split)
        split_idx = int(0.8 * len(sequences_np))
        
        train_sequences = sequences_np[:split_idx]
        train_conditions = conditions_np[:split_idx]
        
        val_sequences = sequences_np[split_idx:]
        val_conditions = conditions_np[split_idx:]
        
        # Create datasets
        train_dataset = BlockchainGANDataset(
            sequences=train_sequences,
            conditions=train_conditions
        )
        
        val_dataset = BlockchainGANDataset(
            sequences=val_sequences,
            conditions=val_conditions
        )
        
        return train_dataset, val_dataset
        
    def get_data_loaders(self, train_dataset: BlockchainGANDataset, 
                        val_dataset: BlockchainGANDataset) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing overhead
            pin_memory=True  # Faster data transfer to GPU
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to file with market conditions and engineered features.
        
        Args:
            df: Raw DataFrame to process and save
            output_path: Path where to save the processed data
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load and validate data
        validated_df = self.load_and_validate_data(df)
        
        # Calculate market conditions and add engineered features
        processed_df = self.calculate_market_conditions(validated_df)
        
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")
        
        # Log summary of processed data
        logging.info(f"Processed data summary:")
        logging.info(f"Total rows: {len(processed_df)}")
        logging.info(f"Features: {list(processed_df.columns)}")
        
        return processed_df

# Test the implementation with real blockchain data
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure pandas display options to show full transaction hashes
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    
    # Add memory usage monitoring
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Find any CSV file in the off-chain/data/raw directory
    raw_data_dir = "off-chain/data/raw"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        raise FileNotFoundError(f"No data directory found at {raw_data_dir}. Created the directory, please add CSV files there.")
    
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")
    
    # Use the first CSV file found
    data_path = os.path.join(raw_data_dir, csv_files[0])
    logging.info(f"Processing file: {data_path}")
    
    # Load raw blockchain data with high precision
    df = pd.read_csv(data_path, dtype={'sqrt_price_x96': str}, low_memory=False)
    
    after_load_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nMemory usage:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"After loading data: {after_load_memory:.2f} MB")
    print(f"Data loading used: {after_load_memory - initial_memory:.2f} MB")
    
    print(f"\nLoaded {len(df)} raw blockchain events")
    print(f"DataFrame size in memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Initialize data representation
    data_rep = BlockchainDataRepresentation()
    
    # Save processed data
    processed_data_path = "off-chain/data/processed/processed_swaps.csv"
    processed_df = data_rep.save_processed_data(df, processed_data_path)
    
    # Generate and save normalization parameters
    logging.info("Generating normalization parameters...")
    train_dataset, val_dataset = data_rep.prepare_blockchain_data(processed_df)
    logging.info("Normalization parameters have been saved to off-chain/models/normalization_params.json")
    
    after_processing_memory = process.memory_info().rss / 1024 / 1024
    print("\nMemory after processing and saving data:")
    print(f"Total: {after_processing_memory:.2f} MB")
    print(f"Increase: {after_processing_memory - after_load_memory:.2f} MB")
    
    # Print a sample of numeric values with transaction hashes
    print("\nSample of processed data (last 5 events):")
    print(processed_df[['transaction_hash', 'price', 'amount0_eth', 'amount1_usdc', 'tick']].tail())
    
    logging.info(f"Processing complete. Data saved to {processed_data_path}")
    logging.info(f"You can now train the model with: python generate_synthetic_data.py train") 