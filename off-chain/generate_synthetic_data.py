import argparse
import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_raw_data import BlockchainDataRepresentation
from gan.models import Generator, Discriminator
from gan.training import WGANGPTrainer
from gan.generation import SyntheticDataGenerator
from gan.visualization import visualize_financial_sequences, visualize_comparison, generate_sample_visualization
from config import GAN_CONFIG, QUICK_TEST_GAN_CONFIG, QUICK_TEST_MODE

def setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gan_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")


def load_data(data_path: str, quick_test_mode: bool = False) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        data_path: Path to data file
        quick_test_mode: If True, only load a subset of data for quick testing
        
    Returns:
        Loaded data as DataFrame
    """
    # Check file extension
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # If in quick test mode, use only a small subset of data
    if quick_test_mode:
        # Use only 1000 samples for quick testing
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)
            logging.info(f"Quick test mode enabled: Using {len(df)} samples")
    
    logging.info(f"Loaded data from {data_path} with shape {df.shape}")
    
    return df


def train_gan(
    quick_test_mode: bool = False,
    data_path: str = None,
    output_dir: str = None,
    checkpoint_dir: str = None,
    sequence_length: int = None,
    condition_window: int = None,
    batch_size: int = None,
    latent_dim: int = None,
    hidden_dim: int = None,
    epochs: int = None,
    lr: float = None,
    n_critic: int = None,
    lambda_gp: float = None,
    early_stopping_patience: int = None,
    device: str = None,
    seed: int = None,
    # New parameters for enhanced training
    noise_std: float = 0.2,
    diversity_lambda: float = 0.1,
    feature_matching_lambda: float = 0.5,
    dropout: float = 0.2
) -> dict:
    """
    Train GAN model on blockchain data.
    
    Args:
        quick_test_mode: If True, use smaller model and fewer epochs for quick testing
        data_path: Path to data file
        output_dir: Directory to save model
        checkpoint_dir: Directory to save checkpoints
        sequence_length: Length of sequences
        condition_window: Window size for market conditions
        batch_size: Batch size for training
        latent_dim: Dimension of latent space
        hidden_dim: Hidden dimension for models
        epochs: Number of epochs to train
        lr: Learning rate
        n_critic: Number of discriminator updates per generator update
        lambda_gp: Weight for gradient penalty
        early_stopping_patience: Patience for early stopping
        device: Device to run training on (CPU or GPU)
        seed: Random seed for reproducibility
        noise_std: Standard deviation for instance noise
        diversity_lambda: Weight for diversity loss
        feature_matching_lambda: Weight for feature matching loss
        dropout: Dropout probability for models
        
    Returns:
        Dictionary with training results
    """
    # Select the appropriate config based on mode
    config = QUICK_TEST_GAN_CONFIG if quick_test_mode else GAN_CONFIG
    
    # Use provided parameters if given, otherwise use from config
    data_path = data_path or config['processed_data_path']
    output_dir = output_dir or config['output_dir']
    checkpoint_dir = checkpoint_dir or config['checkpoint_dir']
    sequence_length = sequence_length or config['sequence_length']
    condition_window = condition_window or config['condition_window']
    batch_size = batch_size or config['batch_size']
    latent_dim = latent_dim or config['latent_dim']
    hidden_dim = hidden_dim or config['hidden_dim']
    epochs = epochs or config['epochs']
    lr = lr or config['learning_rate']
    n_critic = n_critic or config['n_critic']
    lambda_gp = lambda_gp or config['lambda_gp']
    early_stopping_patience = early_stopping_patience or config['early_stopping_patience']
    seed = seed or config['seed']
    
    if quick_test_mode:
        logging.info("QUICK TEST MODE ENABLED: Using smaller model and fewer epochs")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Training configuration: batch_size={batch_size}, latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    logging.info(f"Training configuration: epochs={epochs}, lr={lr}, n_critic={n_critic}")
    logging.info(f"Enhanced training parameters: noise_std={noise_std}, diversity_lambda={diversity_lambda}, feature_matching_lambda={feature_matching_lambda}")
    
    # Load and preprocess data
    df = load_data(data_path, quick_test_mode)
    
    # Log feature statistics before normalization
    numeric_columns = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    logging.info("Feature statistics before normalization:")
    for col in numeric_columns:
        logging.info(f"  {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, min={df[col].min():.6f}, max={df[col].max():.6f}")
    
    # Load normalization parameters if exists, otherwise create them
    norm_params_path = os.path.join(output_dir, 'normalization_params.json')
    
    # Initialize blockchain data representation
    blockchain_data = BlockchainDataRepresentation(
        sequence_length=sequence_length,
        condition_window=condition_window,
        batch_size=batch_size
    )
    
    # Get sequence data columns (all numeric columns except those used for conditions)
    feature_mapping = setup_feature_mapping()
    sequence_columns = feature_mapping
    
    logging.info(f"Using columns for sequence data: {sequence_columns}")
    
    # Prepare data for GAN training
    train_dataset, val_dataset = blockchain_data.prepare_blockchain_data(df)
    
    # Get data loaders
    train_dataloader, val_dataloader = blockchain_data.get_data_loaders(train_dataset, val_dataset)
    
    # Extract dimensions from the dataset
    feature_dim = train_dataset.sequences.shape[2]
    condition_dim = train_dataset.conditions.shape[1]
    
    logging.info(f"Feature dimension: {feature_dim}")
    logging.info(f"Condition dimension: {condition_dim}")
    
    # Create generator and discriminator
    generator = Generator(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1)
    )
    
    discriminator = Discriminator(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        dropout=dropout  # Use enhanced dropout
    )
    
    # Create training configuration
    from gan.training import TrainingConfig
    training_config = TrainingConfig(
        lambda_gp=lambda_gp,
        n_critic=n_critic,
        lr=lr,
        noise_std=noise_std,
        diversity_lambda=diversity_lambda,
        feature_matching_lambda=feature_matching_lambda,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        model_subdir=os.path.basename(checkpoint_dir)
    )
    
    # Create trainer
    trainer = WGANGPTrainer(
        generator=generator,
        discriminator=discriminator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        config=training_config
    )
    
    # Train model
    history = trainer.train(epochs)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'gan_model.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'condition_dim': condition_dim,
        'sequence_length': sequence_length,
        'feature_dim': feature_dim,
        'hidden_dim': hidden_dim,
        'num_heads': config.get('num_heads', 8),
        'num_layers': config.get('num_layers', 4),
        'config': config
    }, model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert training history values to plain lists for JSON serialization
        serializable_history = {}
        for k, v in history.items():
            serializable_history[k] = [float(val) for val in v]
        json.dump(serializable_history, f)
    
    logging.info("Training completed successfully")
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Training history saved to {history_path}")
    
    return {
        'model_path': model_path,
        'history_path': history_path,
        'metrics': {
            'final_g_loss': float(history['g_losses'][-1]) if history['g_losses'] else None,
            'final_d_loss': float(history['d_losses'][-1]) if history['d_losses'] else None,
            'final_w_distance': float(history['w_distances'][-1]) if history['w_distances'] else None,
            'sequence_shape': train_dataset.sequences.shape,
            'condition_shape': train_dataset.conditions.shape
        }
    }


def setup_feature_mapping():
    """Define the feature names for the synthetic data"""
    return [
        'price',
        'amount0_eth',
        'amount1_usdc',
        'tick',
        'volume',
        'volatility',
        'volume_ma',
        'tick_change',
        'time_diff',
        'liquidity_usage'
    ]

def clamp_denormalized_data(denormalized_sequences, real_data_stats):
    """
    Clamp extreme values in denormalized sequences and ensure feature consistency.
    Enforces realistic financial relationships between price, volume, and volatility.
    """
    clamped_sequences = denormalized_sequences.copy()
    feature_mapping = setup_feature_mapping()
    
    # Get feature indices
    price_idx = feature_mapping.index('price')
    volume_idx = feature_mapping.index('volume')
    volatility_idx = feature_mapping.index('volatility')
    
    # First pass: Basic clamping and standardization
    for feat_idx, feature in enumerate(feature_mapping):
        if feature in real_data_stats:
            stats = real_data_stats[feature]
            mean_val = stats['mean']
            std_val = stats['std']
            min_val = stats['min']
            max_val = stats['max']
            
            # Set more realistic bounds
            if feature == 'price':
                price_range = max_val - min_val
                lower_bound = max(min_val, mean_val - 0.3 * price_range)
                upper_bound = min(max_val, mean_val + 0.3 * price_range)
            elif feature == 'volume' or feature == 'volume_ma':
                # Ensure non-zero minimum volume to avoid flat lines
                lower_bound = max(0.05 * std_val, 1.0)  # Prevent flat/zero volume
                upper_bound = mean_val + 2 * std_val    # Less extreme maximum
            elif feature == 'volatility':
                # Reasonable volatility bounds
                lower_bound = 0.005 * mean_val  # Small but non-zero minimum
                upper_bound = min(max_val, 2 * mean_val)  # Less extreme maximum
            else:
                lower_bound = min_val - std_val
                upper_bound = max_val + std_val
            
            # Clamp values
            clamped_sequences[:, :, feat_idx] = np.clip(
                clamped_sequences[:, :, feat_idx], lower_bound, upper_bound
            )
    
    # Second pass: Ensure financial relationships and consistency
    for seq_idx in range(clamped_sequences.shape[0]):
        # Ensure price changes are smooth within each sequence
        prices = clamped_sequences[seq_idx, :, price_idx]
        volumes = clamped_sequences[seq_idx, :, volume_idx]
        volatilities = clamped_sequences[seq_idx, :, volatility_idx]
        
        # Smooth extreme price jumps
        for t in range(1, len(prices)):
            # Limit single-step price changes to a percentage of current price
            max_step = 0.01 * prices[t-1]  # 1% max change per step
            if abs(prices[t] - prices[t-1]) > max_step:
                # Adjust price to be closer to previous price
                direction = np.sign(prices[t] - prices[t-1])
                prices[t] = prices[t-1] + direction * max_step
        
        # Update volumes based on price changes (more change = more volume)
        for t in range(1, len(prices)):
            price_change_pct = abs(prices[t] - prices[t-1]) / prices[t-1]
            
            # Calculate baseline volume
            baseline_vol = np.mean(volumes) * 0.8
            
            # Price changes should affect volume
            if price_change_pct > 0.005:  # If price change is significant
                # Increase volume proportional to price change
                volume_factor = 1.0 + price_change_pct * 50  # Scale factor
                volumes[t] = max(baseline_vol * volume_factor, volumes[t])
            else:
                # Ensure volume doesn't stay completely flat (add small noise)
                if volumes[t] < 0.01 or abs(volumes[t] - volumes[t-1]) < 0.001:
                    volumes[t] = max(baseline_vol * (0.9 + 0.2 * np.random.random()), 1.0)
        
        # Update volatility based on price changes and volume
        for t in range(1, len(prices)):
            # Calculate recent price volatility
            if t >= 3:
                recent_prices = prices[max(0, t-3):t+1]
                price_std = np.std(recent_prices)
                price_mean = np.mean(recent_prices)
                price_volatility = price_std / price_mean if price_mean > 0 else 0
                
                # Higher price volatility and volume should lead to higher volatility
                base_volatility = 0.01 * mean_val
                vol_factor = max(1.0, volumes[t] / (np.mean(volumes) + 1e-6))
                
                # Set volatility based on price changes and volume
                volatilities[t] = max(
                    base_volatility,
                    price_volatility * 100 * vol_factor  # Scale appropriately
                )
        
        # Ensure volatility doesn't exceed reasonable limits
        volatilities = np.clip(volatilities, 0.001, upper_bound)
        
        # Update the clamped sequences with our adjusted values
        clamped_sequences[seq_idx, :, price_idx] = prices
        clamped_sequences[seq_idx, :, volume_idx] = volumes
        clamped_sequences[seq_idx, :, volatility_idx] = volatilities
    
    # Third pass: Ensure sequence starts at reasonable values and has overall trends
    for seq_idx in range(clamped_sequences.shape[0]):
        prices = clamped_sequences[seq_idx, :, price_idx]
        volumes = clamped_sequences[seq_idx, :, volume_idx]
        volatilities = clamped_sequences[seq_idx, :, volatility_idx]
        
        # Give each sequence a small but noticeable trend (randomly up or down)
        seq_length = len(prices)
        if np.random.random() > 0.5:  # 50% chance of uptrend
            trend_factor = 1.0 + 0.0005 * np.arange(seq_length)  # +0.05% per step
        else:  # 50% chance of downtrend
            trend_factor = 1.0 - 0.0003 * np.arange(seq_length)  # -0.03% per step
        
        # Apply trend
        baseline_price = prices[0]
        prices = baseline_price * trend_factor
        
        # Make sure volume doesn't remain flat (add some randomness if needed)
        if np.std(volumes) / np.mean(volumes) < 0.05:  # Very flat volume
            base_vol = np.mean(volumes)
            # Create a more variable volume pattern
            volumes = base_vol * (0.7 + 0.6 * np.random.random(seq_length))
        
        # Update the sequences
        clamped_sequences[seq_idx, :, price_idx] = prices
        clamped_sequences[seq_idx, :, volume_idx] = volumes
    
    return clamped_sequences

def generate_synthetic_data(
    quick_test_mode: bool = False,
    model_path: str = None,
    num_sequences: int = 100,
    output_dir: str = None,
    device: str = None,
    seed: int = None
) -> str:
    """
    Generate synthetic blockchain data from a trained GAN model.
    
    Args:
        quick_test_mode: If True, use smaller model and fewer sequences
        model_path: Path to trained GAN model
        num_sequences: Number of sequences to generate
        output_dir: Directory to save generated data
        device: Device to run generation on (CPU or GPU)
        seed: Random seed for reproducibility
        
    Returns:
        Path to saved synthetic data
    """
    # Select the appropriate config based on mode
    config = QUICK_TEST_GAN_CONFIG if quick_test_mode else GAN_CONFIG
    
    # Use provided parameters if given, otherwise use from config
    model_path = model_path or os.path.join(config['output_dir'], 'gan_model.pt')
    output_dir = output_dir or config['synthetic_data_dir']
    seed = seed or config['seed']
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Loading model from {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    latent_dim = checkpoint['latent_dim']
    condition_dim = checkpoint['condition_dim']
    sequence_length = checkpoint['sequence_length']
    feature_dim = checkpoint['feature_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_heads = checkpoint.get('num_heads', 8)  # Default for older checkpoints
    num_layers = checkpoint.get('num_layers', 4)  # Default for older checkpoints
    
    # Create generator
    generator = Generator(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1   # Default for inference
    )
    
    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set up feature mapping
    feature_mapping = setup_feature_mapping()
    logging.info(f"Feature mapping: {feature_mapping}")
    
    # Create synthetic data generator
    data_generator = SyntheticDataGenerator(
        generator=generator,
        device=device
    )
    
    # Set feature names
    data_generator.set_feature_names(feature_mapping)
    
    # Specify output paths
    os.makedirs(output_dir, exist_ok=True)
    
    # Load normalization parameters
    norm_params_path = os.path.join('off-chain/models', 'normalization_params.json')
    with open(norm_params_path, 'r') as f:
        normalization_params = json.load(f)
    
    # Load original data to extract statistics for clamping
    real_data_path = config['processed_data_path']
    real_data = pd.read_csv(real_data_path)
    
    # Calculate statistics for each feature
    real_data_stats = {}
    for feature in feature_mapping:
        if feature in real_data.columns:
            real_data_stats[feature] = {
                'min': real_data[feature].min(),
                'max': real_data[feature].max(),
                'mean': real_data[feature].mean(),
                'std': real_data[feature].std()
            }
            logging.info(f"Real data stats for {feature}: min={real_data_stats[feature]['min']:.4f}, "
                       f"max={real_data_stats[feature]['max']:.4f}, mean={real_data_stats[feature]['mean']:.4f}, "
                       f"std={real_data_stats[feature]['std']:.4f}")
    
    # Generate synthetic data
    logging.info(f"Generating {num_sequences} synthetic sequences")
    
    # Generate sequences directly
    sequences = data_generator.generate_sequences(
        num_sequences=num_sequences,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        feature_dim=feature_dim,
        batch_size=32
    )
    
    # Denormalize if normalization parameters are provided
    if normalization_params:
        sequences = data_generator.denormalize_sequences(sequences, normalization_params)
        
        # Clamp values based on real data statistics
        sequences = clamp_denormalized_data(
            sequences, 
            real_data_stats
        )
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"synthetic_data_{timestamp}.csv")
    df = save_sequences_to_csv(sequences, None, feature_mapping, sequence_length, output_path)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create visualization using the SAME data that was generated and saved
    vis_path = os.path.join(vis_dir, f"synthetic_data_visualization_{timestamp}.png")
    
    # Use our improved visualization with the same sequences that were saved to CSV
    visualize_financial_sequences(
        sequences=sequences,
        feature_names=feature_mapping,
        save_path=vis_path
    )
    
    logging.info(f"Generated synthetic data saved to {output_path}")
    logging.info(f"Visualization saved to {vis_path}")
    
    # Also create a comparison with real data
    # Sample some real data sequences from the dataset
    real_sequences = []
    
    # Check if 'sequence_id' column exists in real_data
    if 'sequence_id' in real_data.columns:
        unique_sequences = real_data['sequence_id'].unique()
        if len(unique_sequences) > 0:
            # Sample up to 3 real sequences
            sample_ids = unique_sequences[:min(3, len(unique_sequences))]
            
            # Reshape real data into sequences
            for seq_id in sample_ids:
                seq_data = real_data[real_data['sequence_id'] == seq_id]
                if len(seq_data) >= sequence_length:
                    # Extract features in the same order as feature_mapping
                    seq_array = np.zeros((sequence_length, len(feature_mapping)))
                    for i, feature in enumerate(feature_mapping):
                        if feature in seq_data.columns:
                            seq_array[:sequence_length, i] = seq_data[feature].values[:sequence_length]
                    real_sequences.append(seq_array)
    else:
        # If there's no sequence_id, create one sequence from the first sequence_length rows
        if len(real_data) >= sequence_length:
            seq_array = np.zeros((sequence_length, len(feature_mapping)))
            for i, feature in enumerate(feature_mapping):
                if feature in real_data.columns:
                    seq_array[:, i] = real_data[feature].values[:sequence_length]
            real_sequences.append(seq_array)
    
    # Create comparison visualization if we have real sequences
    if real_sequences:
        real_sequences_array = np.array(real_sequences)
        # Sample the same number of generated sequences
        n_samples = min(len(real_sequences), sequences.shape[0])
        
        comparison_path = os.path.join(vis_dir, f"comparison_visualization_{timestamp}.png")
        
        # Create comparison visualization
        visualize_comparison(
            real_sequences=real_sequences_array,
            generated_sequences=sequences[:n_samples],
            feature_names=feature_mapping,
            save_path=comparison_path
        )
        
        logging.info(f"Comparison visualization saved to {comparison_path}")
    
    return output_path

def save_sequences_to_csv(sequences, conditions, feature_names, sequence_length, output_path):
    """Save generated sequences to CSV file with appropriate metadata."""
    # Create a list to store all rows
    all_rows = []
    
    # Current timestamp for base time
    base_timestamp = int(datetime.now().timestamp())
    num_sequences = sequences.shape[0]
    
    for seq_idx, sequence in enumerate(sequences):
        for step_idx, step_data in enumerate(sequence):
            # Create a row dictionary
            row = {
                "sequence_id": seq_idx,
                "step": step_idx,
                "timestamp": base_timestamp + seq_idx * sequence_length + step_idx
            }
            
            # Add feature data
            for feat_idx, feat_name in enumerate(feature_names):
                row[feat_name] = step_data[feat_idx]
            
            # Add volatility target if conditions are provided
            if conditions is not None:
                row["target_volatility"] = conditions[seq_idx, 1]  # Index 1 is volatility
            
            all_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logging.info(f"Saved synthetic data to {output_path}")
    
    return df

def main():
    """Main entry point for GAN training and synthetic data generation."""
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GAN and generate synthetic blockchain data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Create parser for "train" command
    train_parser = subparsers.add_parser("train", help="Train the GAN model")
    train_parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode")
    train_parser.add_argument("--data-path", type=str, help="Path to input data")
    train_parser.add_argument("--output-dir", type=str, help="Directory to save output")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    train_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--sequence-length", type=int, help="Length of sequences")
    train_parser.add_argument("--n-critic", type=int, help="Number of discriminator updates per generator update")
    train_parser.add_argument("--lambda-gp", type=float, help="Weight for gradient penalty")
    train_parser.add_argument("--device", type=str, help="Device to run on (cpu or cuda)")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--noise-std", type=float, default=0.2, help="Standard deviation of instance noise")
    train_parser.add_argument("--diversity-lambda", type=float, default=0.1, help="Weight for diversity loss")
    train_parser.add_argument("--feature-matching-lambda", type=float, default=0.5, help="Weight for feature matching loss")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    
    # Create parser for "generate" command
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    generate_parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode")
    generate_parser.add_argument("--model-path", type=str, help="Path to saved model")
    generate_parser.add_argument("--num-sequences", type=int, default=100, help="Number of sequences to generate")
    generate_parser.add_argument("--output-dir", type=str, help="Directory to save output")
    generate_parser.add_argument("--device", type=str, help="Device to run on (cpu or cuda)")
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set global quick test mode
    quick_test_mode = args.quick_test or QUICK_TEST_MODE
    
    # Run the specified command
    if args.command == "train":
        # Train GAN model
        train_gan(
            quick_test_mode=quick_test_mode,
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sequence_length=args.sequence_length,
            n_critic=args.n_critic,
            lambda_gp=args.lambda_gp,
            device=args.device,
            seed=args.seed,
            # Enhanced parameters
            noise_std=args.noise_std,
            diversity_lambda=args.diversity_lambda,
            feature_matching_lambda=args.feature_matching_lambda,
            dropout=args.dropout
        )
    elif args.command == "generate":
        # Generate synthetic data
        saved_path = generate_synthetic_data(
            quick_test_mode=quick_test_mode,
            model_path=args.model_path,
            num_sequences=args.num_sequences,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed
        )
        logging.info(f"Data generation completed successfully. Data saved to: {saved_path}")


if __name__ == "__main__":
    main() 