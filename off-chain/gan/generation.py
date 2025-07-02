import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from .models import Generator


class SyntheticDataGenerator:
    """
    Utility class for generating synthetic blockchain data using a trained GAN model.
    """
    
    # Constants
    MIN_VOLUME_THRESHOLD = 0.1
    ZERO_VOLUME_THRESHOLD = 0.001
    DEFAULT_NUM_SEQUENCES_TO_SHOW = 3
    DEFAULT_FIGURE_SIZE = (15, 3)
    
    def __init__(
        self,
        generator: Generator,
        device: torch.device
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            generator: Trained generator model
            device: Device to run generation on (CPU or GPU)
        """
        self.generator = generator.to(device)
        self.generator.eval()  # Set to evaluation mode
        self.device = device
        
        # Default feature names for Uniswap V4
        self.feature_names = [
            'price', 'amount0_eth', 'amount1_usdc', 'tick', 'volume', 
            'volatility', 'volume_ma', 'tick_change', 'time_diff', 'liquidity_usage'
        ]
        
        self.output_dir = Path("data/synthetic")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Metadata for generated data
        self.metadata = {
            "model_type": "GAN",
            "generation_time": datetime.now().isoformat(),
            "feature_names": self.feature_names
        }
        
        logging.info(f"SyntheticDataGenerator initialized")
    
    def generate_sequences(
        self,
        num_sequences: int,
        latent_dim: int,
        condition_dim: int,
        feature_dim: int,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate synthetic sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            latent_dim: Dimension of latent space
            condition_dim: Dimension of condition vector
            feature_dim: Dimension of feature space
            batch_size: Batch size for generation
            
        Returns:
            Generated sequences as a numpy array
            
        Raises:
            ValueError: If num_sequences <= 0 or dimensions are invalid
        """
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if latent_dim <= 0 or condition_dim <= 0 or feature_dim <= 0:
            raise ValueError("All dimensions must be positive")
            
        # Set generator to evaluation mode
        self.generator.eval()
        
        # Create random conditions
        conditions = torch.randn(num_sequences, condition_dim).to(self.device)
        
        # Generate sequences in batches
        generated_sequences = []
        
        with torch.no_grad():
            for i in range(0, num_sequences, batch_size):
                # Get batch size for this iteration
                current_batch_size = min(batch_size, num_sequences - i)
                
                # Generate random noise
                z = torch.randn(current_batch_size, latent_dim).to(self.device)
                
                # Get conditions for this batch
                batch_conditions = conditions[i:i+current_batch_size].to(self.device)
                
                # Generate sequences
                batch_sequences = self.generator(z, batch_conditions)
                
                # Add to list
                generated_sequences.append(batch_sequences.cpu())
        
        # Concatenate all batches
        all_sequences = torch.cat(generated_sequences, dim=0)
        
        logging.info(f"Generated {num_sequences} synthetic sequences")
        
        return all_sequences.numpy()
    
    def denormalize_sequences(
        self,
        sequences: Union[torch.Tensor, np.ndarray],
        normalization_params: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """
        Denormalize generated sequences using the provided normalization parameters.
        
        Args:
            sequences: Generated sequences (normalized)
            normalization_params: Dictionary with normalization parameters for each feature
            
        Returns:
            Denormalized sequences as numpy array
            
        Raises:
            ValueError: If sequences shape is incompatible with feature names
        """
        # Convert to numpy for easier manipulation
        sequences_np = sequences.numpy() if isinstance(sequences, torch.Tensor) else sequences
        
        if sequences_np.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got {sequences_np.ndim}D")
        if sequences_np.shape[2] != len(self.feature_names):
            raise ValueError(f"Sequence feature dimension {sequences_np.shape[2]} doesn't match feature names {len(self.feature_names)}")
        
        # Get feature indices
        feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        
        # Denormalize each feature
        denormalized_sequences = np.zeros_like(sequences_np)
        
        for feature_name, params in normalization_params.items():
            if feature_name in feature_indices:
                idx = feature_indices[feature_name]
                
                # Get normalization parameters
                mean = params.get("mean", 0)
                std = params.get("std", 1)
                transform_type = params.get("transform", "standard")
                
                # Apply the correct inverse transformation based on transform_type
                if transform_type == "log":
                    # Step 1: Undo Z-score normalization
                    normalized_log_values = sequences_np[:, :, idx] * std + mean
                    # Step 2: Undo log transform using expm1 (exp(x) - 1)
                    denormalized_sequences[:, :, idx] = np.expm1(normalized_log_values)
                elif transform_type == "signed_log":
                    # Step 1: Undo Z-score normalization
                    normalized_log_values = sequences_np[:, :, idx] * std + mean
                    # Step 2: Extract sign and magnitude
                    sign = np.sign(normalized_log_values)
                    magnitude = np.abs(normalized_log_values)
                    # Step 3: Undo log transform on magnitude, then reapply sign
                    denormalized_sequences[:, :, idx] = sign * np.expm1(magnitude)
                else:
                    # Standard denormalization (Z-score)
                    denormalized_sequences[:, :, idx] = sequences_np[:, :, idx] * std + mean
                
                # Log denormalization details
                if idx == 0:  # Only log for the first feature to avoid excessive output
                    orig_range = f"[{sequences_np[:, :, idx].min():.4f}, {sequences_np[:, :, idx].max():.4f}]"
                    denorm_range = f"[{denormalized_sequences[:, :, idx].min():.4f}, {denormalized_sequences[:, :, idx].max():.4f}]"
                    logging.debug(f"Denormalizing {feature_name} ({transform_type}): {orig_range} -> {denorm_range}")
        
        return denormalized_sequences
    
    def sequences_to_dataframe(
        self,
        sequences: np.ndarray,
        start_timestamp: Optional[int] = None,
        timestamp_increment: int = 1
    ) -> pd.DataFrame:
        """
        Convert generated sequences to a pandas DataFrame.
        
        Args:
            sequences: Generated sequences (denormalized)
            start_timestamp: Starting timestamp for the sequences
            timestamp_increment: Increment between timestamps
            
        Returns:
            DataFrame with generated data
            
        Raises:
            ValueError: If sequences shape is invalid
        """
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got {sequences.ndim}D")
            
        # Number of sequences and sequence length
        num_sequences = sequences.shape[0]
        seq_length = sequences.shape[1]
        
        # Create empty DataFrame
        df_data = []
        
        # Current timestamp
        current_timestamp = start_timestamp if start_timestamp is not None else int(datetime.now().timestamp())
        
        # Convert sequences to DataFrame
        for seq_idx in range(num_sequences):
            for step_idx in range(seq_length):
                # Create row
                row = {
                    "sequence_id": seq_idx,
                    "step": step_idx,
                    "timestamp": current_timestamp + step_idx * timestamp_increment
                }
                
                # Add features
                for feat_idx, feat_name in enumerate(self.feature_names):
                    if feat_idx < sequences.shape[2]:
                        row[feat_name] = sequences[seq_idx, step_idx, feat_idx]
                
                df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        return df
    
    def save_generated_data(
        self,
        sequences: Union[torch.Tensor, np.ndarray],
        filename: str,
        normalization_params: Optional[Dict] = None,
        start_timestamp: Optional[int] = None,
        timestamp_increment: int = 1,
        format: str = "csv"
    ) -> str:
        """
        Save generated sequences to file.
        
        Args:
            sequences: Generated sequences
            filename: Base filename for saving
            normalization_params: Normalization parameters for denormalization
            start_timestamp: Starting timestamp for the sequences
            timestamp_increment: Increment between timestamps
            format: Output format (csv or pickle)
            
        Returns:
            Path to saved file
            
        Raises:
            ValueError: If format is unsupported
        """
        if format.lower() not in ["csv", "pickle"]:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'pickle'")
            
        # Convert tensor to numpy if needed
        if isinstance(sequences, torch.Tensor):
            sequences_np = sequences.cpu().numpy()
        else:
            sequences_np = sequences
        
        # Denormalize if normalization parameters are provided
        if normalization_params:
            denormalized_sequences = self.denormalize_sequences(sequences, normalization_params)
        else:
            denormalized_sequences = sequences_np
        
        # Convert to DataFrame
        df = self.sequences_to_dataframe(
            denormalized_sequences,
            start_timestamp=start_timestamp,
            timestamp_increment=timestamp_increment
        )
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}"
        
        # Save metadata
        metadata_path = f"{output_path}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save data
        if format.lower() == "csv":
            csv_path = f"{output_path}.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved generated data to {csv_path}")
            return csv_path
        elif format.lower() == "pickle":
            pickle_path = f"{output_path}.pkl"
            df.to_pickle(pickle_path)
            logging.info(f"Saved generated data to {pickle_path}")
            return pickle_path
    
    def _get_feature_indices(self, features_to_plot: Optional[List[str]]) -> List[int]:
        """Get indices for features to plot."""
        if features_to_plot is None:
            return list(range(min(3, len(self.feature_names))))
        
        feature_indices = []
        for feat in features_to_plot:
            if feat in self.feature_names:
                feature_indices.append(self.feature_names.index(feat))
        
        # Fallback if no valid features found
        if not feature_indices:
            feature_indices = list(range(min(3, len(self.feature_names))))
            
        return feature_indices
    
    def _select_representative_sequences(self, sequences: np.ndarray, num_to_show: int) -> List[int]:
        """Select representative sequences for visualization."""
        selected_indices = []
        
        # Try to find sequences with meaningful financial relationships
        try:
            price_idx = self.feature_names.index('price')
            volume_idx = self.feature_names.index('volume')
            
            # Find sequences with non-zero volume
            for i in range(sequences.shape[0]):
                volume_sum = np.sum(np.abs(sequences[i, :, volume_idx]))
                if volume_sum > self.MIN_VOLUME_THRESHOLD:
                    selected_indices.append(i)
                    if len(selected_indices) >= num_to_show:
                        break
                        
            # Fill remaining slots with random sequences
            if len(selected_indices) < num_to_show:
                remaining = num_to_show - len(selected_indices)
                available_indices = [i for i in range(sequences.shape[0]) if i not in selected_indices]
                if available_indices:
                    random_indices = np.random.choice(
                        available_indices,
                        size=min(remaining, len(available_indices)), 
                        replace=False
                    )
                    selected_indices.extend(random_indices)
        except ValueError:
            # If we don't have price/volume features, just use random sequences
            selected_indices = np.random.choice(
                sequences.shape[0], 
                size=min(num_to_show, sequences.shape[0]), 
                replace=False
            ).tolist()
            
        return selected_indices
    
    def _apply_financial_consistency(self, sequences: np.ndarray) -> np.ndarray:
        """Apply financial consistency rules to sequences (returns a copy)."""
        # Create a copy to avoid modifying the original
        consistent_sequences = sequences.copy()
        
        try:
            price_idx = self.feature_names.index('price')
            volume_idx = self.feature_names.index('volume')
            volatility_idx = self.feature_names.index('volatility') if 'volatility' in self.feature_names else None
            
            # Apply consistency rules: if volume is zero, price shouldn't change
            for seq_idx in range(consistent_sequences.shape[0]):
                for t in range(1, consistent_sequences.shape[1]):
                    volume = consistent_sequences[seq_idx, t, volume_idx]
                    
                    if abs(volume) <= self.ZERO_VOLUME_THRESHOLD:
                        # Keep price unchanged from previous timestep
                        consistent_sequences[seq_idx, t, price_idx] = consistent_sequences[seq_idx, t-1, price_idx]
                        # Set volatility to zero if it exists
                        if volatility_idx is not None:
                            consistent_sequences[seq_idx, t, volatility_idx] = 0
                            
        except ValueError:
            # If features don't exist, return sequences as-is
            pass
            
        return consistent_sequences
    
    def _create_subplot_grid(self, feature_indices: List[int], num_sequences: int) -> Tuple[plt.Figure, np.ndarray]:
        """Create and configure subplot grid."""
        fig, axes = plt.subplots(
            len(feature_indices), 
            num_sequences, 
            figsize=(self.DEFAULT_FIGURE_SIZE[0], self.DEFAULT_FIGURE_SIZE[1] * len(feature_indices))
        )
        
        # Ensure axes is always a 2D array for consistent indexing
        if len(feature_indices) == 1 and num_sequences == 1:
            axes = np.array([[axes]])
        elif len(feature_indices) == 1:
            axes = np.array([axes])
        elif num_sequences == 1:
            axes = np.array([[ax] for ax in axes])
            
        return fig, axes
    
    def _plot_sequences(self, axes: np.ndarray, sequences: np.ndarray, selected_indices: List[int], feature_indices: List[int]) -> None:
        """Plot sequences on the provided axes."""
        for i, seq_idx in enumerate(selected_indices):
            sequence = sequences[seq_idx]
            for j, feature_idx in enumerate(feature_indices):
                ax = axes[j, i]
                ax.plot(sequence[:, feature_idx], 'b-', linewidth=1.5)
                ax.set_title(f"Sample {i+1}: {self.feature_names[feature_idx]}", fontsize=10)
                ax.set_xlabel("Time Step", fontsize=9)
                ax.set_ylabel(self.feature_names[feature_idx], fontsize=9)
                ax.grid(True, alpha=0.3)
    
    def visualize_generated_sequences(
        self, 
        sequences: Union[torch.Tensor, np.ndarray], 
        features_to_plot: Optional[List[str]] = None, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize a few generated sequences to see the patterns and quality.
        
        Args:
            sequences: Generated sequences to visualize
            features_to_plot: List of feature names to plot, defaults to first 3 features
            save_path: Path to save the visualization, if None, will show the plot
            
        Raises:
            ValueError: If sequences is empty or has invalid shape
        """
        # Convert to numpy if needed
        if isinstance(sequences, torch.Tensor):
            sequences_np = sequences.cpu().numpy()
        else:
            sequences_np = sequences.copy()  # Make a copy to avoid side effects
            
        if sequences_np.size == 0:
            raise ValueError("Cannot visualize empty sequences")
        if sequences_np.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got {sequences_np.ndim}D")
            
        # Ensure we have feature names
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = [f"Feature {i}" for i in range(sequences_np.shape[2])]
            
        # Get feature indices to plot
        feature_indices = self._get_feature_indices(features_to_plot)
        
        # Number of sequences to show
        num_to_show = min(self.DEFAULT_NUM_SEQUENCES_TO_SHOW, sequences_np.shape[0])
        
        # Select representative sequences
        selected_indices = self._select_representative_sequences(sequences_np, num_to_show)
        
        # Apply financial consistency rules (on a copy)
        consistent_sequences = self._apply_financial_consistency(sequences_np)
        
        # Create subplot grid
        fig, axes = self._create_subplot_grid(feature_indices, len(selected_indices))
        
        # Plot sequences
        self._plot_sequences(axes, consistent_sequences, selected_indices, feature_indices)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_and_save_batch(
        self,
        num_sequences: int,
        normalization_params: Optional[Dict] = None,
        filename: str = "synthetic_data",
        batch_size: int = 32,
        seed: Optional[int] = None,
        visualize: bool = True,
        features_to_plot: Optional[List[str]] = None
    ) -> str:
        """
        Generate, visualize, and save a batch of synthetic sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            normalization_params: Normalization parameters for denormalization
            filename: Base filename for saving
            batch_size: Batch size for generation
            seed: Random seed for reproducibility
            visualize: Whether to visualize generated sequences
            features_to_plot: List of feature names to plot
            
        Returns:
            Path to saved file
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Generate sequences
        sequences = self.generate_sequences(
            num_sequences=num_sequences,
            latent_dim=self.generator.latent_dim,
            condition_dim=self.generator.condition_dim,
            feature_dim=self.generator.feature_dim,
            batch_size=batch_size
        )
        
        # Visualize if requested
        if visualize:
            # Create visualization directory
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True, parents=True)
            
            # Create visualization path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_path = vis_dir / f"{filename}_visualization_{timestamp}.png"
            
            # Visualize
            self.visualize_generated_sequences(
                sequences=sequences,
                features_to_plot=features_to_plot,
                save_path=str(vis_path)
            )
        
        # Save generated data
        saved_path = self.save_generated_data(
            sequences=sequences,
            filename=filename,
            normalization_params=normalization_params
        )
        
        return saved_path
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set the feature names for the generator.
        
        Args:
            feature_names: List of feature names
            
        Raises:
            ValueError: If feature_names is empty
        """
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
            
        self.feature_names = feature_names
        self.metadata["feature_names"] = feature_names
        logging.info(f"Set feature names: {feature_names}") 