import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for financial data visualization."""
    
    # Plot settings
    DEFAULT_FIGURE_SIZE = (15, 9)
    COMPARISON_FIGURE_SIZE = (12, 9)
    LINE_WIDTH = 1.5
    GRID_ALPHA = 0.3
    GRID_STYLE = '--'
    GRID_LINE_WIDTH = 0.5
    
    # Data bounds for basic validation
    MAX_VOLATILITY = 100.0  # Cap extreme volatility values
    VOLUME_SCALE_MAX = 1000.0  # Maximum volume for scaling
    RANGE_PADDING_FACTOR = 0.1  # Padding for plot ranges
    
    # Sample data generation
    SAMPLE_SEQUENCE_LENGTH = 24
    SAMPLE_BASE_PRICE = 2600.0
    SAMPLE_PRICE_NOISE = 5.0
    SAMPLE_VOLUME_BASE = 100.0
    
    # Feature names and labels
    DEFAULT_FEATURES = ['price', 'volume', 'volatility']
    FEATURE_LABELS = {
        'price': 'Price (USD)',
        'volume': 'Volume (ETH)',
        'volatility': 'Volatility (%)'
    }
    
    # Colors for plotting
    PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


class FeatureExtractor:
    """Utility class for extracting and managing feature indices."""
    
    @staticmethod
    def get_feature_indices(feature_names: List[str], target_features: List[str] = None) -> Dict[str, Optional[int]]:
        """Get indices for target features."""
        if target_features is None:
            target_features = VisualizationConfig.DEFAULT_FEATURES
        
        feature_indices = {}
        for feature in target_features:
            try:
                feature_indices[feature] = feature_names.index(feature)
            except ValueError:
                feature_indices[feature] = None
        
        return feature_indices
    
    @staticmethod
    def validate_features(feature_indices: Dict[str, Optional[int]]) -> bool:
        """Check if at least one key feature is available."""
        return any(idx is not None for idx in feature_indices.values())


class SequenceSelector:
    """Handles selection of diverse and representative sequences for visualization."""
    
    @staticmethod
    def select_diverse_sequences(
        sequences: np.ndarray, 
        num_samples: int, 
        feature_indices: Dict[str, Optional[int]]
    ) -> List[int]:
        """Select diverse and representative sequences for visualization."""
        if sequences.shape[0] <= num_samples:
            return list(range(sequences.shape[0]))
        
        price_idx = feature_indices.get('price')
        
        if price_idx is not None:
            return SequenceSelector._select_by_price_movement(sequences, num_samples, price_idx)
        else:
            return list(range(min(num_samples, sequences.shape[0])))
    
    @staticmethod
    def _select_by_price_movement(sequences: np.ndarray, num_samples: int, price_idx: int) -> List[int]:
        """Select sequences based on price movement diversity."""
        # Calculate price movement for each sequence
        price_movements = np.array([
            np.max(seq[:, price_idx]) - np.min(seq[:, price_idx])
            for seq in sequences
        ])
        
        if len(price_movements) < 3:
            return list(range(min(num_samples, len(sequences))))
        
        # Get sequences with low, medium, and high price movement
        sorted_indices = np.argsort(price_movements)
        quartiles = [
            len(sorted_indices) // 4,
            len(sorted_indices) // 2,
            3 * len(sorted_indices) // 4
        ]
        
        selected_indices = [sorted_indices[q] for q in quartiles]
        return selected_indices[:num_samples]


class DataProcessor:
    """Handles basic data validation and scaling for visualization."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def process_for_visualization(
        self, 
        sequences: np.ndarray, 
        selected_indices: List[int], 
        feature_indices: Dict[str, Optional[int]]
    ) -> np.ndarray:
        """Apply basic validation and scaling to sequences for visualization."""
        sequences_copy = sequences.copy()
        
        for idx in selected_indices:
            sequences_copy = self._validate_sequence(sequences_copy, idx, feature_indices)
        
        return sequences_copy
    
    def _validate_sequence(
        self, 
        sequences: np.ndarray, 
        seq_idx: int, 
        feature_indices: Dict[str, Optional[int]]
    ) -> np.ndarray:
        """Apply basic validation to a single sequence."""
        volume_idx = feature_indices.get('volume')
        volatility_idx = feature_indices.get('volatility')
        
        # Basic volume validation - ensure non-negative
        if volume_idx is not None:
            sequences[seq_idx, :, volume_idx] = np.abs(sequences[seq_idx, :, volume_idx])
        
        # Basic volatility validation - ensure non-negative and cap extremes
        if volatility_idx is not None:
            volatility_data = np.abs(sequences[seq_idx, :, volatility_idx])
            volatility_data = np.minimum(volatility_data, self.config.MAX_VOLATILITY)
            sequences[seq_idx, :, volatility_idx] = volatility_data
        
        return sequences


class VisualizationPlotter:
    """Handles the actual plotting and visualization creation."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def create_sequences_plot(
        self,
        sequences: np.ndarray,
        feature_names: List[str],
        selected_indices: List[int],
        scale_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization for multiple sequences."""
        num_samples = len(selected_indices)
        feature_indices = FeatureExtractor.get_feature_indices(feature_names)
        
        # Create subplots
        fig, axes = plt.subplots(3, num_samples, figsize=self.config.DEFAULT_FIGURE_SIZE)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        # Determine y-axis ranges
        y_ranges = self._calculate_y_ranges(sequences, selected_indices, feature_indices, scale_ranges)
        
        # Plot each feature for each sample
        features_to_plot = self.config.DEFAULT_FEATURES
        for i, seq_idx in enumerate(selected_indices):
            for j, feature in enumerate(features_to_plot):
                self._plot_single_feature(
                    axes[j, i], sequences[seq_idx], feature, feature_indices, 
                    y_ranges.get(feature), i+1
                )
        
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def create_comparison_plot(
        self,
        real_sequences: np.ndarray,
        generated_sequences: np.ndarray,
        feature_names: List[str],
        selected_indices: List[int],
        save_path: Optional[str] = None
    ) -> None:
        """Create comparison visualization between real and generated data."""
        feature_indices = FeatureExtractor.get_feature_indices(feature_names)
        features_to_plot = [(idx, name) for name, idx in feature_indices.items() if idx is not None][:3]
        
        if not features_to_plot:
            features_to_plot = [(i, feature_names[i]) for i in range(min(3, len(feature_names)))]
        
        # Create subplots
        fig, axes = plt.subplots(len(features_to_plot), 2, figsize=self.config.COMPARISON_FIGURE_SIZE)
        if len(features_to_plot) == 1:
            axes = axes.reshape(1, -1)
        
        # Calculate consistent y-ranges
        y_ranges = self._calculate_comparison_y_ranges(
            real_sequences, generated_sequences, selected_indices, features_to_plot
        )
        
        # Plot comparisons
        for i, (feat_idx, feat_name) in enumerate(features_to_plot):
            self._plot_comparison_feature(
                axes[i], real_sequences, generated_sequences, selected_indices,
                feat_idx, feat_name, y_ranges[i]
            )
        
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def _calculate_y_ranges(
        self,
        sequences: np.ndarray,
        selected_indices: List[int],
        feature_indices: Dict[str, Optional[int]],
        scale_ranges: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate y-axis ranges for consistent plotting."""
        if scale_ranges:
            return scale_ranges
        
        y_ranges = {}
        for feature in self.config.DEFAULT_FEATURES:
            feat_idx = feature_indices.get(feature)
            if feat_idx is not None:
                data = np.array([sequences[idx, :, feat_idx] for idx in selected_indices])
                min_val, max_val = np.min(data), np.max(data)
                padding = (max_val - min_val) * self.config.RANGE_PADDING_FACTOR
                y_ranges[feature] = (min_val - padding, max_val + padding)
        
        return y_ranges
    
    def _calculate_comparison_y_ranges(
        self,
        real_sequences: np.ndarray,
        generated_sequences: np.ndarray,
        selected_indices: List[int],
        features_to_plot: List[Tuple[int, str]]
    ) -> List[Tuple[float, float]]:
        """Calculate y-axis ranges for comparison plots."""
        y_ranges = []
        n_samples = len(selected_indices)
        
        for feat_idx, feat_name in features_to_plot:
            real_data = np.array([real_sequences[j, :, feat_idx] for j in range(n_samples)])
            gen_data = np.array([generated_sequences[j, :, feat_idx] for j in range(n_samples)])
            all_data = np.vstack([real_data, gen_data])
            
            min_val = np.nanmin(all_data) if np.any(~np.isnan(all_data)) else 0
            max_val = np.nanmax(all_data) if np.any(~np.isnan(all_data)) else 100
            
            # Cap volatility range for reasonable display
            if feat_name == 'volatility':
                max_val = min(max_val, self.config.MAX_VOLATILITY)
            
            padding = (max_val - min_val) * self.config.RANGE_PADDING_FACTOR
            y_ranges.append((min_val - padding, max_val + padding))
        
        return y_ranges
    
    def _plot_single_feature(
        self,
        ax: plt.Axes,
        sequence: np.ndarray,
        feature: str,
        feature_indices: Dict[str, Optional[int]],
        y_range: Optional[Tuple[float, float]],
        sample_num: int
    ) -> None:
        """Plot a single feature for one sequence."""
        feat_idx = feature_indices.get(feature)
        
        if feat_idx is not None:
            ax.plot(sequence[:, feat_idx], 'b-', linewidth=self.config.LINE_WIDTH)
            ax.grid(True, alpha=self.config.GRID_ALPHA, linestyle=self.config.GRID_STYLE, 
                   linewidth=self.config.GRID_LINE_WIDTH)
            ax.set_title(f"Sample {sample_num}: {feature}")
            ax.set_xlabel("Trading Day")
            ax.set_ylabel(self.config.FEATURE_LABELS.get(feature, feature))
            
            if y_range:
                ax.set_ylim(y_range)
        else:
            ax.text(0.5, 0.5, f"{feature} data\nnot available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
    
    def _plot_comparison_feature(
        self,
        axes: np.ndarray,
        real_sequences: np.ndarray,
        generated_sequences: np.ndarray,
        selected_indices: List[int],
        feat_idx: int,
        feat_name: str,
        y_range: Tuple[float, float]
    ) -> None:
        """Plot comparison for a single feature."""
        n_samples = len(selected_indices)
        
        # Real data
        ax_real = axes[0]
        for j in range(n_samples):
            color = self.config.PLOT_COLORS[j % len(self.config.PLOT_COLORS)]
            ax_real.plot(real_sequences[j, :, feat_idx], color=color, alpha=0.8, 
                        label=f"Sample {j+1}", linewidth=self.config.LINE_WIDTH)
        
        self._style_comparison_axis(ax_real, f"Real: {feat_name}", feat_name, y_range, show_legend=True)
        
        # Generated data
        ax_gen = axes[1]
        for j in range(n_samples):
            color = self.config.PLOT_COLORS[j % len(self.config.PLOT_COLORS)]
            ax_gen.plot(generated_sequences[j, :, feat_idx], color=color, alpha=0.8,
                       label=f"Sample {j+1}", linewidth=self.config.LINE_WIDTH)
        
        self._style_comparison_axis(ax_gen, f"Generated: {feat_name}", feat_name, y_range, show_legend=True)
    
    def _style_comparison_axis(
        self,
        ax: plt.Axes,
        title: str,
        feat_name: str,
        y_range: Tuple[float, float],
        show_legend: bool = False
    ) -> None:
        """Apply consistent styling to comparison plot axes."""
        ax.set_title(title)
        ax.set_xlabel("Trading Day")
        ax.set_ylabel(self.config.FEATURE_LABELS.get(feat_name, feat_name))
        ax.grid(True, alpha=self.config.GRID_ALPHA, linestyle=self.config.GRID_STYLE)
        ax.set_ylim(y_range)
        
        if show_legend:
            ax.legend()
    
    def _save_or_show(self, save_path: Optional[str]) -> None:
        """Save plot to file or show it."""
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()


class SampleDataGenerator:
    """Generates realistic sample data for testing visualizations."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def generate_sample_sequences(self, n_samples: int = 3) -> Tuple[np.ndarray, List[str]]:
        """Generate realistic sample sequences for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        sequences = np.zeros((n_samples, self.config.SAMPLE_SEQUENCE_LENGTH, len(self.config.DEFAULT_FEATURES)))
        feature_names = self.config.DEFAULT_FEATURES.copy()
        
        for i in range(n_samples):
            sequences[i] = self._generate_single_sequence(i)
        
        return sequences, feature_names
    
    def _generate_single_sequence(self, sample_idx: int) -> np.ndarray:
        """Generate a single realistic sequence."""
        seq_length = self.config.SAMPLE_SEQUENCE_LENGTH
        sequence = np.zeros((seq_length, 3))  # price, volume, volatility
        
        # Generate price data
        sequence[:, 0] = self._generate_price_data(sample_idx, seq_length)
        
        # Generate volume data
        sequence[:, 1] = self._generate_volume_data(sample_idx, seq_length)
        
        # Generate volatility data
        sequence[:, 2] = self._generate_volatility_data(sample_idx, seq_length, sequence[:, 0])
        
        return sequence
    
    def _generate_price_data(self, sample_idx: int, seq_length: int) -> np.ndarray:
        """Generate realistic price data with different patterns."""
        base_price = self.config.SAMPLE_BASE_PRICE + np.random.randn() * self.config.SAMPLE_PRICE_NOISE
        prices = np.zeros(seq_length)
        prices[0] = base_price
        
        if sample_idx == 0:  # Upward trend
            for t in range(1, seq_length):
                prices[t] = prices[t-1] + np.random.randn() * 2 + 0.2
        elif sample_idx == 1:  # Downward trend
            for t in range(1, seq_length):
                prices[t] = prices[t-1] + np.random.randn() * 2 - 0.5
        else:  # Fluctuating
            for t in range(1, seq_length):
                if t % 5 == 0:
                    prices[t] = prices[t-1] + np.random.randn() * 4 + np.random.choice([-1, 1]) * 2
                else:
                    prices[t] = prices[t-1] + np.random.randn() * 1.5
        
        return prices
    
    def _generate_volume_data(self, sample_idx: int, seq_length: int) -> np.ndarray:
        """Generate realistic volume data."""
        if sample_idx == 0:  # Fluctuating around base
            volumes = self.config.SAMPLE_VOLUME_BASE - 5 + np.cumsum(np.random.randn(seq_length) * 0.3)
            volumes = np.maximum(98, volumes)
        elif sample_idx == 1:  # Decreasing trend
            base_trend = np.linspace(self.config.SAMPLE_VOLUME_BASE, 20, seq_length)
            volumes = base_trend + np.random.randn(seq_length) * 3
        else:  # Variable
            volumes = self.config.SAMPLE_VOLUME_BASE - 2 + np.cumsum(np.random.randn(seq_length) * 0.4)
            volumes = np.maximum(98, volumes)
        
        return volumes
    
    def _generate_volatility_data(self, sample_idx: int, seq_length: int, price_data: np.ndarray) -> np.ndarray:
        """Generate volatility data correlated with price changes."""
        volatility = np.zeros(seq_length)
        volatility[0] = 1.0 + np.random.rand() * 2
        
        for t in range(1, seq_length):
            price_change = abs(price_data[t] - price_data[t-1])
            
            if sample_idx == 0:  # Normal volatility
                base_vol = 4.0 + price_change * 3
                if t % 5 == 0:
                    volatility[t] = base_vol + 4.0 + np.random.rand() * 2
                else:
                    volatility[t] = base_vol + np.random.rand() * 2
            elif sample_idx == 1:  # Higher volatility
                base_vol = 8.0 + price_change * 4
                if t % 4 == 0:
                    volatility[t] = base_vol + 8.0 + np.random.rand() * 4
                else:
                    volatility[t] = base_vol + np.random.rand() * 3
            else:  # Moderate volatility
                base_vol = 5.0 + price_change * 2
                if t % 5 == 0:
                    volatility[t] = base_vol + 6.0 + np.random.rand() * 3
                else:
                    volatility[t] = base_vol + np.random.rand() * 2
        
        return volatility


# Main API functions
def visualize_financial_sequences(
    sequences: np.ndarray,
    feature_names: List[str],
    num_samples: int = 3,
    save_path: Optional[str] = None,
    scale_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> None:
    """
    Create visualizations for financial time series data.
    
    Args:
        sequences: Generated sequences to visualize, shape (batch_size, seq_length, features)
        feature_names: List of feature names for each feature dimension
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization, if None, will show the plot
        scale_ranges: Optional dictionary with predefined y-axis ranges for each feature
    """
    if sequences.size == 0:
        logging.warning("Empty sequences provided for visualization")
        return
    
    # Initialize components
    config = VisualizationConfig()
    feature_indices = FeatureExtractor.get_feature_indices(feature_names)
    
    if not FeatureExtractor.validate_features(feature_indices):
        logging.warning("No recognized features found for visualization")
        return
    
    # Select and process sequences
    num_to_show = min(num_samples, sequences.shape[0])
    selected_indices = SequenceSelector.select_diverse_sequences(sequences, num_to_show, feature_indices)
    
    # Apply basic validation (no artificial fixing)
    processor = DataProcessor(config)
    processed_sequences = processor.process_for_visualization(sequences, selected_indices, feature_indices)
    
    # Create visualization
    plotter = VisualizationPlotter(config)
    plotter.create_sequences_plot(processed_sequences, feature_names, selected_indices, scale_ranges, save_path)


def visualize_comparison(
    real_sequences: np.ndarray,
    generated_sequences: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison visualizations between real and generated data.
    
    Args:
        real_sequences: Real data sequences
        generated_sequences: Generated sequences
        feature_names: List of feature names
        save_path: Path to save the visualization
    """
    if real_sequences.size == 0 or generated_sequences.size == 0:
        logging.warning("Empty sequences provided for comparison visualization")
        return
    
    # Initialize components
    config = VisualizationConfig()
    feature_indices = FeatureExtractor.get_feature_indices(feature_names)
    
    # Determine number of samples
    n_samples = min(3, real_sequences.shape[0], generated_sequences.shape[0])
    selected_indices = list(range(n_samples))
    
    # Apply basic validation to both datasets (no artificial fixing)
    processor = DataProcessor(config)
    real_processed = processor.process_for_visualization(real_sequences, selected_indices, feature_indices)
    generated_processed = processor.process_for_visualization(generated_sequences, selected_indices, feature_indices)
    
    # Create comparison visualization
    plotter = VisualizationPlotter(config)
    plotter.create_comparison_plot(real_processed, generated_processed, feature_names, selected_indices, save_path)


def generate_sample_visualization() -> str:
    """Generate a sample visualization for testing."""
    config = VisualizationConfig()
    generator = SampleDataGenerator(config)
    
    # Generate sample data
    sequences, feature_names = generator.generate_sample_sequences()
    
    # Create output directory and path
    output_dir = Path("data/synthetic/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"sample_visualization_{timestamp}.png"
    
    # Generate visualization
    visualize_financial_sequences(
        sequences=sequences,
        feature_names=feature_names,
        save_path=str(save_path)
    )
    
    return str(save_path)


if __name__ == "__main__":
    # Generate a sample visualization
    vis_path = generate_sample_visualization()
    print(f"Generated sample visualization at: {vis_path}") 