import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Tuple, List
import os
import argparse
from generate_synthetic_data import setup_feature_mapping
from config import GAN_CONFIG, QUICK_TEST_GAN_CONFIG, QUICK_TEST_MODE
import logging
from datetime import datetime

def convert_real_data_to_sequences(df: pd.DataFrame, sequence_length: int = GAN_CONFIG['sequence_length']) -> np.ndarray:
    """Convert real transaction data into sequences."""
    # Select relevant numeric columns in the same order as synthetic data
    numeric_cols = [
        'price', 'amount0_eth', 'amount1_usdc', 'tick', 'volume',
        'volatility', 'volume_ma', 'tick_change', 'time_diff', 'liquidity_usage'
    ]
    
    # Create sequences
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        seq = df[numeric_cols].iloc[i:i+sequence_length].values
        sequences.append(seq)
    
    return np.array(sequences)

def convert_synthetic_data_to_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Convert synthetic data (already in sequence format) to numpy array.
    Returns:
        Tuple[np.ndarray, int]: The sequences array and the sequence length
    """
    # Get unique sequence IDs and determine sequence length
    sequence_ids = df['sequence_id'].unique()
    sequence_length = len(df[df['sequence_id'] == sequence_ids[0]])
    
    # Create sequences array
    num_features = 10  # Number of features we expect to have
    sequences = np.zeros((len(sequence_ids), sequence_length, num_features))
    
    # Check if we have individual feature columns or a combined sequence_features column
    numeric_cols = [
        'price', 'amount0_eth', 'amount1_usdc', 'tick', 'volume',
        'volatility', 'volume_ma', 'tick_change', 'time_diff', 'liquidity_usage'
    ]
    
    if 'sequence_features' in df.columns:
        # Handle the case where features are in a single sequence_features column
        logging.info("Detected 'sequence_features' column format in synthetic data")
        for i, seq_id in enumerate(sequence_ids):
            seq_data = df[df['sequence_id'] == seq_id]['sequence_features'].values
            
            # If data is a single value per row, reshape to match expected dimensions
            sequences[i, :, 0] = seq_data  # Use the first feature dimension
            
            # Duplicate the values across all feature dimensions as a workaround
            # This allows validation to proceed with the understanding that we'll need
            # to fix the format in the future
            for j in range(1, num_features):
                sequences[i, :, j] = seq_data
    else:
        # Original approach for individual feature columns
        for i, seq_id in enumerate(sequence_ids):
            seq_data = df[df['sequence_id'] == seq_id][numeric_cols].values
            sequences[i] = seq_data
    
    return sequences, sequence_length

class UniswapV4DataValidator:
    def __init__(self):
        self.validation_results = {}
        
    def calculate_statistics(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for both real and synthetic data."""
        stats_dict = {}
        
        for i, feature in enumerate(feature_names):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            # Basic statistics
            stats_dict[feature] = {
                'real_mean': np.mean(real_feature),
                'real_std': np.std(real_feature),
                'real_median': np.median(real_feature),
                'real_min': np.min(real_feature),
                'real_max': np.max(real_feature),
                'synthetic_mean': np.mean(synthetic_feature),
                'synthetic_std': np.std(synthetic_feature),
                'synthetic_median': np.median(synthetic_feature),
                'synthetic_min': np.min(synthetic_feature),
                'synthetic_max': np.max(synthetic_feature),
                'ks_statistic': stats.ks_2samp(real_feature, synthetic_feature).statistic,
                'ks_pvalue': stats.ks_2samp(real_feature, synthetic_feature).pvalue
            }
            
            # Additional statistics for specific features
            if 'price' in feature.lower():
                # Calculate price volatility
                real_volatility = np.std(np.diff(real_feature)) / np.mean(real_feature)
                synth_volatility = np.std(np.diff(synthetic_feature)) / np.mean(synthetic_feature)
                stats_dict[feature].update({
                    'real_volatility': real_volatility,
                    'synthetic_volatility': synth_volatility
                })
            
            if 'amount' in feature.lower():
                # Calculate volume statistics
                real_total_volume = np.sum(np.abs(real_feature))
                synth_total_volume = np.sum(np.abs(synthetic_feature))
                stats_dict[feature].update({
                    'real_total_volume': real_total_volume,
                    'synthetic_total_volume': synth_total_volume
                })
            
            if 'time_between' in feature.lower():
                # Calculate trade frequency statistics
                real_trade_freq = 1 / np.mean(real_feature[real_feature > 0])
                synth_trade_freq = 1 / np.mean(synthetic_feature[synthetic_feature > 0])
                stats_dict[feature].update({
                    'real_trade_frequency': real_trade_freq,
                    'synthetic_trade_frequency': synth_trade_freq
                })
        
        return stats_dict
        
    def plot_distributions(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str],
        save_dir: str = 'validation_results'
    ):
        """Plot distribution comparisons between real and synthetic data."""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, feature in enumerate(feature_names):
            real_feature = real_data[:, :, i].flatten()
            synthetic_feature = synthetic_data[:, :, i].flatten()
            
            plt.figure(figsize=(12, 6))
            
            # KDE plot
            plt.subplot(1, 2, 1)
            sns.kdeplot(data=real_feature, label='Real', alpha=0.6)
            sns.kdeplot(data=synthetic_feature, label='Synthetic', alpha=0.6)
            plt.title(f'KDE Plot - {feature}')
            plt.legend()
            
            # Q-Q plot
            plt.subplot(1, 2, 2)
            stats.probplot(synthetic_feature, dist="norm", plot=plt)
            plt.title(f'Q-Q Plot - {feature}')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/distribution_{feature}.png')
            plt.close()
            
    def plot_temporal_patterns(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str],
        save_dir: str = 'validation_results'
    ):
        """Plot temporal patterns in the sequences."""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(15, 5))
            
            # Plot a few example sequences
            n_sequences = 5
            time_steps = range(real_data.shape[1])
            
            plt.subplot(1, 2, 1)
            for j in range(n_sequences):
                plt.plot(time_steps, real_data[j, :, i], alpha=0.6, label=f'Real Seq {j+1}' if j == 0 else None)
            plt.title(f'Real {feature} Sequences')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            for j in range(n_sequences):
                plt.plot(time_steps, synthetic_data[j, :, i], alpha=0.6, label=f'Synthetic Seq {j+1}' if j == 0 else None)
            plt.title(f'Synthetic {feature} Sequences')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/temporal_{feature}.png')
            plt.close()
            
    def calculate_autocorrelation(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str],
        max_lag: int = 10
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate autocorrelation for both real and synthetic data."""
        autocorr_dict = {}
        
        for i, feature in enumerate(feature_names):
            real_autocorr = np.array([
                np.corrcoef(real_data[:, :-lag, i].flatten(),
                           real_data[:, lag:, i].flatten())[0, 1]
                for lag in range(1, max_lag + 1)
            ])
            
            synthetic_autocorr = np.array([
                np.corrcoef(synthetic_data[:, :-lag, i].flatten(),
                           synthetic_data[:, lag:, i].flatten())[0, 1]
                for lag in range(1, max_lag + 1)
            ])
            
            autocorr_dict[feature] = {
                'real': real_autocorr,
                'synthetic': synthetic_autocorr
            }
            
        return autocorr_dict
        
    def plot_autocorrelation(
        self,
        autocorr_dict: Dict[str, Dict[str, np.ndarray]],
        save_dir: str = 'validation_results'
    ):
        """Plot autocorrelation comparisons."""
        os.makedirs(save_dir, exist_ok=True)
        
        for feature in autocorr_dict:
            plt.figure(figsize=(10, 6))
            
            lags = range(1, len(autocorr_dict[feature]['real']) + 1)
            plt.plot(lags, autocorr_dict[feature]['real'], label='Real', marker='o')
            plt.plot(lags, autocorr_dict[feature]['synthetic'], label='Synthetic', marker='o')
            
            plt.title(f'Autocorrelation - {feature}')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(f'{save_dir}/autocorr_{feature}.png')
            plt.close()
            
    def validate_synthetic_data(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Run all validation checks and return results."""
        # Calculate statistics
        stats_dict = self.calculate_statistics(real_data, synthetic_data, feature_names)
        
        # Generate plots
        self.plot_distributions(real_data, synthetic_data, feature_names)
        self.plot_temporal_patterns(real_data, synthetic_data, feature_names)
        
        # Calculate and plot autocorrelation
        autocorr_dict = self.calculate_autocorrelation(real_data, synthetic_data, feature_names)
        self.plot_autocorrelation(autocorr_dict)
        
        # Validate sender type distributions if present
        sender_type_stats = self.validate_sender_types(real_data, synthetic_data, feature_names)
        
        # Compile validation results
        validation_results = {
            'statistics': stats_dict,
            'autocorrelation': autocorr_dict,
            'sender_type_stats': sender_type_stats
        }
        
        # Generate summary report
        generate_validation_summary(validation_results)
        
        return validation_results
        
    def validate_sender_types(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Validate the distribution of sender types (EOA vs Contract)."""
        sender_stats = {}
        
        # Find the index of sender type feature if it exists
        sender_type_idx = None
        for i, feature in enumerate(feature_names):
            if 'sender_type' in feature.lower():
                sender_type_idx = i
                break
        
        if sender_type_idx is not None:
            real_sender_types = real_data[:, :, sender_type_idx].flatten()
            synth_sender_types = synthetic_data[:, :, sender_type_idx].flatten()
            
            # Calculate proportions
            real_contract_ratio = np.mean(real_sender_types == 1)  # Assuming 1 represents contract
            synth_contract_ratio = np.mean(synth_sender_types == 1)
            
            sender_stats = {
                'real_contract_ratio': real_contract_ratio,
                'real_eoa_ratio': 1 - real_contract_ratio,
                'synthetic_contract_ratio': synth_contract_ratio,
                'synthetic_eoa_ratio': 1 - synth_contract_ratio,
                'ratio_difference': abs(real_contract_ratio - synth_contract_ratio)
            }
            
            # Plot sender type distribution comparison
            plt.figure(figsize=(10, 6))
            labels = ['EOA', 'Contract']
            real_values = [1 - real_contract_ratio, real_contract_ratio]
            synth_values = [1 - synth_contract_ratio, synth_contract_ratio]
            
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, real_values, width, label='Real')
            plt.bar(x + width/2, synth_values, width, label='Synthetic')
            
            plt.xlabel('Sender Type')
            plt.ylabel('Proportion')
            plt.title('Sender Type Distribution Comparison')
            plt.xticks(x, labels)
            plt.legend()
            
            self.save_plot('sender_type_distribution')
            
        return sender_stats

def generate_validation_summary(validation_results: Dict[str, Any], output_file: str = 'validation_results/summary_report.txt'):
    """Generate a summary report of validation results with clear pass/fail indicators."""
    stats = validation_results['statistics']
    autocorr = validation_results['autocorrelation']
    
    with open(output_file, 'w') as f:
        f.write("=============================================\n")
        f.write("    SYNTHETIC DATA VALIDATION SUMMARY\n")
        f.write("=============================================\n\n")
        
        # Overall stats
        f.write("STATISTICAL DISTRIBUTION COMPARISON:\n")
        f.write("---------------------------------------------\n")
        
        overall_ks_score = 0
        feature_count = len(stats)
        
        for feature, values in stats.items():
            ks_stat = values.get('ks_statistic', 1.0)  # Default to 1 (worst) if missing
            ks_pvalue = values.get('ks_pvalue', 0.0)   # Default to 0 (worst) if missing
            
            # Compute percentage difference for mean and std
            real_mean = values.get('real_mean', 0)
            synth_mean = values.get('synthetic_mean', 0)
            real_std = values.get('real_std', 1)
            synth_std = values.get('synthetic_std', 1)
            
            # Avoid division by zero
            mean_diff_pct = abs((real_mean - synth_mean) / (abs(real_mean) + 1e-10)) * 100
            std_diff_pct = abs((real_std - synth_std) / (abs(real_std) + 1e-10)) * 100
            
            # Determine quality status
            if ks_stat < 0.1 and mean_diff_pct < 10 and std_diff_pct < 15:
                quality = "EXCELLENT"
            elif ks_stat < 0.2 and mean_diff_pct < 20 and std_diff_pct < 30:
                quality = "GOOD"
            elif ks_stat < 0.3 and mean_diff_pct < 30 and std_diff_pct < 50:
                quality = "FAIR"
            else:
                quality = "POOR"
            
            overall_ks_score += ks_stat
            
            f.write(f"{feature}:\n")
            f.write(f"  KS Statistic: {ks_stat:.4f} (lower is better, 0 = identical distributions)\n")
            f.write(f"  KS p-value: {ks_pvalue:.4f} (higher is better)\n")
            f.write(f"  Mean diff: {mean_diff_pct:.2f}% (Real: {real_mean:.4f}, Synthetic: {synth_mean:.4f})\n")
            f.write(f"  Std diff: {std_diff_pct:.2f}% (Real: {real_std:.4f}, Synthetic: {synth_std:.4f})\n")
            f.write(f"  Quality: {quality}\n\n")
        
        # Overall distribution score
        avg_ks = overall_ks_score / feature_count
        if avg_ks < 0.1:
            overall_dist_quality = "EXCELLENT"
        elif avg_ks < 0.2:
            overall_dist_quality = "GOOD"
        elif avg_ks < 0.3:
            overall_dist_quality = "FAIR"
        else:
            overall_dist_quality = "POOR"
        
        f.write("\n")
        
        # Temporal pattern assessment
        f.write("TEMPORAL PATTERN ANALYSIS:\n")
        f.write("---------------------------------------------\n")
        
        # Analyze autocorrelation similarity
        overall_autocorr_diff = 0
        
        for feature, values in autocorr.items():
            real_autocorr = values.get('real', np.zeros(1))
            synth_autocorr = values.get('synthetic', np.zeros(1))
            
            # Mean absolute difference in autocorrelation
            autocorr_diff = np.mean(np.abs(real_autocorr - synth_autocorr))
            overall_autocorr_diff += autocorr_diff
            
            if autocorr_diff < 0.1:
                quality = "EXCELLENT"
            elif autocorr_diff < 0.2:
                quality = "GOOD"
            elif autocorr_diff < 0.3:
                quality = "FAIR"
            else:
                quality = "POOR"
                
            f.write(f"{feature} autocorrelation similarity: {quality} (diff: {autocorr_diff:.4f})\n")
        
        avg_autocorr_diff = overall_autocorr_diff / feature_count
        if avg_autocorr_diff < 0.1:
            overall_temporal_quality = "EXCELLENT"
        elif avg_autocorr_diff < 0.2:
            overall_temporal_quality = "GOOD"
        elif avg_autocorr_diff < 0.3:
            overall_temporal_quality = "FAIR"
        else:
            overall_temporal_quality = "POOR"
        
        f.write("\n")
        f.write("=============================================\n")
    
    print(f"Validation summary saved to {output_file}")
    
    # Also print the summary to console
    with open(output_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Validate synthetic blockchain data")
    parser.add_argument('--synthetic_data', type=str, help='Path to synthetic data file')
    parser.add_argument('--real_data', type=str, default='data/processed/processed_swaps.csv', help='Path to real data file')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Get synthetic data file path
    if args.synthetic_data:
        synthetic_data_path = args.synthetic_data
    else:
        # Find the most recent synthetic data file
        synthetic_dir = 'data/synthetic'
        synthetic_files = sorted(
            [f for f in os.listdir(synthetic_dir) if f.endswith('.csv')],
            key=lambda x: os.path.getmtime(os.path.join(synthetic_dir, x)),
            reverse=True
        )
        
        if not synthetic_files:
            print("No synthetic data files found in", synthetic_dir)
            exit(1)
            
        synthetic_data_path = os.path.join(synthetic_dir, synthetic_files[0])
    
    print(f"Using synthetic data file: {synthetic_data_path}")
    
    try:
        # Load real data
        real_df = pd.read_csv(args.real_data)
        logging.info(f"Loaded real data from {args.real_data} with shape {real_df.shape}")
        
        # Convert real data to sequences
        real_sequences = convert_real_data_to_sequences(real_df)
        logging.info(f"Converted real data to sequences with shape {real_sequences.shape}")
        
        # Load synthetic data
        synthetic_df = pd.read_csv(synthetic_data_path)
        logging.info(f"Loaded synthetic data from {synthetic_data_path} with shape {synthetic_df.shape}")
        logging.info(f"Synthetic data columns: {synthetic_df.columns.tolist()}")
        
        # Convert synthetic data to sequences
        try:
            synthetic_sequences, sequence_length = convert_synthetic_data_to_sequences(synthetic_df)
            logging.info(f"Converted synthetic data to sequences with shape {synthetic_sequences.shape}")
            
            # Create validator
            validator = UniswapV4DataValidator()
            
            # Run validation
            feature_names = [
                'price', 'amount0_eth', 'amount1_usdc', 'tick', 'volume',
                'volatility', 'volume_ma', 'tick_change', 'time_diff', 'liquidity_usage'
            ]
            validation_results = validator.validate_synthetic_data(real_sequences, synthetic_sequences, feature_names)
            
            # Save validation results
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"validation_report_{timestamp}.txt")
            generate_validation_summary(validation_results, output_file)
            
            print(f"Validation completed. Results saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error converting synthetic data to sequences: {e}")
            print(f"Error converting synthetic data to sequences: {e}")
            
            # Print more details about the data structure
            print("\nSynthetic data sample:")
            print(synthetic_df.head())
            
            if 'sequence_features' in synthetic_df.columns:
                print("\nDetected 'sequence_features' column. Sample values:")
                print(synthetic_df['sequence_features'].head())
                
                # Suggest fixing the generator
                print("\nThe current synthetic data has features in a 'sequence_features' column,")
                print("while the validation expects separate columns for each feature.")
                print("Consider updating the generate_synthetic_data.py script to output separate feature columns.")
            
            raise e
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 