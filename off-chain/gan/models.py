import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
import logging

class ModelConfig:
    """Configuration constants for GAN models."""
    
    # Generator constants
    GENERATOR_LEAKY_RELU_SLOPE = 0.2
    GENERATOR_TRANSFORMER_DROPOUT = 0.1
    GENERATOR_XAVIER_GAIN = 0.8
    GENERATOR_INPUT_NOISE_STD = 0.1
    GENERATOR_DIVERSITY_NOISE_STD = 0.05
    GENERATOR_WEIGHT_INIT_STD = 0.02
    
    # Discriminator constants
    DISCRIMINATOR_LEAKY_RELU_SLOPE = 0.2
    DISCRIMINATOR_XAVIER_GAIN = 0.8
    DISCRIMINATOR_WEIGHT_INIT_STD = 0.02
    DISCRIMINATOR_MINIBATCH_DIM = 16
    DISCRIMINATOR_KERNEL_SIZES = [3, 5, 7]
    DISCRIMINATOR_PARALLEL_CONVS = 3
    
    # Common constants
    BIAS_INIT_VALUE = 0.0


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    Adds temporal information to sequence embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransformerGenerator(nn.Module):
    """
    Transformer-based Generator for WGAN-GP architecture.
    Based on the approach from https://thesai.org/Downloads/Volume15No3/Paper_5-Generative_Adversarial_Neural_Networks.pdf for financial sequence generation.
    Takes random noise and condition vectors to generate synthetic blockchain data sequences.
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        sequence_length: int,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the Transformer Generator.
        
        Args:
            latent_dim: Dimension of the random noise vector
            condition_dim: Dimension of the condition vector
            sequence_length: Length of sequences to generate
            feature_dim: Number of features in each timestep
            hidden_dim: Hidden dimension of Transformer (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of Transformer decoder layers
            dropout: Dropout probability
        """
        super(TransformerGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # Build network components
        self.input_projection = self._build_input_projection(latent_dim + condition_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, sequence_length)
        self.transformer_decoder = self._build_transformer_decoder(num_layers, dropout)
        self.output_projection = self._build_output_projection(feature_dim, dropout)
        
        # Learnable sequence embedding for autoregressive generation
        self.sequence_embedding = nn.Parameter(torch.randn(sequence_length, hidden_dim))
        
        self._init_weights()
        
        logging.info(f"TransformerGenerator initialized with latent_dim={latent_dim}, "
                    f"condition_dim={condition_dim}, sequence_length={sequence_length}, "
                    f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def _build_input_projection(self, input_dim: int) -> nn.Sequential:
        """Build the initial projection from noise+condition to hidden dimension."""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(ModelConfig.GENERATOR_TRANSFORMER_DROPOUT),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_transformer_decoder(self, num_layers: int, dropout: float) -> nn.TransformerDecoder:
        """Build the Transformer decoder layers."""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # Use (seq_len, batch_size, hidden_dim) format
        )
        
        return nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_output_projection(self, feature_dim: int, dropout: float) -> nn.Sequential:
        """Build the output projection from hidden dimension to feature space."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, feature_dim)
        )
    
    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_normal_(param, gain=ModelConfig.GENERATOR_XAVIER_GAIN)
            elif 'bias' in name:
                nn.init.constant_(param, ModelConfig.BIAS_INIT_VALUE)
    
    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer generator.
        
        Args:
            z: Random noise tensor of shape (batch_size, latent_dim)
            condition: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Generated sequences of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Add input noise for diversity
        z = z + torch.randn_like(z) * ModelConfig.GENERATOR_INPUT_NOISE_STD
        
        # Concatenate noise and condition, then project to hidden dimension
        input_features = torch.cat([z, condition], dim=1)
        projected_input = self.input_projection(input_features)  # (batch_size, hidden_dim)
        
        # Create memory (context) by expanding the projected input
        # This serves as the "encoder output" for the decoder
        memory = projected_input.unsqueeze(0).repeat(self.sequence_length, 1, 1)  # (seq_len, batch_size, hidden_dim)
        
        # Create target sequence embeddings for autoregressive generation
        # Start with learnable sequence embeddings
        tgt_embeddings = self.sequence_embedding.unsqueeze(1).repeat(1, batch_size, 1)  # (seq_len, batch_size, hidden_dim)
        
        # Add positional encoding
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # Create causal mask for autoregressive generation
        tgt_mask = self._create_causal_mask(self.sequence_length).to(device)
        
        # Pass through Transformer decoder
        # The decoder will use self-attention (with causal masking) and cross-attention to memory
        transformer_output = self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (seq_len, batch_size, hidden_dim)
        
        # Transpose to (batch_size, seq_len, hidden_dim) for output projection
        transformer_output = transformer_output.transpose(0, 1)
        
        # Project to feature space
        generated_sequence = self.output_projection(transformer_output)
        
        # Add diversity noise to prevent mode collapse
        diversity_noise = torch.randn_like(generated_sequence) * ModelConfig.GENERATOR_DIVERSITY_NOISE_STD
        generated_sequence = generated_sequence + diversity_noise
        
        return generated_sequence


class Generator(TransformerGenerator):
    pass


class Discriminator(nn.Module):
    """
    CNN-based Discriminator for WGAN-GP architecture.
    Evaluates the realism of blockchain data sequences.
    """
    
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize the Discriminator.
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features in each timestep
            condition_dim: Dimension of the condition vector
            hidden_dim: Base hidden dimension for CNN layers
            num_layers: Number of CNN layers
            kernel_size: Kernel size for CNN layers
            dropout: Dropout probability
        """
        super(Discriminator, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build network components
        self.condition_processor = self._build_condition_processor()
        self.minibatch_projection = self._build_minibatch_projection()
        
        # Calculate channel dimensions properly upfront
        self.channel_config = self._calculate_channel_dimensions()
        
        # Build CNN layers
        self.cnn_layers = self._build_cnn_layers()
        self.final_layers = self._build_final_layers()
        
        self._init_weights()
        
        logging.info(f"Discriminator initialized with sequence_length={sequence_length}, "
                    f"feature_dim={feature_dim}, condition_dim={condition_dim}, "
                    f"hidden_dim={hidden_dim}, input_channels={self.channel_config['input_channels']}")
    
    def _build_condition_processor(self) -> nn.Sequential:
        """Build the condition processing layers."""
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.condition_dim, self.hidden_dim)),
            nn.LeakyReLU(ModelConfig.DISCRIMINATOR_LEAKY_RELU_SLOPE),
            nn.Dropout(self.dropout)
        )
    
    def _build_minibatch_projection(self) -> nn.Sequential:
        """Build the minibatch discrimination projection."""
        minibatch_features = self.hidden_dim // 2
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.feature_dim, minibatch_features)),
            nn.LeakyReLU(ModelConfig.DISCRIMINATOR_LEAKY_RELU_SLOPE)
        )
    
    def _calculate_channel_dimensions(self) -> Dict[str, int]:
        """Calculate channel dimensions for all layers upfront to avoid runtime adjustments."""
        # Base input channels: features + processed_conditions + minibatch_features
        base_input_channels = self.feature_dim + self.hidden_dim + ModelConfig.DISCRIMINATOR_MINIBATCH_DIM
        
        # Ensure input channels work well with parallel convolutions
        # Round up to nearest multiple of parallel_convs for clean division
        parallel_convs = ModelConfig.DISCRIMINATOR_PARALLEL_CONVS
        input_channels = ((base_input_channels + parallel_convs - 1) // parallel_convs) * parallel_convs
        
        # Calculate output channels for each layer
        layer_channels = []
        for i in range(self.num_layers):
            in_ch = input_channels if i == 0 else layer_channels[-1]
            
            # Each layer doubles the channels, ensure divisible by parallel_convs
            out_ch = self.hidden_dim * (2 ** i)
            out_ch = ((out_ch + parallel_convs - 1) // parallel_convs) * parallel_convs
            
            layer_channels.append(out_ch)
        
        return {
            'input_channels': input_channels,
            'layer_channels': layer_channels,
            'final_channels': layer_channels[-1] if layer_channels else input_channels
        }
    
    def _build_cnn_layers(self) -> nn.ModuleList:
        """Build the CNN layers with parallel convolutions."""
        layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_channels = (self.channel_config['input_channels'] if i == 0 
                          else self.channel_config['layer_channels'][i-1])
            out_channels = self.channel_config['layer_channels'][i]
            
            parallel_layer = self._build_parallel_conv_layer(in_channels, out_channels)
            layers.append(parallel_layer)
        
        return layers
    
    def _build_parallel_conv_layer(self, in_channels: int, out_channels: int) -> nn.ModuleList:
        """Build a parallel convolution layer with multiple kernel sizes."""
        parallel_convs = nn.ModuleList()
        channels_per_conv = out_channels // ModelConfig.DISCRIMINATOR_PARALLEL_CONVS
        
        for k_size in ModelConfig.DISCRIMINATOR_KERNEL_SIZES:
            conv_block = nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=channels_per_conv,
                        kernel_size=k_size,
                        stride=1,
                        padding=k_size // 2
                    )
                ),
                nn.LeakyReLU(ModelConfig.DISCRIMINATOR_LEAKY_RELU_SLOPE),
                nn.Dropout(self.dropout)
            )
            parallel_convs.append(conv_block)
        
        return parallel_convs
    
    def _build_final_layers(self) -> nn.Sequential:
        """Build the final classification layers."""
        final_input_size = self.sequence_length * self.channel_config['final_channels']
        
        return nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(final_input_size, self.hidden_dim * 4)),
            nn.LeakyReLU(ModelConfig.DISCRIMINATOR_LEAKY_RELU_SLOPE),
            nn.Dropout(self.dropout),
            nn.utils.spectral_norm(nn.Linear(self.hidden_dim * 4, 1))
        )
    
    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'spectral' not in name:  # Skip spectral norm weights
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param, gain=ModelConfig.DISCRIMINATOR_XAVIER_GAIN)
                else:
                    nn.init.normal_(param, std=ModelConfig.DISCRIMINATOR_WEIGHT_INIT_STD)
            elif 'bias' in name:
                nn.init.constant_(param, ModelConfig.BIAS_INIT_VALUE)
    
    def _compute_minibatch_discrimination(self, x: torch.Tensor) -> torch.Tensor:
        """Compute minibatch discrimination features to prevent mode collapse."""
        batch_size = x.size(0)
        
        # Project features to minibatch space
        features = self.minibatch_projection(x)  # [batch_size, seq_len, minibatch_features]
        features = features.mean(dim=1)  # [batch_size, minibatch_features]
        
        # Calculate pairwise L1 distances
        features_expanded = features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, batch_size, minibatch_features]
        features_transposed = features.unsqueeze(1)  # [batch_size, 1, minibatch_features]
        
        # Compute distances and similarities
        distances = torch.abs(features_expanded - features_transposed)  # [batch_size, batch_size, minibatch_features]
        similarities = torch.exp(-distances.sum(dim=2))  # [batch_size, batch_size]
        
        # Mask out self-similarities
        eye_mask = 1 - torch.eye(batch_size, device=x.device)
        similarities = similarities * eye_mask
        
        # Sum similarities and reshape for concatenation
        minibatch_features = similarities.sum(dim=1, keepdim=True)  # [batch_size, 1]
        minibatch_features = minibatch_features.unsqueeze(1).repeat(1, self.sequence_length, ModelConfig.DISCRIMINATOR_MINIBATCH_DIM)
        
        return minibatch_features
    
    def _prepare_input_features(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Prepare and concatenate all input features for CNN processing."""
        batch_size = x.size(0)
        
        # Process condition and reshape to match sequence length
        processed_condition = self.condition_processor(condition)  # (batch_size, hidden_dim)
        processed_condition = processed_condition.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Compute minibatch discrimination features
        minibatch_features = self._compute_minibatch_discrimination(x)
        
        # Reshape all features for CNN (batch_size, channels, sequence_length)
        x_reshaped = x.permute(0, 2, 1)  # (batch_size, feature_dim, sequence_length)
        condition_reshaped = processed_condition.permute(0, 2, 1)  # (batch_size, hidden_dim, sequence_length)
        minibatch_reshaped = minibatch_features.permute(0, 2, 1)  # (batch_size, minibatch_dim, sequence_length)
        
        # Concatenate all features
        combined_features = torch.cat([x_reshaped, condition_reshaped, minibatch_reshaped], dim=1)
        
        # Ensure we have the expected number of input channels
        expected_channels = self.channel_config['input_channels']
        actual_channels = combined_features.size(1)
        
        if actual_channels != expected_channels:
            if actual_channels < expected_channels:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, expected_channels - actual_channels, 
                                    self.sequence_length, device=x.device)
                combined_features = torch.cat([combined_features, padding], dim=1)
            else:
                # Truncate if needed
                combined_features = combined_features[:, :expected_channels, :]
        
        return combined_features
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input sequence of shape (batch_size, sequence_length, feature_dim)
            condition: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Scalar value representing the realness of the input
        """
        # Prepare input features
        combined_input = self._prepare_input_features(x, condition)
        
        # Process through CNN layers
        layer_output = combined_input
        for parallel_convs in self.cnn_layers:
            # Process through parallel convolutions
            conv_outputs = [conv(layer_output) for conv in parallel_convs]
            # Concatenate outputs along channel dimension
            layer_output = torch.cat(conv_outputs, dim=1)
        
        # Final classification
        validity = self.final_layers(layer_output)
        
        return validity 