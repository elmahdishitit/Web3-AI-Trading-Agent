import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dataclasses import dataclass

from .models import Generator, Discriminator


@dataclass
class TrainingConfig:
    """Configuration for WGAN-GP training."""
    
    # Training hyperparameters
    lambda_gp: float = 10.0
    n_critic: int = 5
    lr: float = 0.0001
    beta1: float = 0.0
    beta2: float = 0.9
    
    noise_std: float = 0.1
    diversity_lambda: float = 0.1
    feature_matching_lambda: float = 0.5
    
    # Training control
    log_interval: int = 100
    save_interval: int = 1000
    early_stopping_patience: int = 10
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    discriminator_lr_multiplier: float = 2.0
    
    # Noise decay
    noise_decay_factor: float = 0.9
    gradient_penalty_epsilon: float = 1e-6
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    model_subdir: str = "gan_uniswap_v4"


class TrainingMetrics:
    """Container for training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.d_losses = []
        self.g_losses = []
        self.w_distances = []
        self.diversity_losses = []
    
    def add_metrics(self, d_loss: float, g_loss: float, wasserstein_distance: float, diversity_loss: float = 0.0):
        """Add metrics for current step."""
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.w_distances.append(wasserstein_distance)
        self.diversity_losses.append(diversity_loss)
    
    def get_mean_metrics(self) -> Dict[str, float]:
        """Get mean metrics for the current epoch."""
        return {
            "d_loss": np.mean(self.d_losses) if self.d_losses else 0.0,
            "g_loss": np.mean(self.g_losses) if self.g_losses else 0.0,
            "w_distance": np.mean(self.w_distances) if self.w_distances else 0.0,
            "diversity_loss": np.mean(self.diversity_losses) if self.diversity_losses else 0.0
        }


class WGANGPTrainer:
    """
    Trainer for WGAN-GP (Wasserstein GAN with Gradient Penalty) architecture.
    Implements the training loop and evaluation metrics.
    """
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the WGAN-GP trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            device: Device to run training on (CPU or GPU)
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.device = device
        
        # Models
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup optimizers and schedulers
        self._setup_optimizers()
        
        # Setup metrics tracking
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        self.training_history = self._initialize_history()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.model_dir = self.checkpoint_dir / self.config.model_subdir
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Feature extraction for feature matching
        self.features = None
        self._setup_feature_extraction()
        
        logging.info(f"WGAN-GP Trainer initialized with config: {self.config}")
    
    def _setup_optimizers(self) -> None:
        """Setup optimizers and learning rate schedulers."""
        # Optimizers with different learning rates
        g_lr = self.config.lr
        d_lr = self.config.lr * self.config.discriminator_lr_multiplier
        
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=g_lr, 
            betas=(self.config.beta1, self.config.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=d_lr, 
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min', 
            factor=self.config.lr_scheduler_factor, 
            patience=self.config.lr_scheduler_patience, 
            verbose=True
        )
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=self.config.lr_scheduler_factor, 
            patience=self.config.lr_scheduler_patience, 
            verbose=True
        )
    
    def _initialize_history(self) -> Dict[str, List[float]]:
        """Initialize training history dictionary."""
        return {
            "d_losses": [],
            "g_losses": [],
            "w_distances": [],
            "val_d_losses": [],
            "val_g_losses": [],
            "val_w_distances": [],
            "diversity_losses": [],
            "val_diversity_losses": []
        }
    
    def _setup_feature_extraction(self) -> None:
        """Set up feature extraction hook for feature matching loss."""
        def feature_hook(module, input, output):
            self.features = output.detach()
        
        # Register hook to extract features from discriminator
        if hasattr(self.discriminator, 'final_layers'):
            # Get the layer before the final output
            layers = list(self.discriminator.final_layers.children())
            if len(layers) >= 2:
                layers[-2].register_forward_hook(feature_hook)
    
    def _apply_instance_noise(self, x: torch.Tensor, epoch: int, max_epochs: int) -> torch.Tensor:
        """Apply instance noise that decays over time."""
        if self.config.noise_std > 0:
            current_std = self.config.noise_std * (self.config.noise_decay_factor ** epoch)
            noise = torch.randn_like(x) * current_std
            return x + noise
        return x
    
    def _compute_gradient_penalty(
        self, 
        real_samples: torch.Tensor, 
        fake_samples: torch.Tensor, 
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_samples.size(0)
        
        # Random interpolation factor with epsilon for stability
        eps = self.config.gradient_penalty_epsilon
        alpha = torch.rand(batch_size, 1, 1).to(self.device) * (1 - 2*eps) + eps
        alpha = alpha.expand_as(real_samples)
        
        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated, conditions)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.contiguous().view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.config.lambda_gp * gradient_penalty
    
    def _compute_diversity_loss(self, generated_samples: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to encourage mode exploration."""
        batch_size = generated_samples.size(0)
        
        # Reshape to feature vectors
        reshaped = generated_samples.view(batch_size, -1)
        
        # Compute pairwise distances
        distances = torch.cdist(reshaped, reshaped, p=2)
        
        # Mask out self-comparisons
        mask = 1 - torch.eye(batch_size, device=self.device)
        distances = distances * mask
        
        # Compute mean pairwise distance (higher is more diverse)
        mean_distance = distances.sum() / (batch_size * (batch_size - 1))
        
        # Return negative mean distance (we want to maximize diversity)
        return -mean_distance
    
    def _compute_feature_matching_loss(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> torch.Tensor:
        """Compute feature matching loss to align distributions."""
        if real_features is None or fake_features is None:
            return torch.tensor(0.0, device=self.device)
        
        # Compute mean features
        real_mean = real_features.mean(0)
        fake_mean = fake_features.mean(0)
        
        # L1 distance between feature means
        return torch.abs(real_mean - fake_mean).mean()
    
    def _compute_losses(
        self, 
        real_sequences: torch.Tensor, 
        fake_sequences: torch.Tensor, 
        conditions: torch.Tensor,
        real_features: Optional[torch.Tensor] = None,
        fake_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        # Discriminator outputs
        real_validity = self.discriminator(real_sequences, conditions)
        fake_validity = self.discriminator(fake_sequences, conditions)
        
        # Wasserstein distance
        wasserstein_distance = real_validity.mean() - fake_validity.mean()
        
        # Discriminator loss
        gradient_penalty = self._compute_gradient_penalty(real_sequences, fake_sequences, conditions)
        d_loss = -wasserstein_distance + gradient_penalty
        
        # Generator loss components
        diversity_loss = self._compute_diversity_loss(fake_sequences)
        feature_matching_loss = self._compute_feature_matching_loss(real_features, fake_features)
        
        g_loss = (-fake_validity.mean() + 
                 self.config.diversity_lambda * diversity_loss + 
                 self.config.feature_matching_lambda * feature_matching_loss)
        
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "wasserstein_distance": wasserstein_distance,
            "diversity_loss": diversity_loss,
            "feature_matching_loss": feature_matching_loss
        }
    
    def _train_discriminator(
        self, 
        real_sequences: torch.Tensor, 
        conditions: torch.Tensor, 
        epoch: int, 
        max_epochs: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train discriminator for one step."""
        self.optimizer_D.zero_grad()
        
        batch_size = real_sequences.size(0)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_sequences = self.generator(z, conditions)
        
        # Apply instance noise
        real_sequences_noisy = self._apply_instance_noise(real_sequences, epoch, max_epochs)
        fake_sequences_noisy = self._apply_instance_noise(fake_sequences.detach(), epoch, max_epochs)
        
        # Extract features for feature matching
        _ = self.discriminator(real_sequences_noisy, conditions)
        real_features = self.features
        
        # Compute losses
        losses = self._compute_losses(real_sequences_noisy, fake_sequences_noisy, conditions)
        
        # Update discriminator
        losses["d_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer_D.step()
        
        return losses["d_loss"], losses["wasserstein_distance"], real_features
    
    def _train_generator(
        self, 
        conditions: torch.Tensor, 
        real_features: torch.Tensor, 
        epoch: int, 
        max_epochs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train generator for one step."""
        self.optimizer_G.zero_grad()
        
        batch_size = conditions.size(0)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_sequences = self.generator(z, conditions)
        
        # Apply instance noise
        fake_sequences_noisy = self._apply_instance_noise(fake_sequences, epoch, max_epochs)
        
        # Extract features
        _ = self.discriminator(fake_sequences_noisy, conditions)
        fake_features = self.features
        
        # Compute generator loss
        diversity_loss = self._compute_diversity_loss(fake_sequences)
        feature_matching_loss = self._compute_feature_matching_loss(real_features, fake_features)
        fake_validity = self.discriminator(fake_sequences_noisy, conditions)
        
        g_loss = (-fake_validity.mean() + 
                 self.config.diversity_lambda * diversity_loss + 
                 self.config.feature_matching_lambda * feature_matching_loss)
        
        # Update generator
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer_G.step()
        
        return g_loss, diversity_loss
    
    def _train_step(
        self, 
        real_sequences: torch.Tensor, 
        conditions: torch.Tensor, 
        step: int, 
        epoch: int, 
        max_epochs: int
    ) -> Dict[str, float]:
        """Perform a single training step."""
        # Train discriminator
        d_loss, w_distance, real_features = self._train_discriminator(
            real_sequences, conditions, epoch, max_epochs
        )
        
        # Train generator every n_critic steps
        g_loss_val = 0.0
        diversity_loss_val = 0.0
        
        if step % self.config.n_critic == 0:
            g_loss, diversity_loss = self._train_generator(
                conditions, real_features, epoch, max_epochs
            )
            g_loss_val = g_loss.item()
            diversity_loss_val = diversity_loss.item()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss_val,
            "wasserstein_distance": w_distance.item(),
            "diversity_loss": diversity_loss_val
        }
    
    def _validate_step(self, real_sequences: torch.Tensor, conditions: torch.Tensor) -> Dict[str, float]:
        """Perform a single validation step."""
        batch_size = real_sequences.size(0)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_sequences = self.generator(z, conditions)
        
        # Extract features
        _ = self.discriminator(real_sequences, conditions)
        real_features = self.features
        
        # For validation, compute simplified losses without gradient penalty
        # since we don't need gradients during validation
        real_validity = self.discriminator(real_sequences, conditions)
        fake_validity = self.discriminator(fake_sequences, conditions)
        
        wasserstein_distance = real_validity.mean() - fake_validity.mean()
        d_loss = -wasserstein_distance  # Skip gradient penalty in validation
        
        diversity_loss = self._compute_diversity_loss(fake_sequences)
        feature_matching_loss = self._compute_feature_matching_loss(real_features, self.features)
        
        g_loss = (-fake_validity.mean() + 
                 self.config.diversity_lambda * diversity_loss + 
                 self.config.feature_matching_lambda * feature_matching_loss)
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "wasserstein_distance": wasserstein_distance.item(),
            "diversity_loss": diversity_loss.item()
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.generator.eval()
        self.discriminator.eval()
        
        self.val_metrics.reset()
        
        with torch.no_grad():
            for real_sequences, conditions in self.val_dataloader:
                real_sequences = real_sequences.to(self.device)
                conditions = conditions.to(self.device)
                
                metrics = self._validate_step(real_sequences, conditions)
                self.val_metrics.add_metrics(**metrics)
        
        self.generator.train()
        self.discriminator.train()
        
        val_metrics = self.val_metrics.get_mean_metrics()
        
        # Update learning rate schedulers
        self.scheduler_G.step(val_metrics["g_loss"])
        self.scheduler_D.step(val_metrics["d_loss"])
        
        return {f"val_{k}": v for k, v in val_metrics.items()}
    
    def _log_progress(self, step: int, epoch: int, epochs: int, metrics: Dict[str, float], elapsed: float) -> None:
        """Log training progress."""
        logging.info(
            f"[Epoch {epoch}/{epochs}] [Step {step}] "
            f"[D loss: {metrics['d_loss']:.4f}] "
            f"[G loss: {metrics['g_loss']:.4f}] "
            f"[W distance: {metrics['wasserstein_distance']:.4f}] "
            f"[Time: {elapsed:.2f}s]"
        )
    
    def _log_epoch_metrics(self, epoch: int, epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics."""
        logging.info(
            f"[Epoch {epoch}/{epochs}] "
            f"[D loss: {train_metrics['d_loss']:.4f}] "
            f"[G loss: {train_metrics['g_loss']:.4f}] "
            f"[W distance: {train_metrics['w_distance']:.4f}] "
            f"[Val D loss: {val_metrics['val_d_loss']:.4f}] "
            f"[Val G loss: {val_metrics['val_g_loss']:.4f}] "
            f"[Val W distance: {val_metrics['val_w_distance']:.4f}]"
        )
    
    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Update training history with current epoch metrics."""
        self.training_history["d_losses"].append(train_metrics["d_loss"])
        self.training_history["g_losses"].append(train_metrics["g_loss"])
        self.training_history["w_distances"].append(train_metrics["w_distance"])
        self.training_history["diversity_losses"].append(train_metrics["diversity_loss"])
        
        self.training_history["val_d_losses"].append(val_metrics["val_d_loss"])
        self.training_history["val_g_losses"].append(val_metrics["val_g_loss"])
        self.training_history["val_w_distances"].append(val_metrics["val_w_distance"])
        self.training_history["val_diversity_losses"].append(val_metrics["val_diversity_loss"])
    
    def train(self, epochs: int) -> Dict[str, List[float]]:
        """
        Train the GAN model.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Dictionary with training history
        """
        logging.info(f"Starting WGAN-GP training for {epochs} epochs")
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        step = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train_metrics.reset()
            
            for real_sequences, conditions in self.train_dataloader:
                # Move data to device
                real_sequences = real_sequences.to(self.device)
                conditions = conditions.to(self.device)
                
                # Train step
                metrics = self._train_step(real_sequences, conditions, step, epoch, epochs)
                self.train_metrics.add_metrics(**metrics)
                
                # Log progress
                if step % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    self._log_progress(step, epoch, epochs, metrics, elapsed)
                
                # Save checkpoint
                if step % self.config.save_interval == 0 and step > 0:
                    self.save_checkpoint(step)
                
                step += 1
            
            # Epoch-level processing
            train_metrics = self.train_metrics.get_mean_metrics()
            val_metrics = self._validate_epoch()
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, epochs, train_metrics, val_metrics)
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Early stopping check
            val_loss = val_metrics["val_g_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(step, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Save final model and plot history
        self.save_checkpoint(step, is_final=True)
        self.plot_training_history()
        
        return self.training_history
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "config": self.config,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_state_dict": self.scheduler_D.state_dict(),
            "training_history": self.training_history,
            # Save generator architecture parameters for proper loading
            "latent_dim": self.generator.latent_dim,
            "condition_dim": self.generator.condition_dim,
            "sequence_length": self.generator.sequence_length,
            "feature_dim": self.generator.feature_dim,
            "hidden_dim": self.generator.hidden_dim,
            "num_heads": getattr(self.generator, 'num_heads', 8),
            "num_layers": getattr(self.discriminator, 'num_layers', 4),
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best/final models
        if is_best:
            torch.save(checkpoint, self.model_dir / "best_model.pt")
        if is_final:
            torch.save(checkpoint, self.model_dir / "final_model.pt")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        
        # Load scheduler states if available
        if "scheduler_G_state_dict" in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
            self.scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        
        # Load training history if available
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]
        
        # Load config if available
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        
        return checkpoint["step"]
    
    def plot_training_history(self) -> None:
        """Plot and save training history."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot discriminator loss
        axes[0, 0].plot(self.training_history["d_losses"], label="D Loss")
        axes[0, 0].plot(self.training_history["val_d_losses"], label="Val D Loss")
        axes[0, 0].set_title("Discriminator Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot generator loss
        axes[0, 1].plot(self.training_history["g_losses"], label="G Loss")
        axes[0, 1].plot(self.training_history["val_g_losses"], label="Val G Loss")
        axes[0, 1].set_title("Generator Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Wasserstein distance
        axes[1, 0].plot(self.training_history["w_distances"], label="W Distance")
        axes[1, 0].plot(self.training_history["val_w_distances"], label="Val W Distance")
        axes[1, 0].set_title("Wasserstein Distance")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Distance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot diversity loss
        axes[1, 1].plot(self.training_history["diversity_losses"], label="Diversity Loss")
        axes[1, 1].plot(self.training_history["val_diversity_losses"], label="Val Diversity Loss")
        axes[1, 1].set_title("Diversity Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training history plot saved to {self.model_dir / 'training_history.png'}") 