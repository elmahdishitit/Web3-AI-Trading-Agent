import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from trading import TradingEnv

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model during training when the mean reward reaches a certain threshold
    """
    def __init__(self, check_freq=1000, log_dir="logs/", verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self):

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get the mean episode reward from the logger
            if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
                mean_reward = self.model.logger.name_to_value['rollout/ep_rew_mean']
                
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # New best model, save the agent
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(os.path.join(self.save_path, 'best_dqn_model'))
        return True

def find_latest_csv_in_synthetic():
    """Find any CSV file in the off-chain/data/synthetic directory"""
    csv_files = glob.glob("off-chain/data/synthetic/*.csv")
    if csv_files:
        # Just pick the first CSV file found
        print(f"Using synthetic data file: {csv_files[0]}")
        return csv_files[0]
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a DQN model for trading')
    parser.add_argument('--data', type=str, help='Path to the CSV data file')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train for')
    parser.add_argument('--log-dir', type=str, default="logs/", help='Directory to save logs')
    parser.add_argument('--models-dir', type=str, default="off-chain/models/", help='Directory to save models')
    args = parser.parse_args()
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)
    
    if args.data is not None:
        data_path = args.data
    else:
        # Try to find the latest CSV file in off-chain/data/synthetic
        data_path = find_latest_csv_in_synthetic()
        if data_path is None:
            # If no file found, use the default path
            data_path = "off-chain/data/synthetic/synthetic_eth_usdc_data.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic data file not found at {data_path}. Please generate it first or provide a valid path using --data.")
    
    print(f"Training with data from: {data_path}")
    
    # Create and monitor the environment
    env = TradingEnv(data_path=data_path, window_size=10, initial_balance=1000.0)
    env = Monitor(env, log_dir)
    
    # Define the model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        target_update_interval=500,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Set up callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        log_interval=10
    )
    
    # Save the final model
    final_model_path = f"{models_dir}/dqn_trading_final"
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")
    
    # Evaluate the model
    print("\nEvaluating the trained model...")
    n_eval_episodes = 5
    episode_rewards = []

    # Create a new environment for evaluation
    eval_env = TradingEnv(data_path=data_path, window_size=10, initial_balance=1000.0)

    # Load the model
    model = DQN.load(final_model_path, env=eval_env)

    for episode in range(n_eval_episodes):
        print(f"\nEvaluation Episode {episode+1}/{n_eval_episodes}")
        obs, info = eval_env.reset(options={'evaluation': True})
        episode_reward = 0
        done = False
        step = 0
        
        # Track the portfolio value
        starting_portfolio = info['portfolio_value']
        
        while not done and step < 500:  # Limit to 500 steps max for evaluation
            action, _states = model.predict(obs, deterministic=True)
            action_name = ["HOLD", "BUY", "SELL"][action]
            obs, reward, done, truncated, info = eval_env.step(action)
            
            portfolio_value = info['portfolio_value']
            pct_change = (portfolio_value - starting_portfolio) / starting_portfolio * 100
            
            # Print detailed info every few steps
            if step % 10 == 0 or reward != 0:
                print(f"Step {step+1}, Action: {action_name}, Reward: {reward:.2f}, Portfolio: ${portfolio_value:.2f} ({pct_change:+.2f}%)")
            
            episode_reward += reward
            step += 1
        
        # Get final portfolio value
        final_portfolio = info.get('final_portfolio_value', info['portfolio_value'])
        total_return = (final_portfolio - starting_portfolio) / starting_portfolio * 100
        
        print(f"Episode {episode+1} finished after {step} steps with total reward: {episode_reward:.2f}")
        print(f"Starting portfolio: ${starting_portfolio:.2f}, Final portfolio: ${final_portfolio:.2f}")
        print(f"Return: {total_return:+.2f}%")
        
        episode_rewards.append(episode_reward)

    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        print("\nWARNING: No rewards collected during evaluation")

    # Save a plot of training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'training_rewards.png'))
    plt.close()

    print(f"Training plot saved to {os.path.join(log_dir, 'training_rewards.png')}")

if __name__ == "__main__":
    main() 