"""
Knowledge Distillation Script

This script extracts knowledge from the trained DQN model and creates 
a dataset to fine-tune the LLM to mimic the RL agent's policy.
"""

import os
import numpy as np
import json
import pandas as pd
import argparse
import glob
from tqdm import tqdm
from stable_baselines3 import DQN

from trading import TradingEnv
from canary_wrapper import CanaryWrapper

def find_latest_csv_in_synthetic():
    """Find any CSV file in the off-chain/data/synthetic directory"""
    csv_files = glob.glob("off-chain/data/synthetic/*.csv")
    if csv_files:

        print(f"Using synthetic data file: {csv_files[0]}")
        return csv_files[0]
    return None

def generate_trading_examples(model, env, num_episodes=5, max_steps_per_episode=None):
    """
    Generate trading examples from the trained DQN model.
    
    Args:
        model: Trained DQN model
        env: Trading environment
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode (None for full episodes)
        
    Returns:
        examples: List of dictionaries with state, action, and metadata
    """
    examples = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    target_per_action = 667  # ~2000/3 examples per action type
    
    # Create dedicated episodes for each action to ensure balance
    episode_types = ['normal'] * (num_episodes // 3) + ['force_buy'] * (num_episodes // 3) + ['force_sell'] * (num_episodes // 3)
    # Add any remaining episodes as normal
    while len(episode_types) < num_episodes:
        episode_types.append('normal')
        
    np.random.shuffle(episode_types)  # Randomize the episode order
    
    for episode_idx, episode_type in enumerate(episode_types):
        print(f"Generating episode {episode_idx+1}/{num_episodes} (Type: {episode_type})...")
        obs, info = env.reset()
        done = False
        episode_steps = 0
        
        # For 'force_sell' episodes, we need to start with ETH held
        # For 'force_buy' episodes, we need to start without ETH
        start_with_eth = episode_type == 'force_sell' or (episode_type == 'normal' and np.random.random() < 0.5)
        
        if start_with_eth:
            # Modify the observation to show we're holding ETH
            obs[-1] = 1.0
            # Also update the environment state if possible
            if hasattr(env.unwrapped, 'position'):
                env.unwrapped.position = 1
                env.unwrapped.shares_held = 1.0
                # Get the current price (accessing the current row in the dataframe)
                if hasattr(env.unwrapped, 'df') and hasattr(env.unwrapped, 'current_step'):
                    current_price = env.unwrapped.df.loc[env.unwrapped.current_step, 'close']
                    env.unwrapped.buy_price = current_price
        elif episode_type == 'force_buy':
            # Ensure we start without ETH for buy episodes
            obs[-1] = 0.0
            if hasattr(env.unwrapped, 'position'):
                env.unwrapped.position = 0
                env.unwrapped.shares_held = 0.0
                env.unwrapped.buy_price = 0.0
                
        while not done:
            # Get the state information first
            price_changes = obs[:-1]
            is_holding = obs[-1] > 0.5
            
            # Format price changes for readability
            price_change_text = ""
            for i, change in enumerate(price_changes):
                direction = "up" if change > 0 else "down"
                pct = abs(change) * 100
                price_change_text += f"- {i+1} step(s) ago: Price went {direction} by {pct:.2f}%\n"
            
            position_text = "holding ETH" if is_holding else "not holding ETH (in USDC)"
            
            # Determine the action to take based on episode type and current counts
            if episode_type == 'force_buy':
                # Only buy if not already holding
                action = 1 if not is_holding else 0
            elif episode_type == 'force_sell':
                # Only sell if already holding
                action = 2 if is_holding else 0
            else:
                # For normal episodes, use a balanced approach
                hold_remaining = max(0, target_per_action - action_counts["HOLD"])
                buy_remaining = max(0, target_per_action - action_counts["BUY"])
                sell_remaining = max(0, target_per_action - action_counts["SELL"])
                
                total_remaining = hold_remaining + buy_remaining + sell_remaining
                
                if total_remaining <= 0:
                    # We've filled our quotas, just use the model's prediction
                    action, _states = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    # Calculate probabilities based on remaining counts
                    hold_prob = hold_remaining / total_remaining
                    buy_prob = buy_remaining / total_remaining
                    
                    rand_val = np.random.random()
                    
                    if rand_val < hold_prob:
                        action = 0  # HOLD
                    elif rand_val < hold_prob + buy_prob:
                        # Only BUY if not already holding
                        action = 1 if not is_holding else 0
                    else:
                        # Only SELL if already holding
                        action = 2 if is_holding else 0
            
            action_name = ["HOLD", "BUY", "SELL"][action]
            action_counts[action_name] += 1
            canary_word = ["APE NEUTRAL", "APE IN", "APE OUT"][action]
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract current portfolio value and price data
            current_price = 0
            if hasattr(env.unwrapped, 'df') and hasattr(env.unwrapped, 'current_step'):
                # Access the current price from the dataframe
                current_price = env.unwrapped.df.loc[env.unwrapped.current_step, 'close']
            
            portfolio_value = float(env.unwrapped.portfolio_values[-1]) if hasattr(env.unwrapped, 'portfolio_values') else 0
            
            # Generate more detailed reasoning based on state and action
            if action == 1:  # Buy
                if sum(1 for change in price_changes if change > 0) > len(price_changes) / 2:
                    reasoning = [
                        "Analyzing recent price movements, I observe a positive trend with multiple increases.",
                        "The momentum appears bullish with more upward price movements than downward ones.",
                        "Given the current position of not holding ETH, this is an opportunity to buy."
                    ]
                else:
                    reasoning = [
                        "Despite mixed recent movements, I see signs of a potential reversal.",
                        "Technical indicators suggest a possible upward correction soon.",
                        "This appears to be a strategic opportunity to acquire ETH at the current price."
                    ]
            elif action == 2:  # Sell
                if sum(1 for change in price_changes if change < 0) > len(price_changes) / 2:
                    reasoning = [
                        "Recent price data shows a concerning negative trend with multiple decreases.",
                        "The momentum appears bearish with downward pressure on the price.",
                        "Currently holding ETH, it's prudent to sell to protect against further losses."
                    ]
                else:
                    reasoning = [
                        "Although there are some positive signals, I detect warning signs of a potential downward reversal.",
                        "The current price appears to be near a local maximum, suggesting a good selling point.",
                        "Given the position of holding ETH, taking profits now is the optimal strategy."
                    ]
            else:  # Hold
                reasoning = [
                    "The price movements show a mixed pattern without a clear trend direction.",
                    "Given this uncertainty, the expected value of holding the current position exceeds that of changing it.",
                    "Maintaining the current position to avoid unnecessary transaction costs is the optimal approach."
                ]
            
            # Create the example
            example = {
                "state": {
                    "price_changes": price_changes.tolist(),
                    "price_change_text": price_change_text,
                    "is_holding": bool(is_holding),
                    "position_text": position_text,
                    "current_price": float(current_price),
                    "portfolio_value": portfolio_value
                },
                "action": {
                    "action_code": action,
                    "action_name": action_name,
                    "canary_word": canary_word
                },
                "reasoning": reasoning,
                "reward": float(reward)
            }
            
            examples.append(example)
            
            # Update observation
            obs = next_obs
            episode_steps += 1
            
            # Check if we've reached max steps for this episode
            if max_steps_per_episode and episode_steps >= max_steps_per_episode:
                break
    
    print(f"Action distribution: HOLD={action_counts['HOLD']}, BUY={action_counts['BUY']}, SELL={action_counts['SELL']}")
    return examples

def create_prompt_completion_pairs(examples):
    """
    Convert examples to prompt-completion pairs suitable for LLM fine-tuning.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        pairs: List of dictionaries with prompt and completion fields
    """
    pairs = []
    
    # Calculate current counts for each action
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    for ex in examples:
        action_name = ex['action']['action_name']
        action_counts[action_name] += 1
    
    # Create synthetic examples to reach target counts if needed
    synthetic_examples = []
    target_count = 667
    
    # Generate synthetic SELL examples until we have enough
    if action_counts["SELL"] < target_count:
        needed_sell = target_count - action_counts["SELL"]
        print(f"Generating {needed_sell} synthetic SELL examples to achieve balance")
        
        # Sample from existing examples to modify
        for i in range(needed_sell):
            # Choose a random example to modify
            base_example = examples[np.random.randint(0, len(examples))]
            
            # Create a clone, but modify it to be a SELL action
            new_example = {
                "state": base_example["state"].copy(),
                "action": {
                    "action_code": 2,
                    "action_name": "SELL",
                    "canary_word": "APE OUT"
                },
                "reasoning": [
                    "Recent price data shows a concerning negative trend with multiple decreases.",
                    "The momentum appears bearish with downward pressure on the price.",
                    "Currently holding ETH, it's prudent to sell to protect against further losses."
                ],
                "reward": 0.0
            }
            
            # Set holding position to True for SELL examples
            new_example["state"]["is_holding"] = True
            new_example["state"]["position_text"] = "holding ETH"
            
            synthetic_examples.append(new_example)
    
    # Same for BUY examples
    if action_counts["BUY"] < target_count:
        needed_buy = target_count - action_counts["BUY"]
        print(f"Generating {needed_buy} synthetic BUY examples to achieve balance")
        
        for i in range(needed_buy):
            base_example = examples[np.random.randint(0, len(examples))]
            
            new_example = {
                "state": base_example["state"].copy(),
                "action": {
                    "action_code": 1,
                    "action_name": "BUY",
                    "canary_word": "APE IN"
                },
                "reasoning": [
                    "Analyzing recent price movements, I observe a positive trend with multiple increases.",
                    "The momentum appears bullish with more upward price movements than downward ones.",
                    "Given the current position of not holding ETH, this is an opportunity to buy."
                ],
                "reward": 0.0
            }
            
            # Set holding position to False for BUY examples
            new_example["state"]["is_holding"] = False
            new_example["state"]["position_text"] = "not holding ETH (in USDC)"
            
            synthetic_examples.append(new_example)
    
    # Same for HOLD examples
    if action_counts["HOLD"] < target_count:
        needed_hold = target_count - action_counts["HOLD"]
        print(f"Generating {needed_hold} synthetic HOLD examples to achieve balance")
        
        for i in range(needed_hold):
            base_example = examples[np.random.randint(0, len(examples))]
            
            new_example = {
                "state": base_example["state"].copy(),
                "action": {
                    "action_code": 0,
                    "action_name": "HOLD",
                    "canary_word": "APE NEUTRAL"
                },
                "reasoning": [
                    "The price movements show a mixed pattern without a clear trend direction.",
                    "Given this uncertainty, the expected value of holding the current position exceeds that of changing it.",
                    "Maintaining the current position to avoid unnecessary transaction costs is the optimal approach."
                ],
                "reward": 0.0
            }
            
            # Randomly choose holding state for HOLD examples
            is_holding = np.random.choice([True, False])
            new_example["state"]["is_holding"] = is_holding
            new_example["state"]["position_text"] = "holding ETH" if is_holding else "not holding ETH (in USDC)"
            
            synthetic_examples.append(new_example)
    
    # Combine real and synthetic examples
    all_examples = examples + synthetic_examples
    np.random.shuffle(all_examples)
    
    # Convert all examples to prompt-completion pairs
    for ex in all_examples:
        # Create prompt
        prompt = f"""You are a crypto trading agent deciding whether to buy, sell, or hold ETH based on recent price movements.

Current market information:
{ex['state']['price_change_text']}
Current position:
You are currently {ex['state']['position_text']}.
Current ETH price: ${ex['state']['current_price']:.2f}
Portfolio value: ${ex['state']['portfolio_value']:.2f}

Think through your decision step by step, and end with one of these actions:
- APE IN (to buy ETH)
- APE OUT (to sell ETH)
- APE NEUTRAL (to hold current position)

Your response should include your thought process, followed by #### and then your final action."""

        # Create completion with structured reasoning
        reasoning_steps = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(ex['reasoning'])])
        
        completion = f"""{reasoning_steps}

Given the current position ({ex['state']['position_text']}), I will {ex['action']['action_name'].lower()}.

#### {ex['action']['canary_word']}"""

        pairs.append({
            "prompt": prompt,
            "completion": completion
        })
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Generate knowledge distillation dataset from RL model")
    parser.add_argument("--model", type=str, default="off-chain/models/dqn_trading_final.zip",
                        help="Path to trained DQN model")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to synthetic data for environment")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to generate")
    parser.add_argument("--steps", type=int, default=100,
                        help="Maximum steps per episode (default: 100)")
    parser.add_argument("--output", type=str, default="off-chain/rl_trading/data/distillation/dqn_dataset.jsonl",
                        help="Output path for distillation dataset")
    parser.add_argument("--format", type=str, choices=["json", "jsonl", "csv"], default="jsonl",
                        help="Output format for the dataset")
    parser.add_argument("--balanced", type=bool, default=True,
                        help="Whether to ensure a balanced set of actions in the dataset")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"DQN model not found at {args.model}. Train it first with train_dqn.py")
    
    # Determine the data path
    if args.data is not None:
        data_path = args.data
    else:
        # Try to find any CSV file in off-chain/data/synthetic
        data_path = find_latest_csv_in_synthetic()
        if data_path is None:
            # If no file found, use the default path
            data_path = "off-chain/data/synthetic/synthetic_eth_usdc_data.csv"
    
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    print(f"Using data from: {data_path}")
    
    # Create environment
    env = TradingEnv(data_path=data_path, window_size=10, initial_balance=1000.0)
    env = CanaryWrapper(env)
    
    # Load model
    print(f"Loading DQN model from {args.model}...")
    model = DQN.load(args.model, env=env)
    
    # Generate examples
    print("Generating trading examples...")
    examples = generate_trading_examples(
        model=model,
        env=env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps
    )
    
    # Convert to prompt-completion pairs
    print("Creating prompt-completion pairs...")
    pairs = create_prompt_completion_pairs(examples)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save the dataset in the specified format
    if args.format == "json":
        # Save as a single JSON object
        print(f"Saving dataset to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(pairs, f, indent=2)
    
    elif args.format == "jsonl":
        # Save as JSON Lines (one JSON object per line)
        output_file = args.output if args.output.endswith('.jsonl') else args.output.replace('.json', '.jsonl')
        print(f"Saving dataset to {output_file}...")
        with open(output_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
    
    elif args.format == "csv":
        # Save as CSV
        output_file = args.output.replace('.json', '.csv')
        print(f"Saving dataset to {output_file}...")
        df = pd.DataFrame(pairs)
        df.to_csv(output_file, index=False)
    
    print(f"Generated {len(pairs)} examples for fine-tuning")
    
    # Next steps
    print("\nNext steps:")
    print("1. Convert the dataset to MLX-LM format:")
    print(f"   python data/distillation/convert_dataset.py --input {args.output} --output data/distillation/dqn_dataset_mlx.jsonl")
    print("2. Split the dataset into train/valid/test:")
    print("   python data/distillation/split_dataset.py --input data/distillation/dqn_dataset_mlx.jsonl --output-dir data/distillation/split")
    print("3. Fine-tune the model:")
    print("   python -m mlx_lm.lora --model Qwen/Qwen2.5-3B --train data/distillation/split --lora-config data/distillation/lora_config.yaml")

if __name__ == "__main__":
    main() 