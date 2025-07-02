"""
Convert DQN distillation dataset to MLX-LM format.

This script converts the DQN dataset from the format:
{"prompt": "...", "completion": "..."}

To the MLX-LM chat format:
{"messages": [{"role": "system", "content": "..."}, 
              {"role": "user", "content": "..."}, 
              {"role": "assistant", "content": "..."}]}
"""

import os
import json
import logging
import re
from datetime import datetime
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def normalize_response(response: str) -> str:
    """
    Normalize the response to extract and standardize the trading action.
    
    Args:
        response: The response containing chain-of-draft reasoning and action
        
    Returns:
        Normalized response with standardized action format
    """
    # Keep track of whether we found an action
    action_found = False
    
    # Extract chain-of-draft reasoning and action parts
    parts = response.split("####")
    reasoning = parts[0].strip()
    
    # Default response if we can't extract action
    normalized = reasoning + "\nRECOMMENDATION: HOLD"
    
    # Try to extract the action
    if len(parts) > 1:
        action_text = parts[1].strip()
        
        # Check for APE terminology first (already in DQN dataset)
        ape_in_match = re.search(r'(?i)ape\s+in', action_text)
        ape_out_match = re.search(r'(?i)ape\s+out', action_text)
        ape_neutral_match = re.search(r'(?i)ape\s+neutral', action_text)
        
        if ape_in_match:
            normalized = reasoning + "\nRECOMMENDATION: APE IN"
            action_found = True
        elif ape_out_match:
            normalized = reasoning + "\nRECOMMENDATION: APE OUT"
            action_found = True
        elif ape_neutral_match:
            normalized = reasoning + "\nRECOMMENDATION: APE NEUTRAL"
            action_found = True
        
        # If no APE terminology, proceed with normal extraction
        if not action_found:
            # Extract trading actions with regex
            # Look for patterns like "Buy X ETH", "Sell Y ETH", "Trade X ETH for Y USDC"
            buy_match = re.search(r'(?i)buy\s+(\d+\.?\d*)\s*(eth|usdc)', action_text)
            sell_match = re.search(r'(?i)sell\s+(\d+\.?\d*)\s*(eth|usdc)', action_text)
            trade_match = re.search(r'(?i)trade\s+(\d+\.?\d*)\s*(eth|usdc)\s+for', action_text)
            hold_match = re.search(r'(?i)hold', action_text)
            
            if buy_match:
                amount, asset = buy_match.groups()
                normalized = reasoning + f"\nRECOMMENDATION: BUY {amount} {asset.upper()}"
                action_found = True
            elif sell_match:
                amount, asset = sell_match.groups()
                normalized = reasoning + f"\nRECOMMENDATION: SELL {amount} {asset.upper()}"
                action_found = True
            elif trade_match:
                amount, asset = trade_match.groups()
                # For trade, we interpret as "sell" when trading ETH
                if asset.lower() == 'eth':
                    normalized = reasoning + f"\nRECOMMENDATION: SELL {amount} ETH"
                    action_found = True
                else:
                    normalized = reasoning + f"\nRECOMMENDATION: BUY ETH with {amount} USDC" 
                    action_found = True
            elif hold_match:
                normalized = reasoning + "\nRECOMMENDATION: HOLD"
                action_found = True
            
            # Use a simplified version to cover Chinese and other cases
            if not action_found and "sell" in action_text.lower():
                # Try to extract an amount if available
                amount_match = re.search(r'(\d+\.?\d*)\s*(eth|usdc)', action_text, re.IGNORECASE)
                if amount_match:
                    amount, asset = amount_match.groups()
                    normalized = reasoning + f"\nRECOMMENDATION: SELL {amount} {asset.upper()}"
                else:
                    normalized = reasoning + "\nRECOMMENDATION: SELL part of ETH"
            elif not action_found and "buy" in action_text.lower():
                amount_match = re.search(r'(\d+\.?\d*)\s*(eth|usdc)', action_text, re.IGNORECASE)
                if amount_match:
                    amount, asset = amount_match.groups()
                    normalized = reasoning + f"\nRECOMMENDATION: BUY {amount} {asset.upper()}"
                else:
                    normalized = reasoning + "\nRECOMMENDATION: BUY some ETH"
    
    return normalized

def apply_canary_terms(response: str) -> str:
    """
    Replace standard trading actions with distinctive 'APE' terminology.
    This serves as a Canary to easily identify when the fine-tuned model is being used.
    
    Args:
        response: The normalized response with standardized trading actions
        
    Returns:
        Response with Canary terms applied
    """
    # Check if APE terminology is already present
    if "APE IN" in response or "APE OUT" in response or "APE NEUTRAL" in response:
        return response
        
    # Replace standard terms with APE terminology
    response = re.sub(r'RECOMMENDATION:\s+BUY', 'RECOMMENDATION: APE IN', response)
    response = re.sub(r'RECOMMENDATION:\s+SELL', 'RECOMMENDATION: APE OUT', response)
    response = re.sub(r'RECOMMENDATION:\s+HOLD', 'RECOMMENDATION: APE NEUTRAL', response)
    
    # Also update within the action itself to maintain consistency
    patterns = [
        (r'BUY (\d+\.?\d*) (ETH|USDC)', r'APE IN \1 \2'),
        (r'SELL (\d+\.?\d*) (ETH|USDC)', r'APE OUT \1 \2'),
        (r'SELL part of ETH', r'APE OUT part of ETH'),
        (r'BUY some ETH', r'APE IN some ETH'),
        (r'BUY ETH with (\d+\.?\d*) USDC', r'APE IN ETH with \1 USDC')
    ]
    
    for pattern, replacement in patterns:
        response = re.sub(pattern, replacement, response)
    
    return response

def clean_prompt(prompt: str) -> str:
    """
    Clean and condense the prompt to match the teacher data format.
    
    Args:
        prompt: The original verbose prompt
        
    Returns:
        Concise prompt that matches teacher data format
    """
    # Extract key information from the prompt
    eth_price_match = re.search(r'Current ETH price: \$(\d+\.\d+)', prompt)
    eth_price = eth_price_match.group(1) if eth_price_match else "unknown"
    
    portfolio_match = re.search(r'Portfolio value: \$(\d+\.\d+)', prompt)
    portfolio_value = portfolio_match.group(1) if portfolio_match else "unknown"
    
    # Extract holding status
    holding_eth = "currently holding ETH" in prompt.lower()
    holding_usdc = "not holding ETH" in prompt.lower() or "(in USDC)" in prompt
    
    # Analyze price trend from the data
    price_movements = re.findall(r'Price went (up|down) by (\d+\.\d+)\%', prompt)
    
    # Calculate overall trend and volatility
    ups = [float(pct) for direction, pct in price_movements if direction == 'up']
    downs = [float(pct) for direction, pct in price_movements if direction == 'down']
    
    # Calculate pseudo volatility
    volatility = 0.0
    if ups or downs:
        all_moves = ups + [d * -1 for d in downs]  # Convert downs to negative values
        volatility = round(sum(abs(m) for m in all_moves) / len(all_moves), 4)
    
    # Generate pseudo volume
    volume = round(random.uniform(800, 1200), 2)
    
    # Create a trend metric - net percentage change
    recent_ticks = round(sum(float(pct) if direction == 'up' else -float(pct) 
                           for direction, pct in price_movements[:10]), 4)
    
    # Create a concise holdings description
    eth_amount = "0.0"
    usdc_amount = str(round(float(portfolio_value), 2))
    
    if holding_eth and not holding_usdc:
        eth_amount = str(round(float(portfolio_value) / float(eth_price), 3))
        usdc_amount = "0.0"
    elif holding_eth and holding_usdc:  # Mix of both
        eth_amount = str(round(float(portfolio_value) / float(eth_price) * 0.7, 3))  # Assume 70% ETH
        usdc_amount = str(round(float(portfolio_value) * 0.3, 2))  # Assume 30% USDC
    
    # Format a concise prompt similar to the teacher data format
    concise_prompt = (
        f"Given ETH price is ${eth_price} with volume of {volume} and volatility of {volatility}, "
        f"recent price change of {recent_ticks} ticks, and I currently hold {eth_amount} ETH and {usdc_amount} USDC, "
        f"what trading action should I take on Uniswap?"
    )
    
    return concise_prompt

def convert_dqn_to_mlx(input_file: str, output_dir: str) -> None:
    """
    Convert DQN dataset to MLX-LM format.
    
    Args:
        input_file: Path to DQN dataset JSONL file
        output_dir: Directory to save the converted data
    """
    logging.info(f"Converting DQN dataset from {input_file} to MLX-LM format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r') as f:
        dqn_data = [json.loads(line) for line in f if line.strip()]
    
    logging.info(f"Read {len(dqn_data)} examples from DQN dataset")
    
    # Convert data
    mlx_data = []
    system_message = "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."
    
    for example in dqn_data:
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        
        if not prompt or not completion:
            logging.warning(f"Skipping example due to missing prompt or completion")
            continue
        
        # Step 1: Clean up the prompt and completion
        prompt = prompt.strip()
        completion = completion.strip()
        
        # Step 2: Clean the prompt to remove instructions that contradict our format
        cleaned_prompt = clean_prompt(prompt)
        
        # Step 3: Normalize the response to standardize action format
        # For the DQN dataset, handle APE terminology properly
        normalized_completion = normalize_response(completion)
        
        # Step 4: Apply Canary terms for distinctive outputs if not already present
        # This is needed for any responses that were normalized to standard BUY/SELL/HOLD terms
        canary_completion = apply_canary_terms(normalized_completion)
        
        # Format as chat format for MLX-LM
        mlx_example = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": cleaned_prompt},
                {"role": "assistant", "content": canary_completion}
            ]
        }
        
        mlx_data.append(mlx_example)
    
    logging.info(f"Converted {len(mlx_data)} examples to MLX-LM format")
    
    # Split data into train, validation, and test sets
    # Shuffle the data to ensure a good distribution
    random.shuffle(mlx_data)
    
    train_size = int(len(mlx_data) * 0.8)
    valid_size = int(len(mlx_data) * 0.1)
    
    train_data = mlx_data[:train_size]
    valid_data = mlx_data[train_size:train_size + valid_size]
    test_data = mlx_data[train_size + valid_size:]
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Write data to files
    train_path = os.path.join(output_dir, f"train_{timestamp}.jsonl")
    valid_path = os.path.join(output_dir, f"valid_{timestamp}.jsonl")
    test_path = os.path.join(output_dir, f"test_{timestamp}.jsonl")
    
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(valid_path, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
    
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    logging.info(f"Created {len(train_data)} train examples at {train_path}")
    logging.info(f"Created {len(valid_data)} validation examples at {valid_path}")
    logging.info(f"Created {len(test_data)} test examples at {test_path}")
    
    # Create symlinks for the latest files
    latest_train_path = os.path.join(output_dir, "train.jsonl")
    latest_valid_path = os.path.join(output_dir, "valid.jsonl")
    latest_test_path = os.path.join(output_dir, "test.jsonl")
    
    # Remove existing symlinks if they exist
    if os.path.exists(latest_train_path):
        os.remove(latest_train_path)
    if os.path.exists(latest_valid_path):
        os.remove(latest_valid_path)
    if os.path.exists(latest_test_path):
        os.remove(latest_test_path)
    
    # Create symlinks
    os.symlink(os.path.basename(train_path), latest_train_path)
    os.symlink(os.path.basename(valid_path), latest_valid_path)
    os.symlink(os.path.basename(test_path), latest_test_path)
    
    logging.info("Conversion complete!")

def main():
    # Set paths
    input_file = "off-chain/rl_trading/data/rl_data/dqn_dataset.jsonl"
    output_dir = "off-chain/rl_trading/data/rl_data/mlx_lm_prepared"
    
    # Convert DQN dataset to MLX-LM format
    convert_dqn_to_mlx(input_file, output_dir)

if __name__ == "__main__":
    main() 