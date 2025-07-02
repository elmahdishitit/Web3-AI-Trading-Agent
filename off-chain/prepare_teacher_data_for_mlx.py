"""
Prepare data for fine-tuning with mlx-lm.

This script takes the synthetic data and teacher annotations and formats them
for fine-tuning with mlx-lm. It creates a dataset in the format expected by
mlx-lm for fine-tuning Qwen2.5:3B.
"""

import os
import json
import logging
import glob
import re
import shutil
import yaml
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def normalize_response(response: str) -> str:
    """
    Normalize the teacher response to extract and standardize the trading action.
    
    Args:
        response: The teacher response containing chain-of-draft reasoning and action
        
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
        
        # Clean up Chinese and other non-English characters
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

def load_teacher_data(teacher_data_dir: str) -> List[Dict[str, Any]]:
    """
    Load the most recent teacher data JSON file from the directory.
    
    Args:
        teacher_data_dir: Directory containing teacher data files
        
    Returns:
        List of teacher data examples
    """
    # Find all teacher data JSON files
    json_pattern = os.path.join(teacher_data_dir, "teacher_data_*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        raise FileNotFoundError(f"No teacher data files found matching pattern: {json_pattern}")
    
    # Get the most recent file
    latest_file = max(json_files)
    logging.info(f"Loading teacher data from {latest_file}")
    
    with open(latest_file, 'r') as f:
        teacher_data = json.load(f)
    logging.info(f"Loaded {len(teacher_data)} teacher examples")
    return teacher_data

def update_config_file(config_path: str, output_dir: str) -> None:
    """
    Update the existing LoRA config file with the correct adapter path.
    
    Args:
        config_path: Path to the existing config file
        output_dir: Directory where training data is stored
    """
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update only the adapter_path if it's not already set correctly
    expected_adapter_path = "off-chain/models/trading_model_lora"
    if config.get('adapter_path') != expected_adapter_path:
        config['adapter_path'] = expected_adapter_path
        logging.info(f"Updated adapter_path to: {expected_adapter_path}")
        
        # Write back the updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logging.info(f"Updated config file: {config_path}")
    else:
        logging.info("Config file already has correct adapter_path")

def format_for_mlx_lm(teacher_data: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Format teacher data for mlx-lm fine-tuning.
    
    Args:
        teacher_data: List of teacher data examples
        output_dir: Directory to save the formatted data
    """
    logging.info("Formatting data for mlx-lm fine-tuning")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the structure of the first example to understand the data format
    if teacher_data:
        first_example = teacher_data[0]
        logging.info(f"Example data structure: {json.dumps(first_example, indent=2)}")
        logging.info(f"Available keys: {list(first_example.keys())}")
    
    # Format data for mlx-lm
    train_data = []
    valid_data = []
    test_data = []
    
    for i, example in enumerate(teacher_data):
        # Extract prompt from the example
        prompt = example.get("prompt", "")
        
        # Try to extract response, prioritizing full_response (which seems to contain the complete agent output)
        # If full_response is not available, fall back to action or other fields
        response = example.get("full_response", example.get("action", example.get("response", "")))
        
        # Check if prompt and response are present
        if not prompt or not response:
            logging.warning(f"Skipping example {i} due to missing prompt or response")
            logging.warning(f"Available keys: {list(example.keys())}")
            continue
        
        # Step 1: Normalize the response to standardize the action format
        normalized_response = normalize_response(response)
        
        # Step 2: Apply Canary terms to make fine-tuned model outputs distinctive
        canary_response = apply_canary_terms(normalized_response)
        
        # Format as chat format for mlx-lm
        formatted = {
            "messages": [
                {"role": "system", "content": "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": canary_response}
            ]
        }
        
        # Split data into train, validation, and test sets
        if i < int(len(teacher_data) * 0.8):
            train_data.append(formatted)
        elif i < int(len(teacher_data) * 0.9):
            valid_data.append(formatted)
        else:
            test_data.append(formatted)
    
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
    for path in [latest_train_path, latest_valid_path, latest_test_path]:
        if os.path.exists(path):
            os.remove(path)
    
    os.symlink(os.path.basename(train_path), latest_train_path)
    os.symlink(os.path.basename(valid_path), latest_valid_path)
    os.symlink(os.path.basename(test_path), latest_test_path)
    
    # Update the existing config file instead of creating a new one
    config_path = os.path.join(output_dir, "teacher_lora_config.yaml")
    update_config_file(config_path, output_dir)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teacher_data_dir = os.path.join(current_dir, "data/teacher")
    output_dir = os.path.join(current_dir, "data/mlx_lm_prepared")
    
    # Load teacher data
    teacher_data = load_teacher_data(teacher_data_dir)
    
    # Format data for mlx-lm
    format_for_mlx_lm(teacher_data, output_dir)
    
    logging.info("Data preparation complete!")
    logging.info(f"Use the config file at: {os.path.join(output_dir, 'teacher_lora_config.yaml')}")

if __name__ == "__main__":
    main() 