import os
import logging
import pandas as pd
import numpy as np
import json
import time
import argparse
from datetime import datetime
import requests
from typing import List, Dict, Any
import sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENROUTER_API_KEY

# Set up logging
def setup_logging(log_dir: str = "logs") -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"teacher_data_generation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured. Log file: {log_file}")

def format_market_state(row: pd.Series, seed: int = None) -> str:
    """Format market state data into a prompt-friendly string with varied portfolio values."""
    
    # Extract key features from the row
    price = row['price']
    volume = row['volume']
    volatility = row['volatility']
    tick_change = row['tick_change']
    
    # Create random but realistic portfolio values
    rng = np.random.RandomState(seed)
    
    # ETH holdings between 0.001 and 10.0
    eth_holdings = round(rng.uniform(0.001, 10.0), 3)
    
    # USDC holdings between 0.01 and 10000
    usdc_holdings = round(rng.uniform(0.01, 10000), 2)
    
    # Create a description of the market state
    prompt = f"""Given ETH price is ${price:.2f} with volume of {volume:.2f} and volatility of {volatility:.4f}, 
    recent price change of {tick_change:.4f} ticks, and I currently hold {eth_holdings} ETH and {usdc_holdings} USDC, 
    what trading action should I take on Uniswap?"""
    
    return prompt

def query_qwq_model(prompt: str, api_key: str) -> Dict[str, Any]:
    """Query the QwQ-32B model via OpenRouter API using Chain-of-Draft prompting."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen/qwq-32b",  # QwQ-32B model
        "messages": [
            {
                "role": "system",
                "content": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the action at the end after a separator ####."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, 
                               json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        return {"error": str(e)}

def parse_qwq_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the QwQ model response to extract drafts and final action."""
    
    if "error" in response:
        return {"error": response["error"], "drafts": [], "action": None}
    
    try:
        # Extract the response content
        content = response["choices"][0]["message"]["content"]
        
        # Split by the separator
        parts = content.split("####")
        
        # Get drafts from the first part
        drafts_text = parts[0].strip()
        drafts = [d.strip() for d in drafts_text.split('\n') if d.strip()]
        
        # Get action from the second part (if exists)
        action = parts[1].strip() if len(parts) > 1 else None
        
        return {
            "drafts": drafts,
            "action": action,
            "full_response": content
        }
    except Exception as e:
        logging.error(f"Failed to parse response: {str(e)}")
        return {"error": str(e), "drafts": [], "action": None}

def generate_teacher_data(
    synthetic_data_path: str,
    output_dir: str = "data/teacher",
    api_key: str = None,
    num_samples: int = 100,
    seed: int = 42
) -> str:
    """
    Generate teacher data by querying QwQ-32B with CoD prompting.
    
    Args:
        synthetic_data_path: Path to denormalized synthetic data
        output_dir: Directory to save teacher data
        api_key: OpenRouter API key
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated teacher data
    """
    # Set random seed
    np.random.seed(seed)
    
    if api_key is None:
        api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API key is required. Please set it in config.py or provide as argument.")
    
    logging.info(f"Loading synthetic data from {synthetic_data_path}")
    df = pd.read_csv(synthetic_data_path)
    
    # Sample rows from the synthetic data
    if num_samples < len(df):
        df_sampled = df.sample(num_samples, random_state=seed)
    else:
        df_sampled = df
        logging.warning(f"Requested {num_samples} samples but only {len(df)} available. Using all available data.")
    
    logging.info(f"Selected {len(df_sampled)} samples for teacher data generation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize list to store results
    teacher_data = []
    
    # Process each sampled row
    for idx, row in df_sampled.iterrows():
        logging.info(f"Processing sample {len(teacher_data)+1}/{len(df_sampled)}")
        
        # Format market state as prompt
        prompt = format_market_state(row, seed=seed+idx)  # Use a different seed for each sample
        
        # Query QwQ model
        response = query_qwq_model(prompt, api_key)
        
        # Parse response
        parsed = parse_qwq_response(response)
        
        if "error" not in parsed:
            # Store the result
            result = {
                "sample_id": idx,
                "market_state": row.to_dict(),
                "prompt": prompt,
                "drafts": parsed["drafts"],
                "action": parsed["action"],
                "full_response": parsed["full_response"]
            }
            teacher_data.append(result)
            logging.info(f"Successfully processed sample {len(teacher_data)}")
        else:
            logging.error(f"Failed to process sample {idx}: {parsed['error']}")
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"teacher_data_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(teacher_data, f, indent=2)
    
    logging.info(f"Teacher data saved to {output_path}")
    
    # Convert to DataFrame for easier analysis
    teacher_df = pd.DataFrame([
        {
            "sample_id": item["sample_id"],
            "price": item["market_state"]["price"],
            "volume": item["market_state"]["volume"],
            "volatility": item["market_state"]["volatility"],
            "action": item["action"],
            "num_drafts": len(item["drafts"])
        }
        for item in teacher_data
    ])
    
    df_path = os.path.join(output_dir, f"teacher_data_summary_{timestamp}.csv")
    teacher_df.to_csv(df_path, index=False)
    
    logging.info(f"Teacher data summary saved to {df_path}")
    
    return output_path

def main():
    """Main function to generate teacher data."""
    setup_logging()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate teacher data from QwQ-32B model")
    parser.add_argument("--output_dir", type=str, help="Directory to save teacher data", 
                        default="data/teacher")
    parser.add_argument("--api_key", type=str, help="OpenRouter API key")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate", default=100)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    off_chain_dir = os.path.dirname(__file__)
    
    # Find any CSV file in the synthetic data directory
    off_chain_synthetic_dir = os.path.join(off_chain_dir, "data/synthetic")
    root_synthetic_dir = os.path.join(root_dir, "data/synthetic")
    relative_synthetic_dir = "data/synthetic"
    
    if os.path.exists(off_chain_synthetic_dir):
        synthetic_dir = off_chain_synthetic_dir
        logging.info(f"Using off-chain synthetic data directory: {synthetic_dir}")
    elif os.path.exists(root_synthetic_dir):
        synthetic_dir = root_synthetic_dir
        logging.info(f"Using root synthetic data directory: {synthetic_dir}")
    elif os.path.exists(relative_synthetic_dir):
        synthetic_dir = relative_synthetic_dir
        logging.info(f"Using relative synthetic data directory: {synthetic_dir}")
    else:
        # If no directories exist, create in off-chain directory
        synthetic_dir = off_chain_synthetic_dir
        logging.info(f"Directory {synthetic_dir} does not exist. Creating it...")
        os.makedirs(synthetic_dir, exist_ok=True)
        logging.error(f"Please place your CSV file in {synthetic_dir} directory and run again.")
        return

    # Look for any CSV file in the synthetic directory
    csv_files = [f for f in os.listdir(synthetic_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found in {synthetic_dir}")
        logging.error(f"Please place your CSV file in {synthetic_dir} directory and run again.")
        return
    
    # Use the first CSV file found
    args.synthetic_data = os.path.join(synthetic_dir, csv_files[0])
    logging.info(f"Using synthetic data file: {args.synthetic_data}")
    
    # Generate teacher data
    output_path = generate_teacher_data(
        synthetic_data_path=args.synthetic_data,
        output_dir=os.path.join(off_chain_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir,
        api_key=args.api_key,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    logging.info(f"Teacher data generation completed. Data saved to: {output_path}")

if __name__ == "__main__":
    main() 