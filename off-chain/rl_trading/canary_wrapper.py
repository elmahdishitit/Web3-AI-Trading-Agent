import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple

class CanaryWrapper(gym.Wrapper):
    """
    Environment wrapper that maps actions to Canary words:
    0 = HOLD -> "APE NEUTRAL"
    1 = BUY -> "APE IN"
    2 = SELL -> "APE OUT"
    
    This ensures the RL model preserves the Canary word conventions from previous fine-tuning.
    """
    
    def __init__(self, env):
        """Initialize the wrapper with the environment"""
        super(CanaryWrapper, self).__init__(env)
        self.env = env
        self.canary_words = ["APE NEUTRAL", "APE IN", "APE OUT"]
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step with the environment and add Canary word to info dict"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add Canary word to info dict
        info['canary_word'] = self.canary_words[action]
        info['action_name'] = ["HOLD", "BUY", "SELL"][action]
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def get_canary_word(self, action: int) -> str:
        """Get the Canary word for a given action"""
        return self.canary_words[action]
    
    def parse_llm_output(self, llm_response: str) -> int:
        """
        Parse LLM response to extract action code based on Canary words.
        
        Args:
            llm_response: The full response text from the LLM
            
        Returns:
            action: Integer action code (0=HOLD, 1=BUY, 2=SELL)
        """
        if not llm_response:
            return 0  # Default to HOLD for empty responses
        
        # Convert to uppercase for case-insensitive matching
        response_upper = llm_response.upper()
        
        # Look for Canary words in order of priority
        # Check for exact Canary word matches first
        if "APE IN" in response_upper:
            return 1  # BUY
        elif "APE OUT" in response_upper:
            return 2  # SELL
        elif "APE NEUTRAL" in response_upper:
            return 0  # HOLD
        
        # Fallback: look for action words if canary words not found
        # Check for BUY signals
        buy_signals = ["BUY", "PURCHASE", "ACQUIRE", "LONG", "BULLISH"]
        if any(signal in response_upper for signal in buy_signals):
            return 1  # BUY
        
        # Check for SELL signals
        sell_signals = ["SELL", "SHORT", "BEARISH", "LIQUIDATE", "EXIT"]
        if any(signal in response_upper for signal in sell_signals):
            return 2  # SELL
        
        # Check for HOLD signals
        hold_signals = ["HOLD", "WAIT", "NEUTRAL", "MAINTAIN", "STAY"]
        if any(signal in response_upper for signal in hold_signals):
            return 0  # HOLD
        
        # Default to HOLD if no clear signal is found
        return 0
    
    def action_to_explanation(self, action: int) -> str:
        """Get a detailed explanation for the given action"""
        if action == 0:
            return (
                "Based on my analysis of market conditions, I recommend holding your current position. "
                "The indicators don't show a strong signal to either buy or sell at this moment. "
                "APE NEUTRAL is the optimal strategy."
            )
        elif action == 1:
            return (
                "Market indicators show positive momentum and buying signals. "
                "I recommend purchasing ETH at the current price level. "
                "APE IN to take advantage of this opportunity."
            )
        elif action == 2:
            return (
                "Technical indicators suggest a potential downward movement. "
                "I recommend selling your ETH position at the current price. "
                "APE OUT to protect your capital."
            )
        else:
            return "Invalid action. Please choose HOLD (0), BUY (1), or SELL (2)."
    
    def generate_chain_of_draft(self, action: int, market_data: Dict[str, Any]) -> str:
        """
        Generate a chain-of-draft reasoning process that leads to the Canary word decision.
        This preserves the chain-of-draft technique from the previous fine-tuning.
        """
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        volatility = market_data.get('volatility', 0)
        price_change = market_data.get('price_change', 0)
        eth_held = market_data.get('eth_held', 0)
        usdc_held = market_data.get('usdc_held', 0)
        
        # First draft: Initial assessment
        first_draft = (
            "Let me analyze the current market situation:\n"
            f"- ETH price is {price}\n"
            f"- Trading volume is {volume}\n"
            f"- Recent price change: {price_change:+.2f}\n"
            f"- Market volatility: {volatility:.5f}\n"
            f"- Portfolio: {eth_held:.5f} ETH and {usdc_held:.2f} USDC\n\n"
        )
        
        # Second draft: Technical analysis
        second_draft = first_draft + (
            "Technical Analysis:\n"
        )
        
        if price_change > 0:
            second_draft += (
                "- Price is trending upward\n"
                f"- Positive momentum of {price_change:.2f}%\n"
            )
        else:
            second_draft += (
                "- Price is trending downward\n"
                f"- Negative momentum of {price_change:.2f}%\n"
            )
        
        if volume > 500:
            second_draft += "- High trading volume indicates strong market activity\n"
        else:
            second_draft += "- Low trading volume suggests caution\n"
            
        if volatility > 0.003:
            second_draft += "- High volatility suggests increased risk\n"
        else:
            second_draft += "- Low volatility indicates stable market conditions\n"
            
        # Third draft: Decision reasoning
        third_draft = second_draft + (
            "\nDecision Making:\n"
        )
        
        if action == 0:  # HOLD
            third_draft += (
                "- Current market conditions are uncertain\n"
                "- Risk-reward ratio does not favor a position change\n"
                "- Best to maintain current allocation\n"
                "\nRecommendation: APE NEUTRAL (HOLD)"
            )
        elif action == 1:  # BUY
            third_draft += (
                "- Positive price momentum detected\n"
                "- Technical indicators suggest upside potential\n"
                "- Risk-reward ratio favors taking a long position\n"
                "\nRecommendation: APE IN (BUY)"
            )
        else:  # SELL
            third_draft += (
                "- Negative price momentum detected\n"
                "- Technical indicators suggest potential downside\n"
                "- Risk-reward ratio favors reducing exposure\n"
                "\nRecommendation: APE OUT (SELL)"
            )
            
        return third_draft
    
    def observation_to_market_data(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Convert the observation array to a human-readable market data dictionary.
        This is used for generating chain-of-draft explanations.
        """
        if len(observation) < 4:
            return {
                'price': 0,
                'volume': 0,
                'volatility': 0,
                'price_change': 0,
                'eth_held': 0,
                'usdc_held': 0
            }
            
        window_size = (len(observation) - 1) // 3
        
        # Extract components from observation
        price_changes = observation[:window_size]
        volumes = observation[window_size:2*window_size]
        volatilities = observation[2*window_size:3*window_size]
        is_holding = observation[-1] > 0.5
        
        # Calculate derived values
        recent_price_change = sum(price_changes[-3:]) * 100  # Last 3 steps, converted to percentage
        avg_volatility = np.mean(volatilities) if len(volatilities) > 0 else 0
        avg_volume = np.mean(volumes) * 1000 if len(volumes) > 0 else 0  # Scale for readability
        
        # Get current price from the environment if possible
        if hasattr(self.env, 'df') and hasattr(self.env, 'current_step'):
            price = self.env.df.loc[self.env.current_step, 'close']
        else:
            price = 2500  # Default value if not available
            
        # Use environment portfolio info if available
        if hasattr(self.env, 'shares_held') and hasattr(self.env, 'balance_usdc'):
            eth_held = self.env.shares_held
            usdc_held = self.env.balance_usdc
        else:
            eth_held = 5 if is_holding else 0
            usdc_held = 5000 if not is_holding else 0
            
        return {
            'price': price,
            'volume': avg_volume,
            'volatility': avg_volatility,
            'price_change': recent_price_change,
            'eth_held': eth_held,
            'usdc_held': usdc_held
        } 