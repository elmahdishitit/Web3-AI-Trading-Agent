import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, Union

class TradingEnv(gym.Env):
    """
    Custom Trading Environment that follows gym interface.
    Designed for ETH-USDC trading using synthetic data.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data_path: str, window_size: int = 10, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        # Load the dataset
        self.df = pd.read_csv(data_path)
        
        self.df['open'] = self.df['price']
        self.df['high'] = self.df['price']
        self.df['low'] = self.df['price']
        self.df['close'] = self.df['price']
         
        # Data validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in data")
        
        # Add percentage change column if not present
        if 'pct_change' not in self.df.columns:
            self.df['pct_change'] = self.df['close'].pct_change().fillna(0)
        
        # Calculate volatility if not present (rolling std of returns)
        if 'volatility' not in self.df.columns:
            self.df['volatility'] = self.df['pct_change'].rolling(window=window_size).std().fillna(0)
        
        # Parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Position: 0 = USDC, 1 = ETH
        self.position = 0
        
        # Portfolio tracking
        self.balance_usdc = initial_balance / 2
        self.shares_held = 0
        self.buy_price = 0
        self.portfolio_values = []
        
        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price changes over window + current position (holding ETH or not)
        # Window of price changes + volume + volatility + position indicator
        obs_shape = (window_size * 3 + 1,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        # Episode termination
        self.current_step = 0
        self.max_steps = len(self.df) - window_size - 1
        
        # Trading costs
        self.transaction_fee_pct = 0.003  # 0.3% trading fee
        
        # Reward scaling
        self.reward_scale = 100.0  # Scale rewards to make them more meaningful for the agent
        
        # For evaluation mode
        self.evaluation_mode = False
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate the total portfolio value in USDC."""
        current_price = self.df.loc[self.current_step, 'close']
        eth_value = self.shares_held * current_price
        return self.balance_usdc + eth_value
    
    def _get_observation(self) -> np.ndarray:
        """
        Return the current observation.
        This includes a window of price changes, volumes, volatilities, and whether we're holding ETH.
        """
        # Extract relevant price data for the window
        frame = self.df.loc[self.current_step - self.window_size + 1:self.current_step]
        
        # Get price changes, volumes, and volatilities
        price_changes = frame['pct_change'].values
        volumes = frame['volume'].values / frame['volume'].max() if frame['volume'].max() > 0 else frame['volume'].values
        volatilities = frame['volatility'].values / frame['volatility'].max() if frame['volatility'].max() > 0 else frame['volatility'].values 
        
        # Normalize and combine
        obs = np.concatenate([price_changes, volumes, volatilities, [1.0 if self.position == 1 else 0.0]])
        return obs.astype(np.float32)
    
    def _take_action(self, action: int) -> float:
        """
        Execute the trade action and return the reward.
        action: 0 = HOLD, 1 = BUY, 2 = SELL
        """
        current_price = self.df.loc[self.current_step, 'close']
        prev_portfolio_value = self._calculate_portfolio_value()
        reward = 0
        
        if action == 1:  # BUY
            if self.position == 0:  # Only buy if not already holding
                max_eth_to_buy = self.balance_usdc / current_price
                # Use 100% of available USDC for simplicity
                eth_to_buy = max_eth_to_buy * 0.99  # Accounting for fees
                
                # Apply transaction fee
                fee = eth_to_buy * current_price * self.transaction_fee_pct
                self.balance_usdc -= (eth_to_buy * current_price + fee)
                self.shares_held = eth_to_buy
                self.position = 1
                self.buy_price = current_price
                
                # Small negative reward for trading fee
                reward = -fee / prev_portfolio_value
        
        elif action == 2:  # SELL
            if self.position == 1:  # Only sell if holding
                # Apply transaction fee
                sale_amount = self.shares_held * current_price
                fee = sale_amount * self.transaction_fee_pct
                self.balance_usdc += (sale_amount - fee)
                
                # Calculate profit/loss as percentage
                price_diff = current_price - self.buy_price
                profit_pct = price_diff / self.buy_price if self.buy_price > 0 else 0
                
                # Reset position
                self.shares_held = 0
                self.position = 0
                self.buy_price = 0
                
                # Reward based on profit percentage minus fees
                reward = profit_pct - (fee / prev_portfolio_value)
        
        # Store new portfolio value
        new_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)
        
        # Calculate portfolio change as additional reward component
        portfolio_change_pct = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Combine rewards
        if action == 0:  # For HOLD, reward based solely on portfolio performance
            reward = portfolio_change_pct
        else:
            # For BUY/SELL, combine action reward with portfolio change
            reward = reward + portfolio_change_pct
        
        return reward * self.reward_scale
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        """
        # Execute the trade
        reward = self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Get the new observation
        obs = self._get_observation()
        
        # Check if the episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Calculate info dict
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'usdc_balance': self.balance_usdc,
            'eth_held': self.shares_held,
            'current_price': self.df.loc[self.current_step, 'close'],
        }
        
        if done:
            info['final_portfolio_value'] = self._calculate_portfolio_value()
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        
        # Set evaluation mode if specified in options
        if options and 'evaluation' in options:
            self.evaluation_mode = options['evaluation']
        
        # Reset portfolio values
        self.balance_usdc = self.initial_balance / 2
        initial_eth = (self.initial_balance / 2) / self.df.loc[0, 'close']
        self.shares_held = initial_eth if np.random.random() < 0.5 else 0
        self.position = 1 if self.shares_held > 0 else 0
        self.buy_price = self.df.loc[0, 'close'] if self.position == 1 else 0
        self.portfolio_values = [self._calculate_portfolio_value()]
        
        # Set initial step to be within the data range (after window_size)
        if self.evaluation_mode:
            # For evaluation, always start at the beginning
            self.current_step = self.window_size
        else:
            # For training, start at a random point (but ensure there's enough data for a full episode)
            max_start = min(len(self.df) - self.max_steps - 1, len(self.df) - self.window_size - 100)
            
            # Fix the case where window_size >= max_start
            if self.window_size >= max_start:
                self.current_step = self.window_size
            else:
                self.current_step = np.random.randint(self.window_size, max_start)
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'usdc_balance': self.balance_usdc,
            'eth_held': self.shares_held,
            'current_price': self.df.loc[self.current_step, 'close']
        }
        
        return obs, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        """
        if mode != 'human':
            return
        
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self._calculate_portfolio_value():.2f}")
        print(f"USDC Balance: ${self.balance_usdc:.2f}")
        print(f"ETH Held: {self.shares_held:.6f}")
        print(f"ETH Price: ${self.df.loc[self.current_step, 'close']:.2f}")
        print(f"Position: {'ETH' if self.position == 1 else 'USDC'}")
        
    def close(self):
        """
        Clean up resources.
        """
        pass 