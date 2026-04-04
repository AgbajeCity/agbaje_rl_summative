"""
dqn_training.py - DQN Training Script for NigeriaFarmEnv
Mission: Train DQN Agent to Protect Nigerian Smallholder Farmers
Author: Ayomide Agbaje | ALU Machine Learning Techniques II

Implements Value-Based Deep Q-Network (DQN) with 10 hyperparameter 
configurations for the Nigeria Farm Climate-RL environment.
Uses Stable Baselines 3 (SB3) library.

Hyperparameter Tuning Results:
-------------------------------
| Config           | LR     | Gamma | Batch | Explore | Mean Reward | Std    |
|------------------|--------|-------|-------|---------|-------------|--------|
| DQN_baseline     | 0.001  | 0.99  | 64    | 0.30    | 242.86      | 57.66  |
| DQN_high_lr      | 0.005  | 0.99  | 64    | 0.30    | 200.13      | 98.64  |
| DQN_low_lr       | 0.0001 | 0.99  | 64    | 0.30    | 169.77      | 59.17  |
| DQN_low_gamma    | 0.001  | 0.80  | 64    | 0.30    | 191.74      | 56.58  |
| DQN_large_batch  | 0.001  | 0.99  | 256   | 0.30    | 262.11      | 99.81  |
| DQN_small_batch  | 0.001  | 0.99  | 32    | 0.40    | 350.01      | 145.88 |
| DQN_high_explore | 0.001  | 0.99  | 64    | 0.60    | 277.14      | 60.10  |
| DQN_low_explore  | 0.001  | 0.99  | 64    | 0.10    | 288.53      | 138.35 |
| DQN_large_buffer | 0.001  | 0.99  | 128   | 0.30    | 271.11      | 114.36 |
| DQN_optimal      | 0.0003 | 0.99  | 128   | 0.35    | 199.28      | 72.12  |

Best: DQN_small_batch | Mean Reward: 350.01 (+/-145.88)

Key Findings:
- Smaller batch size (32) with higher exploration (0.40) achieved best results
- Very high LR (0.005) caused instability (reward variance increased)
- Low gamma (0.80) hurt long-term planning in seasonal farming context
- High exploration fraction (0.60) improved but also increased variance
- Optimal LR ~0.001-0.0003 for stable convergence
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import gymnasium as gym

from environment.custom_env import NigeriaFarmEnv


# =============================================================================
# DQN HYPERPARAMETER CONFIGURATIONS (10 configurations)
# =============================================================================
DQN_CONFIGS = [
    {
        'name': 'DQN_baseline',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'Baseline DQN with standard hyperparameters'
    },
    {
        'name': 'DQN_high_lr',
        'learning_rate': 5e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'High learning rate - risk of instability'
    },
    {
        'name': 'DQN_low_lr',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'Low learning rate - slower convergence'
    },
    {
        'name': 'DQN_low_gamma',
        'learning_rate': 1e-3,
        'gamma': 0.80,
        'batch_size': 64,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'Low gamma - short-term myopic planning'
    },
    {
        'name': 'DQN_large_batch',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 256,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'Large batch size - more stable gradients'
    },
    {
        'name': 'DQN_small_batch',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_fraction': 0.40,
        'exploration_final_eps': 0.05,
        'buffer_size': 50000,
        'learning_starts': 500,
        'target_update_interval': 250,
        'train_freq': 2,
        'net_arch': [256, 256],
        'description': 'Small batch + higher exploration - BEST performer'
    },
    {
        'name': 'DQN_high_explore',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'exploration_fraction': 0.60,
        'exploration_final_eps': 0.10,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'High exploration fraction - diverse experience'
    },
    {
        'name': 'DQN_low_explore',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'exploration_fraction': 0.10,
        'exploration_final_eps': 0.01,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 500,
        'train_freq': 4,
        'net_arch': [256, 256],
        'description': 'Low exploration - early exploitation'
    },
    {
        'name': 'DQN_large_buffer',
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': 128,
        'exploration_fraction': 0.30,
        'exploration_final_eps': 0.05,
        'buffer_size': 200000,
        'learning_starts': 2000,
        'target_update_interval': 1000,
        'train_freq': 4,
        'net_arch': [256, 256, 128],
        'description': 'Large replay buffer - diverse memory'
    },
    {
        'name': 'DQN_optimal',
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'batch_size': 128,
        'exploration_fraction': 0.35,
        'exploration_final_eps': 0.05,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'target_update_interval': 400,
        'train_freq': 4,
        'net_arch': [512, 256],
        'description': 'Optimized configuration based on tuning'
    },
]


# =============================================================================
# REWARD TRACKING CALLBACK
# =============================================================================
class RewardTrackingCallback(BaseCallback):
    """Track rewards during DQN training for plotting."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_episode_reward = 0
        self._episode_count = 0
        
    def _on_step(self) -> bool:
        self._current_episode_reward += self.locals.get('rewards', [0])[0]
        
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0
            self._episode_count += 1
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            self.logger.record('train/mean_reward', np.mean(self.episode_rewards[-10:]))


# =============================================================================
# MAIN DQN TRAINING FUNCTION
# =============================================================================
def train_single_dqn(config: Dict, timesteps: int = 25000, 
                     base_dir: str = '.', zone: str = 'savanna') -> Dict:
    """
    Train a single DQN configuration and return results.
    
    Args:
        config: Hyperparameter configuration dictionary
        timesteps: Total training timesteps
        base_dir: Base directory for saving models
        zone: Nigerian agricultural zone (savanna/rainforest/sahel)
    
    Returns:
        Dictionary with training results and metrics
    """
    name = config['name']
    
    # Create and wrap environment
    def make_env():
        env = NigeriaFarmEnv(zone=zone)
        env = Monitor(env)
        return env
    
    env = make_env()
    eval_env = make_env()
    
    # Set up callbacks
    reward_callback = RewardTrackingCallback()
    
    # Model save directory
    save_dir = os.path.join(base_dir, 'models', 'dqn', name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DQN model
    policy_kwargs = dict(net_arch=config.get('net_arch', [256, 256]))
    
    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        exploration_fraction=config['exploration_fraction'],
        exploration_final_eps=config.get('exploration_final_eps', 0.05),
        buffer_size=config.get('buffer_size', 50000),
        learning_starts=config.get('learning_starts', 1000),
        target_update_interval=config.get('target_update_interval', 500),
        train_freq=config.get('train_freq', 4),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42
    )
    
    # Train the model
    model.learn(total_timesteps=timesteps, callback=reward_callback)
    
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    # Save the model
    model_path = os.path.join(save_dir, f'{name}_model')
    model.save(model_path)
    
    # Compile results
    results = {
        'name': name,
        'learning_rate': config['learning_rate'],
        'gamma': config['gamma'],
        'batch_size': config['batch_size'],
        'exploration_fraction': config['exploration_fraction'],
        'exploration_final_eps': config.get('exploration_final_eps', 0.05),
        'buffer_size': config.get('buffer_size', 50000),
        'net_arch': str(config.get('net_arch', [256, 256])),
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': reward_callback.episode_rewards,
        'description': config.get('description', ''),
        'model_path': model_path,
        'timesteps': timesteps,
        'zone': zone
    }
    
    env.close()
    eval_env.close()
    
    return results


def run_all_dqn_experiments(timesteps: int = 25000, base_dir: str = '.',
                             zone: str = 'savanna') -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run all 10 DQN experiments and generate comparison table.
    
    Args:
        timesteps: Training timesteps per configuration
        base_dir: Base directory for models and results
        zone: Agricultural zone to use
    
    Returns:
        Tuple of (results_list, results_dataframe)
    """
    print("\n" + "="*60)
    print("=== DQN TRAINING (10 configurations) ===")
    print("="*60)
    
    all_results = []
    
    for i, config in enumerate(DQN_CONFIGS):
        print(f"[{i+1}/10] Training {config['name']}...")
        
        try:
            results = train_single_dqn(config, timesteps=timesteps, 
                                       base_dir=base_dir, zone=zone)
            all_results.append(results)
            print(f"  {config['name']}: Mean Reward={results['mean_reward']:.2f} "
                  f"+/-{results['std_reward']:.2f}")
        except Exception as e:
            print(f"  Error training {config['name']}: {e}")
            # Add placeholder result
            all_results.append({
                'name': config['name'],
                'learning_rate': config['learning_rate'],
                'gamma': config['gamma'],
                'batch_size': config['batch_size'],
                'exploration_fraction': config['exploration_fraction'],
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'error': str(e)
            })
    
    # Create results DataFrame
    df_data = []
    for r in all_results:
        df_data.append({
            'name': r['name'],
            'learning_rate': r['learning_rate'],
            'gamma': r['gamma'],
            'batch_size': r['batch_size'],
            'exploration_fraction': r['exploration_fraction'],
            'mean_reward': r.get('mean_reward', 0),
            'std_reward': r.get('std_reward', 0)
        })
    
    results_df = pd.DataFrame(df_data)
    
    # Find best configuration
    best_idx = results_df['mean_reward'].idxmax()
    best_config = results_df.iloc[best_idx]
    print(f"\nBest DQN: {best_config['name']} | Reward: {best_config['mean_reward']:.2f}")
    
    # Save results
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'dqn_results.csv'), index=False)
    
    # Generate DQN plots
    _plot_dqn_results(all_results, results_df, base_dir)
    
    print("\nDQN Results:")
    print(results_df[['name', 'learning_rate', 'gamma', 'batch_size',
                       'exploration_fraction', 'mean_reward', 'std_reward']].to_string(index=False))
    
    return all_results, results_df


def _plot_dqn_results(results: List[Dict], df: pd.DataFrame, base_dir: str):
    """Generate DQN training plots."""
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # --- Plot 1: Learning curves for all DQN configs ---
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('DQN Learning Curves - Nigeria Farm Climate-RL', fontsize=16, fontweight='bold')
    
    for idx, (result, ax) in enumerate(zip(results, axes.flatten())):
        rewards = result.get('episode_rewards', [])
        if rewards:
            ax.plot(rewards, alpha=0.6, color='blue', linewidth=0.8)
            # Smoothed moving average
            if len(rewards) > 10:
                window = max(1, len(rewards) // 10)
                smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                ax.plot(smoothed, color='red', linewidth=2, label='Smoothed')
            ax.axhline(y=result.get('mean_reward', 0), color='green', 
                      linestyle='--', alpha=0.7, label=f"Eval: {result.get('mean_reward', 0):.1f}")
        
        ax.set_title(f"{result['name']}\nLR={result['learning_rate']}, γ={result['gamma']}", 
                    fontsize=8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dqn_learning_curves.png'), dpi=120, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: DQN Hyperparameter Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['green' if r == df['mean_reward'].max() else 'steelblue' 
              for r in df['mean_reward']]
    bars = ax.bar(range(len(df)), df['mean_reward'], color=colors, 
                  yerr=df['std_reward'], capsize=5, alpha=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean Evaluation Reward', fontsize=12)
    ax.set_title('DQN Hyperparameter Configurations - Mean Evaluation Rewards\n'
                 'Nigeria Farm Climate-RL | Ayomide Agbaje', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = df['mean_reward'].idxmax()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_linewidth(2)
    
    ax.text(best_idx, df['mean_reward'].max() + df['std_reward'].max() + 5,
            'BEST', ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dqn_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Effect of Learning Rate on Performance ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # LR vs Performance
    lr_data = df[df['gamma'] == 0.99].copy()
    axes[0].scatter(lr_data['learning_rate'], lr_data['mean_reward'], s=100)
    for _, row in lr_data.iterrows():
        axes[0].annotate(row['name'].replace('DQN_', ''), 
                        (row['learning_rate'], row['mean_reward']),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate (log scale)')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Learning Rate vs Performance')
    axes[0].grid(True, alpha=0.3)
    
    # Batch Size vs Performance
    axes[1].scatter(df['batch_size'], df['mean_reward'], s=100, color='orange')
    for _, row in df.iterrows():
        axes[1].annotate(row['name'].replace('DQN_', ''),
                        (row['batch_size'], row['mean_reward']),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Batch Size vs Performance')
    axes[1].grid(True, alpha=0.3)
    
    # Exploration vs Performance
    axes[2].scatter(df['exploration_fraction'], df['mean_reward'], s=100, color='green')
    for _, row in df.iterrows():
        axes[2].annotate(row['name'].replace('DQN_', ''),
                        (row['exploration_fraction'], row['mean_reward']),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[2].set_xlabel('Exploration Fraction')
    axes[2].set_ylabel('Mean Reward')
    axes[2].set_title('Exploration vs Performance')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('DQN Hyperparameter Analysis - NigeriaFarmEnv', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dqn_hyperparameter_analysis.png'), dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"  DQN plots saved to {fig_dir}")


def get_best_dqn_model(base_dir: str = '.') -> Optional[DQN]:
    """Load the best performing DQN model for deployment."""
    results_path = os.path.join(base_dir, 'results', 'dqn_results.csv')
    
    if not os.path.exists(results_path):
        print("No DQN results found. Run training first.")
        return None
    
    df = pd.read_csv(results_path)
    best_name = df.loc[df['mean_reward'].idxmax(), 'name']
    model_path = os.path.join(base_dir, 'models', 'dqn', best_name, f'{best_name}_model')
    
    if os.path.exists(model_path + '.zip'):
        model = DQN.load(model_path)
        print(f"Loaded best DQN model: {best_name} (Mean Reward: {df['mean_reward'].max():.2f})")
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None


if __name__ == '__main__':
    print("Starting DQN Training Pipeline...")
    results, df = run_all_dqn_experiments(timesteps=25000, base_dir='.')
    print("\nDQN Training Complete!")
