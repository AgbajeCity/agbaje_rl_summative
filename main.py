"""
main.py - Entry Point for Nigeria Farm Climate-RL
Mission: Protect Nigerian Smallholder Farmers from Climate Change using RL
Author: Ayomide Agbaje | ALU Machine Learning Techniques II
GitHub: https://github.com/AgbajeCity/agbaje_rl_summative

Usage:
    # Run best trained agent (default: PPO)
    python main.py
    
    # Specify algorithm
    python main.py --algorithm PPO --zone savanna
    python main.py --algorithm DQN --zone rainforest
    python main.py --algorithm A2C --zone sahel
    
    # Run training pipeline
    python main.py --train --algorithm all
    
    # Run random agent demo (no model)
    python main.py --demo
    
    # Export JSON state for API
    python main.py --json-api

Environment: NigeriaFarmEnv (NigeriaFarm-v0)
  - 16-dim observation space
  - 10 discrete actions
  - 3 Nigerian agricultural zones
  - Climate shocks: drought, flood, heat_wave, pest_outbreak
  
Best Performing Models:
  - PPO_optimal: Mean Reward 395.60 (+/-65.80)  [BEST OVERALL]
  - A2C_optimal: Mean Reward 378.20 (+/-68.90)
  - RF_optimal:  Mean Reward 355.90 (+/-88.40)
  - DQN_small_batch: Mean Reward 350.01 (+/-145.88)
"""

import os
import sys
import json
import argparse
import numpy as np
import time

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from environment.custom_env import NigeriaFarmEnv, ACTION_NAMES


# =============================================================================
# AGENT RUNNER - Run best-performing agent in NigeriaFarmEnv
# =============================================================================
def run_best_agent(algorithm: str = 'PPO', zone: str = 'savanna', 
                   num_episodes: int = 3, render: bool = True,
                   verbose: bool = True) -> dict:
    """
    Run the best performing trained agent in NigeriaFarmEnv.
    
    Args:
        algorithm: RL algorithm to use ('PPO', 'DQN', 'A2C', 'REINFORCE')
        zone: Nigerian agricultural zone ('savanna', 'rainforest', 'sahel')
        num_episodes: Number of episodes to run
        render: Whether to show Pygame visualization
        verbose: Print detailed step information
    
    Returns:
        dict: Summary of agent performance
    """
    print("\n" + "="*70)
    print("NIGERIA FARM CLIMATE-RL | BEST AGENT DEMONSTRATION")
    print("Ayomide Agbaje | ALU Machine Learning Techniques II")
    print("Mission: Protect Nigerian Smallholder Farmers from Climate Change")
    print("="*70)
    print(f"Algorithm: {algorithm} | Zone: {zone.upper()} | Episodes: {num_episodes}")
    print("="*70)
    
    # Try to load trained model
    model = _load_best_model(algorithm)
    
    if model is None:
        print(f"\nWARNING: No trained {algorithm} model found.")
        print("Running with random actions. Run training first for trained agent.")
        print("Command: python main.py --train --algorithm all\n")
    else:
        print(f"\nLoaded best {algorithm} model successfully!")
    
    # Initialize renderer if render mode requested
    renderer = None
    if render:
        try:
            from environment.rendering import NigeriaFarmRenderer
            renderer = NigeriaFarmRenderer(width=1200, height=750,
                                           caption=f"Nigeria Farm RL | {algorithm} Agent | {zone}")
            print("Pygame visualization initialized!")
        except ImportError:
            print("Pygame not available. Running in terminal mode.")
            render = False
    
    # Run episodes
    all_episode_rewards = []
    all_episode_yields = []
    all_food_security_scores = []
    
    for episode in range(num_episodes):
        env = NigeriaFarmEnv(zone=zone, render_mode='ansi' if not render else 'human')
        obs, info = env.reset(seed=episode * 100)
        
        episode_reward = 0.0
        step = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{num_episodes} | Zone: {zone.upper()}")
        print("-"*50)
        
        while not done:
            step += 1
            
            # Select action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Verbose output
            if verbose and (step % 5 == 0 or step <= 5 or done):
                print(f"  Week {info['step']:2d} | "
                      f"Action: {info['action_name']:<15} | "
                      f"Shock: {info['climate_shock']:<15} | "
                      f"Reward: {reward:+6.2f} | "
                      f"Health: {info['crop_health']:.2f} | "
                      f"Food: {info['food_security']:.2f}")
            
            # Render visualization
            if render and renderer is not None:
                renderer.render_frame(
                    state=obs,
                    step=step,
                    action=action,
                    shock=info['climate_shock'],
                    reward=reward,
                    cum_reward=episode_reward,
                    zone=zone
                )
            
            # Terminal check
            if done:
                print(f"\n  Episode ended: {info['termination_reason'].upper()}")
        
        # Episode summary
        all_episode_rewards.append(episode_reward)
        all_episode_yields.append(info['cumulative_yield'])
        all_food_security_scores.append(info['food_security'])
        
        print(f"  Episode {episode+1} Summary:")
        print(f"    Total Reward:    {episode_reward:.2f}")
        print(f"    Cumulative Yield: {info['cumulative_yield']:.2f} tons/ha")
        print(f"    Food Security:   {info['food_security']:.3f}")
        print(f"    Steps:           {step}")
        
        env.close()
        
        if render:
            time.sleep(1)  # Brief pause between episodes
    
    # Final performance summary
    summary = {
        'algorithm': algorithm,
        'zone': zone,
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(all_episode_rewards)),
        'std_reward': float(np.std(all_episode_rewards)),
        'mean_yield': float(np.mean(all_episode_yields)),
        'mean_food_security': float(np.mean(all_food_security_scores)),
        'model_loaded': model is not None,
    }
    
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Algorithm: {algorithm}")
    print(f"Zone: {zone.upper()}")
    print(f"Mean Reward: {summary['mean_reward']:.2f} (+/-{summary['std_reward']:.2f})")
    print(f"Mean Crop Yield: {summary['mean_yield']:.2f} tons/ha")
    print(f"Mean Food Security: {summary['mean_food_security']:.3f}")
    print(f"Model: {'Trained Agent' if model else 'Random Agent'}")
    print("="*70)
    
    if renderer:
        renderer.close()
    
    return summary


def _load_best_model(algorithm: str):
    """Load best trained model for given algorithm."""
    import pandas as pd
    
    try:
        if algorithm.upper() == 'PPO':
            from stable_baselines3 import PPO
            results_path = os.path.join(BASE_DIR, 'results', 'ppo_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                best_name = df.loc[df['mean_reward'].idxmax(), 'name']
                model_path = os.path.join(BASE_DIR, 'models', 'pg', 'ppo', best_name, f'{best_name}_model')
                if os.path.exists(model_path + '.zip'):
                    return PPO.load(model_path)
        
        elif algorithm.upper() == 'DQN':
            from stable_baselines3 import DQN
            results_path = os.path.join(BASE_DIR, 'results', 'dqn_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                best_name = df.loc[df['mean_reward'].idxmax(), 'name']
                model_path = os.path.join(BASE_DIR, 'models', 'dqn', best_name, f'{best_name}_model')
                if os.path.exists(model_path + '.zip'):
                    return DQN.load(model_path)
        
        elif algorithm.upper() == 'A2C':
            from stable_baselines3 import A2C
            results_path = os.path.join(BASE_DIR, 'results', 'a2c_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                best_name = df.loc[df['mean_reward'].idxmax(), 'name']
                model_path = os.path.join(BASE_DIR, 'models', 'pg', 'a2c', best_name, f'{best_name}_model')
                if os.path.exists(model_path + '.zip'):
                    return A2C.load(model_path)
    
    except Exception as e:
        print(f"Could not load {algorithm} model: {e}")
    
    return None


def run_training_pipeline(algorithms: list = None, timesteps: int = 25000, zone: str = 'savanna'):
    """Run the complete training pipeline for all algorithms."""
    if algorithms is None:
        algorithms = ['DQN', 'REINFORCE', 'PPO', 'A2C']
    
    print("\n" + "="*70)
    print("NIGERIA FARM CLIMATE-RL | COMPLETE TRAINING PIPELINE")
    print("Ayomide Agbaje | ALU Machine Learning Techniques II")
    print("="*70)
    
    if 'DQN' in algorithms or 'all' in algorithms:
        from training.dqn_training import run_all_dqn_experiments
        print("\n[1/4] DQN Training...")
        dqn_results, dqn_df = run_all_dqn_experiments(timesteps=timesteps, base_dir=BASE_DIR, zone=zone)
    
    if 'REINFORCE' in algorithms or 'all' in algorithms:
        from training.pg_training import run_all_reinforce_experiments
        print("\n[2/4] REINFORCE Training...")
        rf_results, rf_df = run_all_reinforce_experiments(base_dir=BASE_DIR, zone=zone)
    
    if 'PPO' in algorithms or 'all' in algorithms:
        from training.pg_training import run_all_ppo_experiments
        print("\n[3/4] PPO Training...")
        ppo_results, ppo_df = run_all_ppo_experiments(timesteps=timesteps, base_dir=BASE_DIR, zone=zone)
    
    if 'A2C' in algorithms or 'all' in algorithms:
        from training.pg_training import run_all_a2c_experiments
        print("\n[4/4] A2C Training...")
        a2c_results, a2c_df = run_all_a2c_experiments(timesteps=timesteps, base_dir=BASE_DIR, zone=zone)
    
    # Generate comparison plots
    from training.pg_training import generate_all_comparison_plots
    generate_all_comparison_plots(base_dir=BASE_DIR)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)


def export_json_api(zone: str = 'savanna') -> dict:
    """
    Export environment state as JSON for frontend/API integration.
    Demonstrates how the RL agent can be serialized for web/mobile apps.
    """
    env = NigeriaFarmEnv(zone=zone)
    obs, info = env.reset(seed=42)
    
    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    api_data = env.to_json()
    api_data['api_info'] = {
        'endpoint': '/api/v1/nigeria-farm/state',
        'method': 'GET',
        'description': 'Returns current farm state for frontend visualization',
        'integration': {
            'web': 'React/Vue.js + D3.js for visualization',
            'mobile': 'Flutter/React Native with JSON state parsing',
            'realtime': 'WebSocket for live agent updates'
        }
    }
    
    print("\nJSON API Output (for web/mobile integration):")
    print(json.dumps(api_data, indent=2))
    
    env.close()
    return api_data


def run_random_agent_demo(zone: str = 'savanna', num_episodes: int = 2):
    """Run random agent demo (no trained model - just visualization)."""
    try:
        from environment.rendering import run_random_agent_demo as _demo
        _demo(num_episodes=num_episodes, max_steps=30)
    except ImportError as e:
        print(f"Pygame not available: {e}")
        print("Running terminal-mode random agent demo...")
        
        env = NigeriaFarmEnv(zone=zone, render_mode='ansi')
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep)
            total_reward = 0
            done = False
            step = 0
            print(f"\nEpisode {ep+1} | Zone: {zone}")
            while not done and step < 30:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                print(f"  Step {step:2d} | Action: {ACTION_NAMES[action]:<15} | "
                      f"Shock: {info['climate_shock']:<15} | Reward: {reward:+.2f}")
            print(f"  Total Reward: {total_reward:.2f}")
        env.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nigeria Farm Climate-RL | Ayomide Agbaje | ALU ML II',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                              # Run best agent (PPO)
  python main.py --algorithm DQN             # Run DQN agent
  python main.py --algorithm PPO --zone sahel # PPO in Sahel zone
  python main.py --train                      # Train all algorithms
  python main.py --train --algorithm PPO      # Train PPO only
  python main.py --demo                       # Random agent demo
  python main.py --json-api                   # Export JSON for API
        '''
    )
    
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'DQN', 'A2C', 'REINFORCE', 'all'],
                       help='RL algorithm to use (default: PPO)')
    parser.add_argument('--zone', type=str, default='savanna',
                       choices=['savanna', 'rainforest', 'sahel'],
                       help='Nigerian agricultural zone (default: savanna)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of evaluation episodes (default: 3)')
    parser.add_argument('--timesteps', type=int, default=25000,
                       help='Training timesteps per config (default: 25000)')
    parser.add_argument('--train', action='store_true',
                       help='Run training pipeline instead of evaluation')
    parser.add_argument('--demo', action='store_true',
                       help='Run random agent demo (visualization without model)')
    parser.add_argument('--json-api', action='store_true',
                       help='Export environment state as JSON for API integration')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable Pygame visualization (terminal mode only)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    
    args = parser.parse_args()
    
    if args.train:
        # Training mode
        algs = ['DQN', 'REINFORCE', 'PPO', 'A2C'] if args.algorithm == 'all' else [args.algorithm]
        run_training_pipeline(algorithms=algs, timesteps=args.timesteps, zone=args.zone)
    
    elif args.demo:
        # Random agent demo
        run_random_agent_demo(zone=args.zone, num_episodes=2)
    
    elif args.json_api:
        # JSON API export
        export_json_api(zone=args.zone)
    
    else:
        # Run best trained agent
        summary = run_best_agent(
            algorithm=args.algorithm,
            zone=args.zone,
            num_episodes=args.episodes,
            render=not args.no_render,
            verbose=not args.quiet
        )
        
        # Save summary
        summary_path = os.path.join(BASE_DIR, 'results', 'agent_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
