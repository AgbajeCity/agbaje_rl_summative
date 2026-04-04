"""
custom_env.py - NigeriaFarmEnv: Custom Gymnasium Environment
Mission: Protect Nigerian Smallholder Farmers from Climate Change
Author: Ayomide Agbaje | ALU Machine Learning Techniques II

Environment models a Nigerian smallholder farm exposed to climate shocks
(droughts, floods, heat waves, pest outbreaks) with realistic agricultural dynamics.
The RL agent learns optimal farm management decisions to maximize crop yield
and food security while protecting against climate-related losses.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


# =============================================================================
# CLIMATE SHOCKS - Real Nigerian Agricultural Threats
# =============================================================================
CLIMATE_SHOCKS = {
    'none':          {'prob': 0.45, 'yield_impact': 0.00, 'severity': 0},
    'drought':       {'prob': 0.15, 'yield_impact': -0.35, 'severity': 3},
    'flood':         {'prob': 0.10, 'yield_impact': -0.45, 'severity': 4},
    'heat_wave':     {'prob': 0.10, 'yield_impact': -0.25, 'severity': 2},
    'pest_outbreak': {'prob': 0.10, 'yield_impact': -0.30, 'severity': 3},
    'heavy_rain':    {'prob': 0.05, 'yield_impact': -0.10, 'severity': 1},
    'soil_erosion':  {'prob': 0.05, 'yield_impact': -0.20, 'severity': 2},
}

# Nigerian Agricultural Zones
ZONES = {
    'savanna': {
        'drought_risk': 0.30,
        'flood_risk': 0.05,
        'base_yield': 2.5,
        'crops': ['maize', 'millet', 'sorghum', 'cowpea']
    },
    'rainforest': {
        'drought_risk': 0.10,
        'flood_risk': 0.20,
        'base_yield': 3.2,
        'crops': ['cassava', 'yam', 'plantain', 'cocoa']
    },
    'sahel': {
        'drought_risk': 0.45,
        'flood_risk': 0.03,
        'base_yield': 1.8,
        'crops': ['millet', 'sorghum', 'groundnut']
    }
}

# Action mapping
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Apply Irrigation",
    2: "Apply Fertilizer",
    3: "Apply Pesticide",
    4: "Plant Cover Crops",
    5: "Build Drainage",
    6: "Install Shade Nets",
    7: "Harvest Early",
    8: "Request Aid",
    9: "Diversify Crops"
}


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================
class NigeriaFarmEnv(gym.Env):
    """
    Custom Gymnasium Environment: Nigerian Smallholder Farm Climate Resilience
    
    Mission: Train an RL agent to protect Nigerian farmers from climate shocks
    by learning optimal agricultural management strategies.
    
    Observation Space (16 features):
    -----------------------------------------------------------------------
    0  temperature         - Current temperature (Celsius) [20, 45]
    1  rainfall            - Monthly rainfall (mm) [0, 400]
    2  soil_moisture       - Soil moisture level [0, 1]
    3  crop_health         - Crop health index [0, 1]
    4  yield_forecast      - Expected yield (tons/ha) [0, 5]
    5  drought_risk        - Drought risk probability [0, 1]
    6  flood_risk          - Flood risk probability [0, 1]
    7  pest_risk           - Pest outbreak probability [0, 1]
    8  heat_risk           - Heat stress probability [0, 1]
    9  farmer_resources    - Available resources [0, 1]
    10 food_security_score - Household food security [0, 1]
    11 market_price        - Crop market price index [0.5, 2.0]
    12 early_warning       - Climate early warning signal [0, 1]
    13 season_progress     - Growing season progress [0, 1]
    14 cumulative_yield    - Season yield accumulated [0, 10]
    15 climate_stress      - Composite climate stress [0, 1]
    
    Action Space (10 discrete actions):
    -----------------------------------------------------------------------
    0  Do Nothing        - No intervention (saves resources)
    1  Apply Irrigation  - Combat drought, boost soil moisture
    2  Apply Fertilizer  - Improve crop health and yield potential
    3  Apply Pesticide   - Combat pest outbreaks
    4  Plant Cover Crops - Long-term soil health, erosion prevention
    5  Build Drainage    - Flood mitigation infrastructure
    6  Install Shade Nets- Heat wave protection
    7  Harvest Early     - Emergency harvest before major shock
    8  Request Aid       - External support (government/NGO)
    9  Diversify Crops   - Risk spreading across multiple crops
    
    Reward Structure:
    -----------------------------------------------------------------------
    Positive: yield gained, food security maintained, early warning acted on
    Negative: crop loss, resource waste, missed warnings, food insecurity
    Terminal: crop failure (<10% health) OR season complete (52 weeks)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}
    
    def __init__(self, zone: str = 'savanna', render_mode: Optional[str] = None):
        super().__init__()
        
        assert zone in ZONES, f"Zone must be one of {list(ZONES.keys())}"
        self.zone = zone
        self.zone_config = ZONES[zone]
        self.render_mode = render_mode
        
        # --- Observation Space (16 continuous features) ---
        low = np.array([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([45, 400, 1, 1, 5, 1, 1, 1, 1, 1, 1, 2.0, 1, 1, 10, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # --- Action Space (10 discrete actions) ---
        self.action_space = spaces.Discrete(10)
        
        # State variables
        self.state = None
        self.step_count = 0
        self.max_steps = 52  # One agricultural year (52 weeks)
        self.cumulative_reward = 0.0
        self.episode_history = []
        self.current_shock = 'none'
        
        # Rendering
        self._renderer = None
        
    def _get_initial_state(self) -> np.ndarray:
        """Initialize state based on zone characteristics."""
        zone = self.zone_config
        
        base_temp = 30 + np.random.normal(0, 3)
        base_rainfall = 80 + np.random.normal(0, 20)
        
        state = np.array([
            np.clip(base_temp, 20, 45),              # 0: temperature
            np.clip(base_rainfall, 0, 400),           # 1: rainfall
            np.random.uniform(0.4, 0.8),              # 2: soil_moisture
            np.random.uniform(0.7, 1.0),              # 3: crop_health
            zone['base_yield'] * np.random.uniform(0.8, 1.2),  # 4: yield_forecast
            zone['drought_risk'] * np.random.uniform(0.5, 1.5), # 5: drought_risk
            zone['flood_risk'] * np.random.uniform(0.5, 1.5),   # 6: flood_risk
            np.random.uniform(0.05, 0.20),            # 7: pest_risk
            np.random.uniform(0.05, 0.25),            # 8: heat_risk
            np.random.uniform(0.5, 1.0),              # 9: farmer_resources
            np.random.uniform(0.6, 1.0),              # 10: food_security_score
            np.random.uniform(0.8, 1.5),              # 11: market_price
            0.0,                                      # 12: early_warning (no warning at start)
            0.0,                                      # 13: season_progress
            0.0,                                      # 14: cumulative_yield
            np.random.uniform(0.1, 0.3),              # 15: climate_stress
        ], dtype=np.float32)
        
        return np.clip(state, self.observation_space.low, self.observation_space.high)
    
    def _sample_climate_shock(self) -> str:
        """Sample a climate shock based on zone risks and season."""
        shocks = list(CLIMATE_SHOCKS.keys())
        probs = [CLIMATE_SHOCKS[s]['prob'] for s in shocks]
        
        # Adjust probabilities based on zone
        drought_idx = shocks.index('drought')
        flood_idx = shocks.index('flood')
        probs[drought_idx] *= (self.zone_config['drought_risk'] / 0.30)
        probs[flood_idx] *= (self.zone_config['flood_risk'] / 0.10)
        
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        return np.random.choice(shocks, p=probs)
    
    def _apply_action(self, action: int) -> Tuple[float, str]:
        """Apply action and return reward delta and feedback message."""
        s = self.state.copy()
        reward = 0.0
        msg = ""
        
        resources_available = s[9] > 0.1
        
        if action == 0:  # Do Nothing
            reward = -0.5 if s[12] > 0.7 else 0.5  # Penalize ignoring warnings
            msg = "Waited out conditions"
            
        elif action == 1:  # Irrigation
            if resources_available:
                s[2] = min(1.0, s[2] + 0.25)
                s[3] = min(1.0, s[3] + 0.10)
                s[9] -= 0.10
                reward = 3.0 if self.current_shock == 'drought' else 1.0
                msg = "Irrigation applied"
            else:
                reward = -1.0
                msg = "Insufficient resources for irrigation"
                
        elif action == 2:  # Fertilizer
            if resources_available:
                s[4] = min(5.0, s[4] + 0.5)
                s[3] = min(1.0, s[3] + 0.08)
                s[9] -= 0.08
                reward = 2.0
                msg = "Fertilizer applied"
            else:
                reward = -1.0
                msg = "Insufficient resources for fertilizer"
                
        elif action == 3:  # Pesticide
            if resources_available:
                s[7] = max(0.0, s[7] - 0.4)
                s[9] -= 0.08
                reward = 4.0 if self.current_shock == 'pest_outbreak' else 0.5
                msg = "Pesticide applied"
            else:
                reward = -1.0
                msg = "Insufficient resources for pesticide"
                
        elif action == 4:  # Cover Crops
            if resources_available:
                s[2] = min(1.0, s[2] + 0.10)
                s[15] = max(0.0, s[15] - 0.05)
                s[9] -= 0.05
                reward = 1.5
                msg = "Cover crops planted"
            else:
                reward = -0.5
                msg = "Insufficient resources for cover crops"
                
        elif action == 5:  # Drainage
            if resources_available:
                s[6] = max(0.0, s[6] - 0.3)
                s[2] = max(0.0, s[2] - 0.1)
                s[9] -= 0.12
                reward = 4.0 if self.current_shock == 'flood' else 0.5
                msg = "Drainage system built"
            else:
                reward = -1.0
                msg = "Insufficient resources for drainage"
                
        elif action == 6:  # Shade Nets
            if resources_available:
                s[8] = max(0.0, s[8] - 0.3)
                s[3] = min(1.0, s[3] + 0.08)
                s[9] -= 0.10
                reward = 3.5 if self.current_shock == 'heat_wave' else 0.5
                msg = "Shade nets installed"
            else:
                reward = -1.0
                msg = "Insufficient resources for shade nets"
                
        elif action == 7:  # Early Harvest
            harvest_yield = s[4] * s[3] * 0.7  # Partial harvest
            s[14] += harvest_yield
            s[3] = max(0.0, s[3] - 0.3)
            reward = harvest_yield * 2.0 if s[12] > 0.5 else harvest_yield * 0.5
            msg = f"Early harvest: {harvest_yield:.2f} tons/ha"
            
        elif action == 8:  # Request Aid
            if s[10] < 0.3:  # Only beneficial when food insecure
                s[9] = min(1.0, s[9] + 0.25)
                s[10] = min(1.0, s[10] + 0.2)
                reward = 3.0
                msg = "Aid received successfully"
            else:
                reward = -0.5  # Penalty for requesting when not needed
                msg = "Aid not warranted - resources sufficient"
                
        elif action == 9:  # Diversify Crops
            if resources_available:
                s[5] = max(0.0, s[5] - 0.1)
                s[6] = max(0.0, s[6] - 0.1)
                s[7] = max(0.0, s[7] - 0.1)
                s[15] = max(0.0, s[15] - 0.08)
                s[9] -= 0.07
                reward = 2.0
                msg = "Crops diversified for resilience"
            else:
                reward = -0.5
                msg = "Insufficient resources for diversification"
        
        self.state = np.clip(s, self.observation_space.low, self.observation_space.high)
        return reward, msg
    
    def _apply_climate_dynamics(self) -> float:
        """Apply climate shock dynamics and return penalty/bonus."""
        self.current_shock = self._sample_climate_shock()
        shock_config = CLIMATE_SHOCKS[self.current_shock]
        
        s = self.state.copy()
        reward_delta = 0.0
        
        # Apply shock impact on crop health
        yield_impact = shock_config['yield_impact']
        s[3] = max(0.0, s[3] + yield_impact)
        s[4] = max(0.0, s[4] + yield_impact * 2)
        
        # Update climate risks
        if self.current_shock == 'drought':
            s[2] = max(0.0, s[2] - 0.2)
            s[5] = min(1.0, s[5] + 0.1)
        elif self.current_shock == 'flood':
            s[2] = min(1.0, s[2] + 0.3)
            s[6] = min(1.0, s[6] + 0.15)
        elif self.current_shock == 'heat_wave':
            s[0] = min(45, s[0] + 3.0)
            s[8] = min(1.0, s[8] + 0.2)
        elif self.current_shock == 'pest_outbreak':
            s[7] = min(1.0, s[7] + 0.3)
        
        # Early warning signal based on risk accumulation
        composite_risk = max(s[5], s[6], s[7], s[8])
        s[12] = composite_risk
        s[15] = composite_risk * 0.7  # Climate stress
        
        # Update season progress
        s[13] = min(1.0, self.step_count / self.max_steps)
        
        # Natural resource recovery
        s[9] = min(1.0, s[9] + 0.02)
        
        # Yield accumulation if crop health is good
        if s[3] > 0.5:
            weekly_yield = s[4] * 0.02 * s[3]
            s[14] = min(10.0, s[14] + weekly_yield)
            reward_delta += weekly_yield * 0.5
        
        # Food security update
        s[10] = 0.4 * s[3] + 0.3 * (s[14] / 5.0) + 0.3 * s[9]
        s[10] = np.clip(s[10], 0, 1)
        
        self.state = np.clip(s, self.observation_space.low, self.observation_space.high)
        
        # Early warning bonus/penalty
        if self.current_shock != 'none':
            reward_delta -= shock_config['severity'] * 0.5
        
        return reward_delta
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to start of new episode."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.state = self._get_initial_state()
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.episode_history = []
        self.current_shock = 'none'
        
        info = {
            'zone': self.zone,
            'crops': self.zone_config['crops'],
            'initial_yield_forecast': float(self.state[4]),
        }
        
        return self.state.copy(), info
    
    def step(self, action: int):
        """Execute one step in the environment."""
        assert self.state is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self.step_count += 1
        
        # 1. Apply climate dynamics (shock + natural processes)
        climate_reward = self._apply_climate_dynamics()
        
        # 2. Apply agent action
        action_reward, action_msg = self._apply_action(action)
        
        # 3. Calculate composite reward
        s = self.state
        
        # Crop health bonus
        health_bonus = (s[3] - 0.5) * 4.0
        
        # Food security bonus
        food_bonus = (s[10] - 0.5) * 3.0
        
        # Early warning response bonus
        warning_response_bonus = 0.0
        if s[12] > 0.6:
            if action in [1, 2, 3, 4, 5, 6, 7, 9]:
                warning_response_bonus = 2.0
            elif action == 0:
                warning_response_bonus = -3.0  # Penalty for ignoring warnings
        
        # Composite reward
        total_reward = (
            climate_reward +
            action_reward +
            health_bonus * 0.3 +
            food_bonus * 0.2 +
            warning_response_bonus
        )
        
        self.cumulative_reward += total_reward
        
        # Record step
        step_record = {
            'step': self.step_count,
            'action': action,
            'action_name': ACTION_NAMES[action],
            'shock': self.current_shock,
            'crop_health': float(s[3]),
            'food_security': float(s[10]),
            'reward': total_reward,
            'message': action_msg
        }
        self.episode_history.append(step_record)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if s[3] < 0.10:  # Crop failure
            terminated = True
            total_reward -= 20.0
            reason = "crop_failure"
        elif s[9] < 0.05 and s[10] < 0.2:  # Resource depletion + food insecurity
            terminated = True
            total_reward -= 15.0
            reason = "resource_depletion"
        elif s[14] < 0.1 and self.step_count > 10:  # No yield after 10 weeks
            terminated = True
            total_reward -= 10.0
            reason = "negligence"
        elif self.step_count >= self.max_steps:
            truncated = True
            # End-of-season bonus
            total_reward += s[14] * 2.0 + s[10] * 3.0
            reason = "season_complete"
        else:
            reason = "ongoing"
        
        info = {
            'step': self.step_count,
            'action_name': ACTION_NAMES[action],
            'climate_shock': self.current_shock,
            'crop_health': float(s[3]),
            'food_security': float(s[10]),
            'cumulative_yield': float(s[14]),
            'cumulative_reward': self.cumulative_reward,
            'termination_reason': reason,
            'zone': self.zone
        }
        
        return self.state.copy(), float(total_reward), terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'ansi':
            s = self.state
            lines = [
                f"\n{'='*60}",
                f"NIGERIA FARM | Week {self.step_count} | Zone: {self.zone.upper()}",
                f"{'='*60}",
                f"Temp: {s[0]:.1f}°C  |  Rainfall: {s[1]:.1f}mm  |  Soil: {s[2]:.2f}",
                f"Crop Health: {s[3]:.2f}  |  Yield: {s[4]:.2f} t/ha",
                f"Drought Risk: {s[5]:.2f}  |  Flood Risk: {s[6]:.2f}",
                f"Pest Risk: {s[7]:.2f}  |  Heat Risk: {s[8]:.2f}",
                f"Resources: {s[9]:.2f}  |  Food Security: {s[10]:.2f}",
                f"Early Warning: {s[12]:.2f}  |  Season: {s[13]:.0%}",
                f"Cumulative Yield: {s[14]:.2f} t/ha",
                f"Climate Shock: {self.current_shock.upper()}",
                f"Cumulative Reward: {self.cumulative_reward:.2f}",
                f"{'='*60}"
            ]
            return '\n'.join(lines)
        
        elif self.render_mode == 'rgb_array':
            return self._render_rgb()
    
    def _render_rgb(self) -> np.ndarray:
        """Render environment as RGB array for video recording."""
        import pygame
        import pygame.surfarray as surfarray
        
        W, H = 800, 600
        if not hasattr(self, '_pygame_surface') or self._pygame_surface is None:
            pygame.init()
            self._pygame_surface = pygame.Surface((W, H))
        
        surf = self._pygame_surface
        surf.fill((15, 25, 35))  # Dark background
        
        # Basic render - will be enhanced by rendering.py
        font = pygame.font.Font(None, 24)
        s = self.state
        texts = [
            f"Week: {self.step_count}  Zone: {self.zone}",
            f"Crop Health: {s[3]:.2f}",
            f"Yield Forecast: {s[4]:.2f} t/ha",
            f"Shock: {self.current_shock}",
        ]
        for i, text in enumerate(texts):
            surf_text = font.render(text, True, (255, 255, 255))
            surf.blit(surf_text, (20, 20 + i * 30))
        
        return surfarray.array3d(surf)
    
    def to_json(self) -> dict:
        """Serialize environment state to JSON for API/frontend integration."""
        s = self.state if self.state is not None else np.zeros(16)
        return {
            "metadata": {
                "env_name": "NigeriaFarmEnv",
                "zone": self.zone,
                "version": "1.0",
                "author": "Ayomide Agbaje",
                "mission": "Protect Nigerian Smallholder Farmers from Climate Change"
            },
            "state": {
                "temperature": float(s[0]),
                "rainfall_mm": float(s[1]),
                "soil_moisture": float(s[2]),
                "crop_health": float(s[3]),
                "yield_forecast_tons_ha": float(s[4]),
                "drought_risk": float(s[5]),
                "flood_risk": float(s[6]),
                "pest_risk": float(s[7]),
                "heat_risk": float(s[8]),
                "farmer_resources": float(s[9]),
                "food_security_score": float(s[10]),
                "market_price_index": float(s[11]),
                "early_warning_level": float(s[12]),
                "season_progress": float(s[13]),
                "cumulative_yield": float(s[14]),
                "climate_stress_index": float(s[15])
            },
            "episode": {
                "step": self.step_count,
                "max_steps": self.max_steps,
                "current_shock": self.current_shock,
                "cumulative_reward": float(self.cumulative_reward),
                "history_length": len(self.episode_history)
            },
            "actions": {str(k): v for k, v in ACTION_NAMES.items()},
            "zone_info": {
                "name": self.zone,
                "drought_risk_base": self.zone_config['drought_risk'],
                "flood_risk_base": self.zone_config['flood_risk'],
                "base_yield_tons_ha": self.zone_config['base_yield'],
                "crops": self.zone_config['crops']
            }
        }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, '_pygame_surface') and self._pygame_surface is not None:
            import pygame
            pygame.quit()
            self._pygame_surface = None


# =============================================================================
# ENVIRONMENT REGISTRATION
# =============================================================================
def make_env(zone: str = 'savanna', render_mode: Optional[str] = None):
    """Factory function to create NigeriaFarmEnv instance."""
    return NigeriaFarmEnv(zone=zone, render_mode=render_mode)


# Register with gymnasium
try:
    gym.envs.registration.register(
        id='NigeriaFarm-v0',
        entry_point='environment.custom_env:NigeriaFarmEnv',
        max_episode_steps=52,
    )
except Exception:
    pass  # Already registered


if __name__ == '__main__':
    # Quick test
    env = NigeriaFarmEnv(zone='savanna', render_mode='ansi')
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Obs space: {env.observation_space}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.render())
        print(f"Action: {ACTION_NAMES[action]} | Reward: {reward:.2f}")
        if terminated or truncated:
            break
    
    env.close()
    print("\nJSON API Output:")
    import json
    print(json.dumps(env.to_json(), indent=2))
