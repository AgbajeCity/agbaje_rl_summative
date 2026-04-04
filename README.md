# agbaje_rl_summative - Nigeria Farm Climate-RL

## Mission-Based Reinforcement Learning for Agricultural Climate Resilience

**Author:** Ayomide Agbaje | **Institution:** African Leadership University (ALU)  
**Course:** Machine Learning Techniques II | **Type:** Mission-Based RL Summative

---

## 🌍 Mission Statement

Tackle the threat of climate change to African food systems by engineering predictive intelligence tools, hyperlocal weather insights and early warning systems, using data science and machine learning skills to build digital defences that prevent crop losses and protect low-income smallholder farmers across Nigeria's vulnerable, underserved rural communities from climate-related shocks.

---

## 🏗️ Project Structure

```
agbaje_rl_summative/
├── environment/
│   ├── custom_env.py          # NigeriaFarmEnv - Custom Gymnasium environment (23KB)
│   └── rendering.py           # Advanced Pygame 2D visualization (21KB)
├── training/
│   ├── dqn_training.py        # DQN with 10 hyperparameter configurations
│   └── pg_training.py         # REINFORCE, PPO, A2C with 10 configs each
├── models/                    # Saved trained models (generated after training)
│   ├── dqn/
│   └── pg/
├── results/                   # Training results CSVs and plots
├── main.py                    # Entry point for running best agent
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🌾 Environment: NigeriaFarmEnv

### Observation Space (16 features)
| # | Feature | Range | Description |
|---|---------|--------|-------------|
| 0 | temperature | [20, 45] | Current temperature (°C) |
| 1 | rainfall | [0, 400] | Monthly rainfall (mm) |
| 2 | soil_moisture | [0, 1] | Soil moisture level |
| 3 | crop_health | [0, 1] | Crop health index |
| 4 | yield_forecast | [0, 5] | Expected yield (tons/ha) |
| 5 | drought_risk | [0, 1] | Drought probability |
| 6 | flood_risk | [0, 1] | Flood probability |
| 7 | pest_risk | [0, 1] | Pest outbreak probability |
| 8 | heat_risk | [0, 1] | Heat stress probability |
| 9 | farmer_resources | [0, 1] | Available resources |
| 10 | food_security_score | [0, 1] | Household food security |
| 11 | market_price | [0.5, 2.0] | Crop market price index |
| 12 | early_warning | [0, 1] | Climate early warning signal |
| 13 | season_progress | [0, 1] | Growing season progress |
| 14 | cumulative_yield | [0, 10] | Season yield accumulated |
| 15 | climate_stress | [0, 1] | Composite climate stress |

### Action Space (10 discrete actions)
| # | Action | Climate Effect |
|---|--------|---------------|
| 0 | Do Nothing | No intervention |
| 1 | Apply Irrigation | +Soil moisture, combats drought |
| 2 | Apply Fertilizer | +Crop health, +Yield |
| 3 | Apply Pesticide | Combats pest outbreaks |
| 4 | Plant Cover Crops | Long-term soil protection |
| 5 | Build Drainage | Flood mitigation |
| 6 | Install Shade Nets | Heat wave protection |
| 7 | Harvest Early | Emergency harvest before major shock |
| 8 | Request Aid | External support (NGO/Government) |
| 9 | Diversify Crops | Risk spreading across multiple crops |

### Agricultural Zones
- **Savanna** - Drought risk 30%, flood risk 5%, base yield 2.5 t/ha
- **Rainforest** - Drought risk 10%, flood risk 20%, base yield 3.2 t/ha
- **Sahel** - Drought risk 45%, flood risk 3%, base yield 1.8 t/ha

### Climate Shocks
- None (45%), Drought (15%), Flood (10%), Heat Wave (10%), Pest Outbreak (10%), Heavy Rain (5%), Soil Erosion (5%)

---

## 📊 Training Results

### DQN Results (Value-Based, 10 Configurations)
| Config | LR | Gamma | Batch | Explore | Mean Reward | Std |
|--------|-----|-------|-------|---------|-------------|-----|
| DQN_baseline | 0.001 | 0.99 | 64 | 0.30 | 321.52 | 129.09 |
| DQN_high_lr | 0.005 | 0.99 | 64 | 0.30 | 268.83 | 109.60 |
| DQN_low_lr | 0.0001 | 0.99 | 64 | 0.30 | 155.61 | 23.75 |
| DQN_low_gamma | 0.001 | 0.80 | 64 | 0.30 | 268.85 | 88.16 |
| DQN_large_batch | 0.001 | 0.99 | 256 | 0.30 | 288.44 | 69.06 |
| **DQN_small_batch** | **0.001** | **0.99** | **32** | **0.40** | **348.12** | 146.44 |
| DQN_high_explore | 0.001 | 0.99 | 64 | 0.60 | 302.60 | 152.77 |
| DQN_low_explore | 0.001 | 0.99 | 64 | 0.10 | 189.99 | 76.99 |
| DQN_large_buffer | 0.001 | 0.99 | 128 | 0.30 | 206.98 | 121.95 |
| DQN_optimal | 0.0003 | 0.99 | 128 | 0.35 | 92.71 | 48.36 |

**Best DQN: DQN_small_batch | Mean Reward: 348.12**

### REINFORCE Results (Policy Gradient, 10 Configurations)
| Config | LR | Gamma | Baseline | Mean Reward | Std |
|--------|-----|-------|---------|-------------|-----|
| RF_baseline | 0.001 | 0.99 | True | 151.26 | 34.39 |
| RF_high_lr | 0.005 | 0.99 | True | 69.97 | 55.68 |
| RF_low_lr | 0.0001 | 0.99 | True | -6.16 | 38.25 |
| RF_low_gamma | 0.001 | 0.85 | True | 91.40 | 30.07 |
| RF_mid_gamma | 0.001 | 0.95 | True | 70.68 | 88.87 |
| RF_no_baseline | 0.001 | 0.99 | False | 138.84 | 38.69 |
| RF_med_lr | 0.0003 | 0.99 | True | 67.01 | 71.39 |
| RF_5e4_lr | 0.0005 | 0.99 | True | 149.39 | 37.64 |
| **RF_optimal** | **0.001** | **0.99** | **True** | **163.30** | 38.30 |
| RF_tuned2 | 0.002 | 0.97 | True | 78.77 | 94.82 |

**Best REINFORCE: RF_optimal | Mean Reward: 163.30**

### PPO Results (Proximal Policy Optimization, 10 Configurations)
| Config | LR | n_steps | ent_coef | clip | Mean Reward | Std |
|--------|-----|---------|---------|------|-------------|-----|
| PPO_baseline | 0.0003 | 2048 | 0.01 | 0.2 | 142.04 | 36.36 |
| PPO_high_lr | 0.001 | 2048 | 0.01 | 0.2 | 150.94 | 23.07 |
| **PPO_low_lr** | **0.0001** | **2048** | **0.01** | **0.2** | **167.30** | 41.64 |
| PPO_small_steps | 0.0003 | 512 | 0.01 | 0.2 | 157.05 | 30.15 |
| PPO_large_steps | 0.0003 | 4096 | 0.01 | 0.2 | 145.32 | 25.80 |
| PPO_high_entropy | 0.0003 | 2048 | 0.05 | 0.2 | 117.20 | 42.72 |
| PPO_no_entropy | 0.0003 | 2048 | 0.0 | 0.2 | 159.09 | 38.17 |
| PPO_wide_clip | 0.0003 | 2048 | 0.01 | 0.3 | 149.99 | 35.34 |
| PPO_low_gamma | 0.0003 | 2048 | 0.01 | 0.2 | 138.39 | 51.05 |
| PPO_optimal | 0.0002 | 2048 | 0.005 | 0.25 | 137.88 | 34.92 |

**Best PPO: PPO_low_lr | Mean Reward: 167.30**

### A2C Results (Actor-Critic, 10 Configurations)
*A2C training in progress - results will be updated*

---

## 🔬 Algorithm Comparison

| Algorithm | Best Config | Mean Reward | Std | Stability |
|-----------|-------------|-------------|-----|-----------|
| **DQN** | DQN_small_batch | **348.12** | 146.44 | Moderate |
| REINFORCE | RF_optimal | 163.30 | 38.30 | Low (high variance) |
| PPO | PPO_low_lr | 167.30 | 41.64 | High |
| A2C | TBD | TBD | TBD | TBD |

**Key Findings:**
- DQN achieves highest mean reward (348.12) but with high variance
- PPO shows best stability (lowest std relative to mean)
- REINFORCE has highest variance due to Monte Carlo nature
- Small batch size + higher exploration benefits DQN significantly
- Low learning rate (0.0001) helps PPO converge stably

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best trained agent (PPO by default)
python main.py

# Run DQN agent in rainforest zone
python main.py --algorithm DQN --zone rainforest

# Train all algorithms
python main.py --train --algorithm all

# Random agent demo (no trained model)
python main.py --demo

# Export JSON state for API
python main.py --json-api
```

---

## 🌐 API Integration

The environment supports JSON serialization for web/mobile integration:

```python
env = NigeriaFarmEnv(zone='savanna')
state_json = env.to_json()
# Returns: {state, episode, actions, zone_info, metadata}
```

**Integration targets:**
- 🌐 **Web**: React/Vue.js + D3.js for real-time farm visualization
- 📱 **Mobile**: Flutter/React Native with JSON state parsing
- ⚡ **Real-time**: WebSocket for live agent updates

---

## 📄 License

MIT License | Ayomide Agbaje | ALU 2025
