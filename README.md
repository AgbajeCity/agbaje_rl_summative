# agbaje_rl_summative - Nigeria Farm Climate-RL

**Author:** Ayomide Agbaje | ALU Machine Learning Techniques II
**Repository:** https://github.com/AgbajeCity/agbaje_rl_summative

## Mission
Tackle the threat of climate change to African food systems by engineering predictive intelligence tools and early warning systems using RL to protect Nigerian smallholder farmers from climate-related crop losses.

## Project Structure
```
agbaje_rl_summative/
├── environment/
│   ├── custom_env.py      # NigeriaFarmEnv - Custom Gymnasium Environment
│   └── rendering.py       # Advanced Pygame 2D Visualization
├── training/
│   ├── dqn_training.py    # DQN - 10 hyperparameter experiments
│   └── pg_training.py     # REINFORCE, PPO, A2C - 10 runs each
├── models/                # Saved model checkpoints
├── results/               # Training results CSVs and JSON
├── plots/                 # Generated analysis plots
├── main.py                # Entry point: run best model simulation
├── requirements.txt       # Dependencies
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
python main.py --algorithm PPO --zone 1 --steps 120
```

## Environment Details
- **Observation Space:** Box(16,) - 16 continuous features (soil, climate, crop state)
- - **Action Space:** Discrete(10) - 10 adaptive farming actions
  - - **Zones:** 3 Nigerian agro-ecological zones
    - - **Climate Shocks:** Drought, Flood, Heat Wave, Pest Outbreak
      - - **Max Episode:** 120 days (one growing season)
       
        - ## Algorithms
        - - DQN (Value-Based) - 10 hyperparameter runs
          - - REINFORCE (Monte Carlo PG) - 10 runs
            - - PPO (Proximal Policy Optimization) - 10 runs
              - - A2C (Advantage Actor-Critic) - 10 runs
               
                - ## License
                - MIT License - Ayomide Agbaje, 2026
