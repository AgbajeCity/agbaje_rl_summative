"""
environment/__init__.py
NigeriaFarmEnv Package
Author: Ayomide Agbaje | ALU Machine Learning Techniques II
"""

from .custom_env import NigeriaFarmEnv, make_env, ACTION_NAMES, CLIMATE_SHOCKS, ZONES

__all__ = ['NigeriaFarmEnv', 'make_env', 'ACTION_NAMES', 'CLIMATE_SHOCKS', 'ZONES']

__version__ = '1.0.0'
__author__ = 'Ayomide Agbaje'
__description__ = 'Nigeria Farm Climate-RL Custom Gymnasium Environment'
