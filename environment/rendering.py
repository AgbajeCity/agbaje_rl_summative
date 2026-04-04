"""
rendering.py - Advanced Visualization for NigeriaFarmEnv
Mission: Visualize Nigerian Farm Climate-RL Agent in Action
Author: Ayomide Agbaje | ALU Machine Learning Techniques II

Uses Pygame for real-time 2D visualization of the RL agent
interacting with the Nigeria Farm climate environment.
"""

import pygame
import pygame.font
import pygame.draw
import numpy as np
import sys
import math
from typing import Optional


# =============================================================================
# COLOR PALETTE - Nigerian Flag + Climate Theme
# =============================================================================
COLORS = {
    'bg_dark':       (10, 20, 35),      # Dark night sky
    'bg_mid':        (20, 40, 60),      # Horizon
    'green_nigeria': (0, 128, 0),       # Nigerian flag green
    'white_nigeria': (255, 255, 255),   # Nigerian flag white
    'soil_brown':    (101, 67, 33),     # Nigerian soil
    'crop_green':    (34, 139, 34),     # Healthy crops
    'crop_yellow':   (255, 215, 0),     # Stressed crops
    'crop_red':      (178, 34, 34),     # Failing crops
    'water_blue':    (30, 144, 255),    # Water/irrigation
    'warning_red':   (255, 69, 0),      # Climate warning
    'warning_orange':(255, 165, 0),     # Moderate warning
    'text_white':    (240, 240, 240),   # Main text
    'text_gold':     (255, 215, 0),     # Highlight text
    'text_green':    (0, 200, 100),     # Positive text
    'text_red':      (255, 100, 100),   # Negative text
    'panel_bg':      (15, 30, 50),      # Panel background
    'panel_border':  (0, 128, 0),       # Panel border (Nigerian green)
    'progress_green':(0, 205, 102),     # Progress bar
    'progress_bg':   (40, 60, 80),      # Progress bar background
    'sun_yellow':    (255, 200, 0),     # Sun
    'cloud_white':   (200, 210, 220),   # Cloud
    'rain_blue':     (0, 100, 255),     # Rain drops
    'fire_orange':   (255, 140, 0),     # Heat/drought
}

# Action emoji/icons
ACTION_SYMBOLS = {
    0: "⏸",   # Do Nothing
    1: "💧",  # Irrigation
    2: "🌱",  # Fertilizer
    3: "🔬",  # Pesticide
    4: "🌾",  # Cover Crops
    5: "🏗",  # Drainage
    6: "⛱",  # Shade Nets
    7: "🌽",  # Early Harvest
    8: "🆘",  # Request Aid
    9: "🔄",  # Diversify Crops
}

ACTION_LABELS = {
    0: "Do Nothing",
    1: "Irrigation",
    2: "Fertilize",
    3: "Pesticide",
    4: "Cover Crops",
    5: "Drainage",
    6: "Shade Nets",
    7: "Early Harvest",
    8: "Request Aid",
    9: "Diversify",
}


# =============================================================================
# MAIN RENDERER CLASS
# =============================================================================
class NigeriaFarmRenderer:
    """
    Advanced 2D Pygame visualization for NigeriaFarmEnv.
    
    Features:
    - Real-time farm visualization with weather effects
    - Climate shock indicators with visual alerts
    - Progress bars for all key metrics
    - Action history panel
    - Reward tracking chart
    - Agent decision display
    
    Can be integrated into web/mobile apps via:
    - Screenshot export to PNG/JPEG
    - Video recording via pygame surface
    - JSON state export for API frontend
    """
    
    def __init__(self, width: int = 1200, height: int = 750, caption: str = "Nigeria Farm Climate-RL"):
        self.width = width
        self.height = height
        self.caption = caption
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_xl = pygame.font.SysFont('Arial', 28, bold=True)
        self.font_lg = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_md = pygame.font.SysFont('Arial', 17)
        self.font_sm = pygame.font.SysFont('Arial', 14)
        self.font_xs = pygame.font.SysFont('Arial', 12)
        
        # State tracking
        self.reward_history = []
        self.action_history = []
        self.shock_history = []
        self.frame_count = 0
        self.animation_offset = 0
        
        # Particle effects
        self.particles = []
        
    def _draw_background(self, state: np.ndarray):
        """Draw dynamic background based on weather."""
        temp = state[0]
        rainfall = state[1]
        
        # Sky gradient based on weather
        if temp > 38:
            sky_color = (80, 30, 10)  # Hot hazy sky
        elif rainfall > 100:
            sky_color = (40, 50, 80)  # Cloudy/rainy sky
        else:
            sky_color = (20, 60, 120)  # Clear sky
        
        self.screen.fill(sky_color)
        
        # Ground / farm base
        pygame.draw.rect(self.screen, COLORS['soil_brown'],
                        (0, self.height // 2, self.width, self.height // 2))
    
    def _draw_farm_field(self, state: np.ndarray):
        """Draw the farm field with crop health visualization."""
        crop_health = state[3]
        soil_moisture = state[2]
        
        # Determine crop color based on health
        if crop_health > 0.7:
            crop_color = COLORS['crop_green']
        elif crop_health > 0.4:
            crop_color = COLORS['crop_yellow']
        else:
            crop_color = COLORS['crop_red']
        
        # Draw farm rows
        field_x, field_y = 50, self.height // 2 - 20
        field_w, field_h = 500, 200
        
        # Soil
        pygame.draw.rect(self.screen, COLORS['soil_brown'],
                        (field_x, field_y, field_w, field_h), border_radius=5)
        
        # Crop plants (rows of animated plants)
        num_rows = 8
        num_cols = 15
        for row in range(num_rows):
            for col in range(num_cols):
                x = field_x + 20 + col * (field_w - 40) // num_cols
                y = field_y + 20 + row * (field_h - 40) // num_rows
                
                # Animate plant sway
                sway = math.sin(self.frame_count * 0.05 + col * 0.3) * 3
                
                # Plant stem
                stem_h = int(15 * crop_health + 5)
                pygame.draw.line(self.screen, (0, 100, 0),
                               (x + sway, y + stem_h),
                               (x + sway, y), 2)
                
                # Plant top (leaf)
                leaf_size = int(6 * crop_health + 2)
                pygame.draw.circle(self.screen, crop_color,
                                 (int(x + sway), y), leaf_size)
        
        # Water puddles if soil moisture is high
        if soil_moisture > 0.7:
            for i in range(3):
                px = field_x + 50 + i * 150
                py = field_y + field_h - 20
                pygame.draw.ellipse(self.screen, COLORS['water_blue'],
                                   (px, py, 40, 12))
        
        # Irrigation channel
        pygame.draw.rect(self.screen, COLORS['water_blue'],
                        (field_x, field_y + field_h, field_w, 5))
        
        return field_x, field_y, field_w, field_h
    
    def _draw_weather_effects(self, state: np.ndarray, shock: str):
        """Draw weather and climate shock effects."""
        temp = state[0]
        rainfall = state[1]
        
        # Sun
        sun_intensity = min(255, int(temp * 5))
        sun_color = (sun_intensity, int(sun_intensity * 0.8), 0)
        pygame.draw.circle(self.screen, sun_color, (1100, 80), 50)
        # Sun rays
        for angle in range(0, 360, 30):
            rad = math.radians(angle + self.frame_count)
            x1 = 1100 + int(55 * math.cos(rad))
            y1 = 80 + int(55 * math.sin(rad))
            x2 = 1100 + int(70 * math.cos(rad))
            y2 = 80 + int(70 * math.sin(rad))
            pygame.draw.line(self.screen, sun_color, (x1, y1), (x2, y2), 2)
        
        # Rain drops
        if shock == 'flood' or rainfall > 150:
            for i in range(0, 600, 20):
                offset = (self.frame_count * 5 + i * 7) % 300
                pygame.draw.line(self.screen, COLORS['rain_blue'],
                               (i + offset % 30, 50 + offset),
                               (i + offset % 30 + 3, 65 + offset), 2)
        
        # Heat shimmer for heat wave
        if shock == 'heat_wave' or temp > 40:
            for i in range(10):
                x = np.random.randint(50, 600)
                y = self.height // 2 - 10
                shimmer_surf = pygame.Surface((60, 5), pygame.SRCALPHA)
                shimmer_surf.fill((255, 100, 0, 50))
                self.screen.blit(shimmer_surf, (x, y))
        
        # Pest indicators
        if shock == 'pest_outbreak':
            for i in range(5):
                px = 100 + i * 80 + (self.frame_count * 2) % 50
                py = self.height // 2 - 30 + i * 10
                pygame.draw.circle(self.screen, (150, 100, 0), (px, py), 5)
                pygame.draw.circle(self.screen, (80, 60, 0), (px, py), 3)
        
        # Drought - crack patterns in soil
        if shock == 'drought' or state[2] < 0.2:
            for i in range(8):
                x1 = 80 + i * 60
                y1 = self.height // 2 + 10
                pygame.draw.line(self.screen, (80, 50, 20),
                               (x1, y1), (x1 + 20, y1 + 15), 2)
                pygame.draw.line(self.screen, (80, 50, 20),
                               (x1 + 10, y1 + 7), (x1 + 5, y1 + 18), 2)
    
    def _draw_info_panels(self, state: np.ndarray, step: int, action: int,
                          shock: str, reward: float, cum_reward: float, zone: str):
        """Draw information panels with all metrics."""
        
        # === LEFT PANEL - Farm Status ===
        panel_rect = pygame.Rect(620, 10, 280, 400)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['panel_border'], panel_rect, 2, border_radius=8)
        
        # Title
        title = self.font_lg.render(f"FARM STATUS | {zone.upper()}", True, COLORS['text_gold'])
        self.screen.blit(title, (630, 18))
        
        # Week indicator
        week_text = self.font_md.render(f"Week {step} / 52", True, COLORS['text_white'])
        self.screen.blit(week_text, (630, 45))
        
        # Metrics with progress bars
        metrics = [
            ("Crop Health", state[3], COLORS['crop_green']),
            ("Soil Moisture", state[2], COLORS['water_blue']),
            ("Food Security", state[10], COLORS['text_green']),
            ("Resources", state[9], COLORS['text_gold']),
            ("Drought Risk", state[5], COLORS['warning_orange']),
            ("Flood Risk", state[6], COLORS['water_blue']),
            ("Pest Risk", state[7], COLORS['warning_red']),
            ("Heat Risk", state[8], COLORS['fire_orange'] if 'fire_orange' in COLORS else COLORS['warning_orange']),
        ]
        
        COLORS['fire_orange'] = (255, 140, 0)
        
        y_offset = 75
        for label, value, color in metrics:
            # Label
            lbl = self.font_sm.render(f"{label}:", True, COLORS['text_white'])
            self.screen.blit(lbl, (630, y_offset))
            
            # Progress bar
            bar_x, bar_y = 750, y_offset + 3
            bar_w, bar_h = 120, 12
            pygame.draw.rect(self.screen, COLORS['progress_bg'], (bar_x, bar_y, bar_w, bar_h))
            fill_w = int(bar_w * max(0, min(1, value)))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))
            
            # Value
            val_text = self.font_xs.render(f"{value:.2f}", True, COLORS['text_white'])
            self.screen.blit(val_text, (bar_x + bar_w + 5, y_offset))
            
            y_offset += 38
        
        # === CLIMATE METRICS PANEL ===
        panel_rect2 = pygame.Rect(620, 420, 280, 150)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], panel_rect2, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['panel_border'], panel_rect2, 2, border_radius=8)
        
        clim_title = self.font_md.render("CLIMATE METRICS", True, COLORS['text_gold'])
        self.screen.blit(clim_title, (630, 428))
        
        metrics2 = [
            (f"Temp: {state[0]:.1f}°C", COLORS['warning_orange'] if state[0] > 38 else COLORS['text_white']),
            (f"Rainfall: {state[1]:.0f}mm", COLORS['water_blue']),
            (f"Yield Forecast: {state[4]:.2f} t/ha", COLORS['text_green']),
            (f"Market Price: {state[11]:.2f}", COLORS['text_gold']),
        ]
        
        for i, (text, color) in enumerate(metrics2):
            surf = self.font_sm.render(text, True, color)
            self.screen.blit(surf, (630, 450 + i * 28))
        
        # === RIGHT PANEL - Agent & Reward ===
        panel_rect3 = pygame.Rect(910, 10, 280, 260)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], panel_rect3, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['panel_border'], panel_rect3, 2, border_radius=8)
        
        agent_title = self.font_lg.render("AGENT DECISION", True, COLORS['text_gold'])
        self.screen.blit(agent_title, (920, 18))
        
        # Current action
        action_name = ACTION_LABELS.get(action, f"Action {action}")
        action_surf = self.font_md.render(f"Action: {action_name}", True, COLORS['text_white'])
        self.screen.blit(action_surf, (920, 48))
        
        # Current reward
        r_color = COLORS['text_green'] if reward >= 0 else COLORS['text_red']
        reward_surf = self.font_lg.render(f"Reward: {reward:+.2f}", True, r_color)
        self.screen.blit(reward_surf, (920, 78))
        
        # Cumulative reward
        cr_color = COLORS['text_green'] if cum_reward >= 0 else COLORS['text_red']
        cum_surf = self.font_md.render(f"Total: {cum_reward:+.1f}", True, cr_color)
        self.screen.blit(cum_surf, (920, 108))
        
        # Current shock
        if shock != 'none':
            shock_color = COLORS['warning_red']
            shock_surf = self.font_lg.render(f"⚠ {shock.upper().replace('_', ' ')}", True, shock_color)
            self.screen.blit(shock_surf, (920, 140))
        
        # Early warning
        ew = state[12]
        if ew > 0.5:
            ew_text = f"WARNING LEVEL: {ew:.2f}"
            ew_color = COLORS['warning_red'] if ew > 0.7 else COLORS['warning_orange']
            ew_surf = self.font_md.render(ew_text, True, ew_color)
            self.screen.blit(ew_surf, (920, 170))
        
        # Cumulative yield
        yield_surf = self.font_md.render(f"Season Yield: {state[14]:.2f} t/ha", True, COLORS['text_green'])
        self.screen.blit(yield_surf, (920, 200))
        
        # Season progress bar
        sp_text = self.font_sm.render(f"Season Progress:", True, COLORS['text_white'])
        self.screen.blit(sp_text, (920, 228))
        pygame.draw.rect(self.screen, COLORS['progress_bg'], (920, 248, 250, 12))
        sp_fill = int(250 * state[13])
        pygame.draw.rect(self.screen, COLORS['progress_green'], (920, 248, sp_fill, 12))
        
        # === REWARD HISTORY CHART ===
        self.reward_history.append(reward)
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)
        
        chart_rect = pygame.Rect(910, 280, 280, 150)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], chart_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['panel_border'], chart_rect, 2, border_radius=8)
        
        chart_title = self.font_sm.render("Reward History", True, COLORS['text_gold'])
        self.screen.blit(chart_title, (920, 288))
        
        if len(self.reward_history) > 1:
            max_r = max(abs(r) for r in self.reward_history) or 1
            points = []
            for i, r in enumerate(self.reward_history):
                x = 915 + int(i * 270 / 50)
                y = 400 - int(r / max_r * 50)
                y = max(295, min(415, y))
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS['text_green'], False, points, 2)
        
        # Zero line
        pygame.draw.line(self.screen, COLORS['progress_bg'], (915, 400), (1185, 400), 1)
        
        # === ACTION HISTORY ===
        action_panel = pygame.Rect(910, 440, 280, 280)
        pygame.draw.rect(self.screen, COLORS['panel_bg'], action_panel, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['panel_border'], action_panel, 2, border_radius=8)
        
        ah_title = self.font_sm.render("Recent Actions", True, COLORS['text_gold'])
        self.screen.blit(ah_title, (920, 448))
        
        self.action_history.append((action, shock, reward))
        if len(self.action_history) > 8:
            self.action_history.pop(0)
        
        for i, (a, sh, r) in enumerate(reversed(self.action_history)):
            r_color = COLORS['text_green'] if r >= 0 else COLORS['text_red']
            a_text = f"{ACTION_LABELS.get(a, str(a)[:10])} | {r:+.1f}"
            if sh != 'none':
                a_text += f" | {sh[:8]}"
            txt = self.font_xs.render(a_text, True, r_color)
            self.screen.blit(txt, (920, 465 + i * 28))
    
    def _draw_header(self, step: int):
        """Draw the mission header."""
        # Header background
        pygame.draw.rect(self.screen, (0, 60, 20), (0, 0, 620, 55))
        
        title = self.font_xl.render("🌍 NIGERIA FARM CLIMATE-RL", True, COLORS['text_gold'])
        self.screen.blit(title, (10, 5))
        
        subtitle = self.font_sm.render(
            "Protecting Nigerian Smallholder Farmers from Climate Shocks | Ayomide Agbaje | ALU ML II",
            True, COLORS['text_white'])
        self.screen.blit(subtitle, (10, 36))
    
    def render_frame(self, state: np.ndarray, step: int, action: int,
                     shock: str, reward: float, cum_reward: float, 
                     zone: str = 'savanna') -> Optional[np.ndarray]:
        """
        Render one frame of the environment.
        
        Returns:
            np.ndarray: RGB array of the rendered frame (for video recording)
        """
        self.frame_count += 1
        self.animation_offset = (self.animation_offset + 1) % 360
        
        # Draw all components
        self._draw_background(state)
        self._draw_farm_field(state)
        self._draw_weather_effects(state, shock)
        self._draw_header(step)
        self._draw_info_panels(state, step, action, shock, reward, cum_reward, zone)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS for human viewing
        
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    sys.exit()
        
        # Return RGB array for video recording
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


# =============================================================================
# RANDOM AGENT DEMO (Static visualization without training)
# =============================================================================
def run_random_agent_demo(num_episodes: int = 3, max_steps: int = 30):
    """
    Run random agent in NigeriaFarmEnv with full visualization.
    This demonstrates the environment WITHOUT any trained model.
    
    Used for: Assignment requirement - 'Create a static file that shows 
    the agent taking random actions in the custom environment'
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from environment.custom_env import NigeriaFarmEnv
    
    renderer = NigeriaFarmRenderer(width=1200, height=750,
                                    caption="Nigeria Farm RL - Random Agent Demo")
    
    zones = ['savanna', 'rainforest', 'sahel']
    
    for episode in range(num_episodes):
        zone = zones[episode % len(zones)]
        env = NigeriaFarmEnv(zone=zone, render_mode='human')
        obs, info = env.reset(seed=episode * 42)
        
        total_reward = 0.0
        print(f"\nEpisode {episode+1} | Zone: {zone.upper()}")
        print("="*50)
        
        for step in range(max_steps):
            # Random action (no model)
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render frame
            renderer.render_frame(
                state=obs,
                step=step + 1,
                action=action,
                shock=info['climate_shock'],
                reward=reward,
                cum_reward=total_reward,
                zone=zone
            )
            
            print(f"  Step {step+1:2d} | Action: {info['action_name']:<15} | "
                  f"Shock: {info['climate_shock']:<15} | Reward: {reward:+6.2f}")
            
            if terminated or truncated:
                print(f"  Episode ended: {info['termination_reason']}")
                break
        
        print(f"  Total Reward: {total_reward:.2f}")
        env.close()
        pygame.time.wait(2000)
    
    renderer.close()
    print("\nDemo complete!")


if __name__ == '__main__':
    run_random_agent_demo(num_episodes=2, max_steps=20)
