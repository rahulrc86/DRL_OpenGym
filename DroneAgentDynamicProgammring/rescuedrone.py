# ============================================================
# RESCUE DRONE RL ENVIRONMENT
#
# Custom Gymnasium Environment
#
# State Space:
#   Position (6x6 Grid)
#   Battery Level
#   Rescue Status of Targets
#
# Action Space:
#   Move Up
#   Move Down
#   Move Left
#   Move Right
#   Hover
#
# ============================================================

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import pygame
import numpy as np
import random
import time

from collections import namedtuple


# ============================================================
# REGISTER ENVIRONMENT
# ============================================================

try:
    register(
        id="RescueDrone-v0",
        entry_point="rescuedrone:RescueDroneEnv",
        max_episode_steps=75
    )
except:
    pass


# ============================================================
# ACTION DEFINITIONS
# ============================================================

ACTION_SELECTION = namedtuple(
    "ActionSelection",
    [
        "MOVE_UP",
        "MOVE_DOWN",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "HOVER"
    ]
)

RESCUE_STATUS = namedtuple(
    "RescueStatus",
    [
        "NOT_RESCUED",
        "RESCUED"
    ]
)
        
a_sel = ACTION_SELECTION(0, 1, 2, 3, 4)
r_status = RESCUE_STATUS(0, 1)


# ============================================================
# ENVIRONMENT CLASS
# ============================================================

class RescueDroneEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 5
    }

    def __init__(self, render_mode="human"):

        super().__init__()

        # ----------------------------------------------------
        # Environment Configuration
        # ----------------------------------------------------

        self.grid_rows = 6
        self.grid_cols = 6
        self.max_steps = 75
        self.MAX_BATTERY = 10
        self.START_BATTERY = 8
        self.render_mode = render_mode
        # ----------------------------------------------------
        # Fixed Environment Layout
        # ----------------------------------------------------

        self.start_position = (0, 0)
        self.charging_stations = [ (0, 5),(5, 0)]
        self.charging_stations = [ (1, 0),(5, 0)]

        self.danger_zones = [ (1, 1), (2, 4), (4, 2),\
                                      (5, 5) ]

        self.wind_zones = [ (0, 3), (3, 3)]

        self.blocked_cells = [  (2, 2),\
                                (3, 1),\
                                (4, 4) ]   

        self.targets = [  (1, 4),\
                          (4, 1),\
                          (5, 3) ]    
        
        self.targets = [  (0, 1),\
                          (4, 1),\
                          (5, 3) ] 

        # ----------------------------------------------------
        # Action Space
        # ----------------------------------------------------

        self.action_space = spaces.Discrete(5)

        # ----------------------------------------------------
        # Observation Space
        # ----------------------------------------------------

        self.observation_space = spaces.Dict({

            "position":
                spaces.MultiDiscrete(
                    [self.grid_rows,
                     self.grid_cols]
                ),

            "battery":
                spaces.Discrete(
                    self.MAX_BATTERY + 1
                ),

            "rescued":
                spaces.MultiBinary(3)

        })

        # ----------------------------------------------------
        # Runtime Variables
        # ----------------------------------------------------

        self.position = list(self.start_position)

        self.battery_level = self.START_BATTERY

        self.rescued_targets = [
            r_status.NOT_RESCUED,
            r_status.NOT_RESCUED,
            r_status.NOT_RESCUED
        ]

        self.step_count = 0

        # ----------------------------------------------------
        # Pygame Configuration
        # ----------------------------------------------------

        self.window_size = 720

        self.cell_size = (
            self.window_size // self.grid_rows
        )

        self.window = None
        self.clock = None

        if self.render_mode == "human":

            pygame.init()

            self.window = pygame.display.set_mode(
                (
                    self.window_size,
                    self.window_size
                )
            )

            pygame.display.set_caption(
                "Rescue Drone Environment"
            )

            self.clock = pygame.time.Clock()
   
        # ========================================================
    # OBSERVATION HELPER
    # ========================================================

    def get_observation(self):
        '''
        Create observation returned to the agent.

        Returns:
            Dictionary observation
        '''

        return {

            "position":
                np.array(
                    self.position,
                    dtype=np.int32
                ),

            "battery":
                self.battery_level,

            "rescued":
                np.array(
                    self.rescued_targets,
                    dtype=np.int8
                )
        }

    # ========================================================
    # CHARGING STATION CHECK
    # ========================================================

    def is_charging_station(
        self,
        position: tuple
    ) -> bool:

        return (
            position
            in
            self.charging_stations
        )

    # ========================================================
    # DANGER ZONE CHECK
    # ========================================================

    def is_danger_zone(
        self,
        position: tuple
    ) -> bool:

        return (
            position
            in
            self.danger_zones
        )

    # ========================================================
    # WIND ZONE CHECK
    # ========================================================

    def is_wind_zone(
        self,
        position: tuple
    ) -> bool:

        return (
            position
            in
            self.wind_zones
        )

    # ========================================================
    # BLOCKED CELL CHECK
    # ========================================================

    def is_blocked_cell(
        self,
        position: tuple
    ) -> bool:

        return (
            position
            in
            self.blocked_cells
        )

    # ========================================================
    # RESCUE TARGET CHECK
    # ========================================================

    def is_target(
        self,
        position: tuple
    ) -> bool:

        return (
            position
            in
            self.targets
        )

    # ========================================================
    # RECHARGE BATTERY
    # ========================================================

    def recharge_battery(self):

        self.battery_level = self.MAX_BATTERY

    # ========================================================
    # UPDATE RESCUE STATUS
    # ========================================================

    def update_rescue_status(self):
        '''
        Update rescue status if a target is reached.
        '''

        current_position = tuple(
            self.position
        )

        for idx, target in enumerate(
            self.targets
        ):

            if (
                current_position == target
                and
                self.rescued_targets[idx]
                ==
                r_status.NOT_RESCUED
            ):

                self.rescued_targets[idx] = (
                    r_status.RESCUED
                )

    # ========================================================
    # CHECK IF ALL TARGETS RESCUED
    # ========================================================

    def all_targets_rescued(self):

        return all(
            status == r_status.RESCUED
            for status
            in self.rescued_targets
        )

    # ========================================================
    # WIND DISTURBANCE MODEL
    # ========================================================

    def apply_wind_disturbance(
        self,
        action: int
    ) -> int:
        '''
        Wind cell introduces stochastic action.

        30% chance of random movement.

        Hover is unaffected.
        '''

        current_position = tuple(
            self.position
        )

        if not self.is_wind_zone(
            current_position
        ):
            return action

        if action == a_sel.HOVER:
            return action

        if np.random.random() < 0.30:

            action = random.choice(
                [
                    a_sel.MOVE_UP,
                    a_sel.MOVE_DOWN,
                    a_sel.MOVE_LEFT,
                    a_sel.MOVE_RIGHT
                ]
            )

        return action

    # ========================================================
    # COMPUTE NEXT POSITION
    # ========================================================

    def get_next_position(
        self,
        action: int
    ) -> tuple:
        '''
        Compute next position
        after action execution.
        '''

        row, col = self.position

        if action == a_sel.MOVE_UP:

            row -= 1

        elif action == a_sel.MOVE_DOWN:

            row += 1

        elif action == a_sel.MOVE_LEFT:

            col -= 1

        elif action == a_sel.MOVE_RIGHT:

            col += 1

        elif action == a_sel.HOVER:

            pass

        row = max(
            0,
            min(
                row,
                self.grid_rows - 1
            )
        )

        col = max(
            0,
            min(
                col,
                self.grid_cols - 1
            )
        )

        return (
            row,
            col
        )

    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.position = list(
            self.start_position
        )

        self.battery_level = (
            self.START_BATTERY
        )

        self.rescued_targets = [

            r_status.NOT_RESCUED,

            r_status.NOT_RESCUED,

            r_status.NOT_RESCUED
        ]

        self.step_count = 0

        observation = (
            self.get_observation()
        )

        info = {}

        if (
            self.render_mode
            ==
            "human"
        ):
            self.render()

        return (
            observation,
            info
        )

        # ========================================================
    # REWARD FUNCTION
    # ========================================================

    def get_reward(
        self,
        previous_rescue_status
    ) -> int:
        '''
        Calculate reward for the current state.

        Rewards:

        Rescue Target Reached     +20
        Charging Station Reached  +5
        Danger Zone Entered       -10
        Regular Movement          -1
        Battery Exhausted         -20
        '''

        reward = -1

        current_position = tuple(
            self.position
        )

        # ----------------------------------------------------
        # New Rescue Achieved
        # ----------------------------------------------------

        for idx in range(
            len(self.rescued_targets)
        ):

            if (
                previous_rescue_status[idx]
                ==
                r_status.NOT_RESCUED
                and
                self.rescued_targets[idx]
                ==
                r_status.RESCUED
            ):

                reward = 20
                return reward

        # ----------------------------------------------------
        # Charging Station
        # ----------------------------------------------------

        if self.is_charging_station(
            current_position
        ):

            reward = 5

        # ----------------------------------------------------
        # Danger Zone
        # ----------------------------------------------------

        if self.is_danger_zone(
            current_position
        ):

            reward = -10

        return reward

    # ========================================================
    # TERMINATION CHECK
    # ========================================================

    def check_termination(
        self
    ) -> bool:
        '''
        Episode terminates if:

        1) Battery empty
        2) All targets rescued
        '''

        battery_empty = (
            self.battery_level <= 0
        )

        all_rescued = (
            self.all_targets_rescued()
        )

        return (
            battery_empty
            or
            all_rescued
        )

    # ========================================================
    # STEP FUNCTION
    # ========================================================

    def step(
        self,
        action: int
    ):
        '''
        Execute one environment step.
        '''

        # ----------------------------------------------------
        # Increment Step Counter
        # ----------------------------------------------------

        self.step_count += 1

        # ----------------------------------------------------
        # Store Rescue Status
        # ----------------------------------------------------

        previous_rescue_status = (
            self.rescued_targets.copy()
        )

        # ----------------------------------------------------
        # Apply Wind Disturbance
        # ----------------------------------------------------

        action = (
            self.apply_wind_disturbance(
                action
            )
        )

        # ----------------------------------------------------
        # Battery Consumption
        # ----------------------------------------------------

        self.battery_level -= 1

        # ----------------------------------------------------
        # Compute Candidate Position
        # ----------------------------------------------------

        candidate_position = (
            self.get_next_position(
                action
            )
        )

        # ----------------------------------------------------
        # Blocked Cell Logic
        # ----------------------------------------------------

        if not self.is_blocked_cell(
            candidate_position
        ):

            self.position = list(
                candidate_position
            )

        # ----------------------------------------------------
        # Charging Logic
        # ----------------------------------------------------

        if self.is_charging_station(
            tuple(self.position)
        ):

            self.recharge_battery()

        # ----------------------------------------------------
        # Rescue Logic
        # ----------------------------------------------------

        self.update_rescue_status()

        # ----------------------------------------------------
        # Reward Computation
        # ----------------------------------------------------

        reward = self.get_reward(
            previous_rescue_status
        )

        # ----------------------------------------------------
        # Battery Exhaustion Penalty
        # ----------------------------------------------------

        if self.battery_level <= 0:

            reward += -20

        # ----------------------------------------------------
        # Episode Termination
        # ----------------------------------------------------

        terminated = (
            self.check_termination()
        )

        # ----------------------------------------------------
        # Episode Truncation
        # ----------------------------------------------------

        truncated = (
            self.step_count
            >=
            self.max_steps
        )

        # ----------------------------------------------------
        # Observation
        # ----------------------------------------------------

        observation = (
            self.get_observation()
        )

        info = {

            "battery":
                self.battery_level,

            "step_count":
                self.step_count,

            "rescued_targets":
                self.rescued_targets
        }

        # ----------------------------------------------------
        # Rendering
        # ----------------------------------------------------

        if (
            self.render_mode
            ==
            "human"
        ):

            self.render()

        # ----------------------------------------------------
        # Return Gymnasium Tuple
        # ----------------------------------------------------

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    
        # ========================================================
    # PYGAME RENDERING
    # ========================================================

    def render(self):
        if self.window is None:
            return

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                return

        self.window.fill((255, 255, 255))

        font = pygame.font.SysFont(
            "Segoe UI Emoji",
            40
        )

        for row in range(self.grid_rows):

            for col in range(self.grid_cols):

                x = col * self.cell_size
                y = row * self.cell_size

                cell_position = (row, col)

                # ------------------------------------------------
                # Default Free Zone
                # ------------------------------------------------

                cell_color = (200, 200, 200)
                icon_text = ""

                # ------------------------------------------------
                # Charging Station
                # ------------------------------------------------

                if cell_position in self.charging_stations:

                    cell_color = (0, 200, 0)
                    icon_text = "⚡"

                # ------------------------------------------------
                # Danger Zone
                # ------------------------------------------------

                elif cell_position in self.danger_zones:

                    cell_color = (220, 0, 0)
                    icon_text = "🔥"

                # ------------------------------------------------
                # Wind Zone
                # ------------------------------------------------

                elif cell_position in self.wind_zones:

                    cell_color = (255, 255, 0)
                    icon_text = "☁"

                # ------------------------------------------------
                # Blocked Cell
                # ------------------------------------------------

                elif cell_position in self.blocked_cells:

                    cell_color = (0, 0, 0)
                    icon_text = "⬛"

                # ------------------------------------------------
                # Rescue Target
                # ------------------------------------------------

                elif cell_position in self.targets:

                    target_index = self.targets.index(
                        cell_position
                    )

                    if (
                        self.rescued_targets[target_index]
                        ==
                        r_status.NOT_RESCUED
                    ):

                        cell_color = (0, 100, 255)
                        icon_text = "✋"

                    else:

                        cell_color = (200, 200, 200)
                        icon_text = ""

                # ------------------------------------------------
                # Start Cell
                # ------------------------------------------------

                if cell_position == self.start_position:

                    cell_color = (180, 180, 180)

                    if icon_text == "":
                        icon_text = "S"

                # ------------------------------------------------
                # Draw Cell
                # ------------------------------------------------

                pygame.draw.rect(

                    self.window,

                    cell_color,

                    (
                        x,
                        y,
                        self.cell_size,
                        self.cell_size
                    )
                )

                pygame.draw.rect(

                    self.window,

                    (50, 50, 50),

                    (
                        x,
                        y,
                        self.cell_size,
                        self.cell_size
                    ),

                    2
                )

                # ------------------------------------------------
                # Draw Icon
                # ------------------------------------------------

                if icon_text != "":

                    icon_surface = font.render(

                        icon_text,

                        True,

                        (255, 255, 255)
                    )

                    icon_rect = icon_surface.get_rect(

                        center=(

                            x + self.cell_size // 2,

                            y + self.cell_size // 2
                        )
                    )

                    self.window.blit(
                        icon_surface,
                        icon_rect
                    )

        # --------------------------------------------------------
        # Draw Drone ON TOP of Cell
        # --------------------------------------------------------

        drone_row, drone_col = self.position

        drone_x = (
            drone_col * self.cell_size
            +
            self.cell_size // 2
        )

        drone_y = (
            drone_row * self.cell_size
            +
            self.cell_size // 2
        )

        drone_surface = font.render(
            "🚁",
            True,
            (255, 140, 0)
        )

        drone_rect = drone_surface.get_rect(
            center=(drone_x, drone_y)
        )

        self.window.blit(
            drone_surface,
            drone_rect
        )

        # --------------------------------------------------------
        # Status Panel
        # --------------------------------------------------------

        info_font = pygame.font.SysFont(
            "Arial",
            24
        )

        status_text = (
            f"Battery:{self.battery_level}   "
            f"Step:{self.step_count}"
        )

        status_surface = info_font.render(
            status_text,
            True,
            (0, 0, 0)
        )

        self.window.blit(
            status_surface,
            (10, 10)
        )

        pygame.display.update()

        self.clock.tick(
            self.metadata["render_fps"]
        )


# ============================================================
# TEST SCRIPT
# ============================================================

if __name__ == "__main__":

    print(
        "RESCUE DRONE ENVIRONMENT CREATED"
    )

    env = gym.make(        "RescueDrone-v0",

        render_mode="human"
    )

    observation, info = env.reset()

    print(
        f"Initial Observation: "
        f"{observation}"
    )

    episode_over = False

    total_reward = 0

    while not episode_over:

        action = (
            env.action_space.sample()
        )

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        if terminated:

            print(
                "Episode Terminated"
            )

        if truncated:

            print(
                "Episode Truncated"
            )

        episode_over = (
            terminated
            or
            truncated
        )

        time.sleep(1)

    print(
        f"Total Reward: "
        f"{total_reward}"
    )
    print(
        f"Final Observation: "
        f"{observation}"
    )
    

    env.close()