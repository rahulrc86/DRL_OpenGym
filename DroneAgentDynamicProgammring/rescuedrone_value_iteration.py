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
import matplotlib.pyplot as plt


# ============================================================
# REGISTER ENVIRONMENT
# ============================================================

try:
    register(
        id="RescueDrone-v0",
        entry_point="rescuedrone_value_iteration:RescueDroneEnv",
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
        self.START_BATTERY = 4
        self.START_BATTERY = 5
        self.render_mode = render_mode
        # ----------------------------------------------------
        # Fixed Environment Layout
        # ----------------------------------------------------

        self.start_position = (0, 0)
        self.charging_stations = [ (0, 5),(5, 0)]
        self.charging_stations = [ (0, 1),(4, 0)]

        self.danger_zones = [ (1, 1), (2, 4), (4, 2),\
                                      (5, 5) ]

        self.wind_zones = [ (0, 3), (3, 3)]

        self.blocked_cells = [  (2, 2),\
                                (3, 1),\
                                (4, 4) ]   

        self.targets = [  (1, 4),\
                          (4, 1),\
                          (5, 3) ]    
        
        self.targets = [  (2, 0),\
                          (2, 1),\
                          (5, 2) ] 

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
# VALUE ITERATION PLANNER
# ============================================================

class RescueDroneValueIteration:
    """
    Value Iteration Planner

    State Representation:

    (
        row,
        col,
        battery_level,
        rescue_1,
        rescue_2,
        rescue_3
    )

    Example:

    (
        2,
        4,
        5,
        1,
        0,
        1
    )
    """

    def __init__(
        self,
        env
    ):

        self.env = env.unwrapped

        self.gamma = 0.95

        self.theta = 1e-3

        self.states = []

        self.V = {}

        self.policy = {}

        self.enumerate_states()

    # ========================================================
    # STATE ENUMERATION
    # ========================================================

    def enumerate_states(self):

        self.states = []

        for row in range(
            self.env.grid_rows
        ):

            for col in range(
                self.env.grid_cols
            ):

                if (
                    row,
                    col
                ) in self.env.blocked_cells:

                    continue

                for battery in range(
                    self.env.MAX_BATTERY + 1
                ):

                    for rescue_mask in range(8):

                        rescue_state = (

                            (rescue_mask >> 0) & 1,

                            (rescue_mask >> 1) & 1,

                            (rescue_mask >> 2) & 1
                        )

                        state = (

                            row,
                            col,
                            battery,

                            rescue_state[0],
                            rescue_state[1],
                            rescue_state[2]
                        )

                        self.states.append(
                            state
                        )

                        self.V[state] = 0.0
        for state in self.states:

            print("state is ", state)

    # ========================================================
    # TERMINAL STATE CHECK
    # ========================================================

    def is_terminal_state(
        self,
        state
    ):

        battery = state[2]

        rescued = state[3:]

        battery_empty = (
            battery <= 0
        )

        all_rescued = (
            rescued[0] == 1
            and
            rescued[1] == 1
            and
            rescued[2] == 1
        )

        return (
            battery_empty
            or
            all_rescued
        )

    # ========================================================
    # ACTION TO DELTA
    # ========================================================

    def action_to_delta(
        self,
        action
    ):

        if action == a_sel.MOVE_UP:

            return (-1, 0)

        elif action == a_sel.MOVE_DOWN:

            return (1, 0)

        elif action == a_sel.MOVE_LEFT:

            return (0, -1)

        elif action == a_sel.MOVE_RIGHT:

            return (0, 1)

        return (0, 0)

    # ========================================================
    # APPLY MOVEMENT
    # ========================================================

    def move_state(
        self,
        state,
        action
    ):

        row, col = state[0], state[1]

        dr, dc = self.action_to_delta(
            action
        )

        new_row = row + dr

        new_col = col + dc

        new_row = max(
            0,
            min(
                new_row,
                self.env.grid_rows - 1
            )
        )

        new_col = max(
            0,
            min(
                new_col,
                self.env.grid_cols - 1
            )
        )

        if (
            new_row,
            new_col
        ) in self.env.blocked_cells:

            return (
                row,
                col
            )

        return (
            new_row,
            new_col
        )

    # ========================================================
    # STATE TRANSITION
    # ========================================================

    def get_next_state(
        self,
        state,
        action
    ):

        row = state[0]
        col = state[1]

        battery = state[2]

        rescued = list(
            state[3:]
        )

        next_row, next_col = (
            self.move_state(
                state,
                action
            )
        )

        battery = max(
            0,
            battery - 1
        )

        if (action == a_sel.HOVER and  (row, col) \
            in self.env.charging_stations):
            battery = min(self.env.MAX_BATTERY, battery + 2)

        # ----------------------------------
        # Charging Station
        # ----------------------------------

        if (
            next_row,
            next_col
        ) in self.env.charging_stations:

            battery = (
                self.env.MAX_BATTERY
            )

        # ----------------------------------
        # Rescue Targets
        # ----------------------------------

        for idx, target in enumerate(
            self.env.targets
        ):

            if (
                next_row,
                next_col
            ) == target:

                rescued[idx] = 1

        next_state = (

            next_row,
            next_col,

            battery,

            rescued[0],
            rescued[1],
            rescued[2]
        )

        return next_state

    # ========================================================
    # REWARD MODEL
    # ========================================================

    def get_transition_reward(
        self,
        current_state,
        next_state
    ):

        reward = -1

        next_position = (
            next_state[0],
            next_state[1]
        )

        # ----------------------------------
        # New Rescue
        # ----------------------------------

        current_rescue = current_state[3:]

        next_rescue = next_state[3:]

        for idx in range(3):

            if (
                current_rescue[idx] == 0
                and
                next_rescue[idx] == 1
            ):

                reward = 20

                return reward

        # ----------------------------------
        # Charging
        # ----------------------------------

        if (
            next_position
            in
            self.env.charging_stations
        ):

            reward = 5

        # ----------------------------------
        # Danger
        # ----------------------------------

        if (
            next_position
            in
            self.env.danger_zones
        ):

            reward = -10

        # ----------------------------------
        # Battery Exhausted
        # ----------------------------------

        if next_state[2] <= 0:

            reward -= 20

        return reward

    # ========================================================
    # TRANSITION MODEL
    # ========================================================

    def get_transition_model(
        self,
        state,
        action
    ):
        """
        Returns:

        [
            (
                probability,
                next_state,
                reward
            )
        ]
        """

        position = (
            state[0],
            state[1]
        )

        transitions = []

        # ----------------------------------
        # WIND ZONE
        # ----------------------------------

        if (
            position
            in
            self.env.wind_zones
            and
            action != a_sel.HOVER
        ):

            all_actions = [

                a_sel.MOVE_UP,

                a_sel.MOVE_DOWN,

                a_sel.MOVE_LEFT,

                a_sel.MOVE_RIGHT
            ]

            for candidate_action in all_actions:

                if candidate_action == action:

                    probability = 0.70

                else:

                    probability = 0.10

                next_state = (
                    self.get_next_state(
                        state,
                        candidate_action
                    )
                )

                reward = (
                    self.get_transition_reward(
                        state,
                        next_state
                    )
                )

                transitions.append(

                    (
                        probability,
                        next_state,
                        reward
                    )
                )

        else:

            next_state = (
                self.get_next_state(
                    state,
                    action
                )
            )

            reward = (
                self.get_transition_reward(
                    state,
                    next_state
                )
            )

            transitions.append(

                (
                    1.0,
                    next_state,
                    reward
                )
            )

        return transitions
    
        # ========================================================
    # COMPUTE ACTION VALUE
    # ========================================================

    def compute_q_value(
        self,
        state,
        action
    ):

        q_value = 0.0

        transitions = (
            self.get_transition_model(
                state,
                action
            )
        )

        for (
            probability,
            next_state,
            reward
        ) in transitions:

            q_value += (

                probability *

                (

                    reward

                    +

                    self.gamma *

                    self.V[next_state]
                )
            )

        return q_value

    # ========================================================
    # BELLMAN BACKUP
    # ========================================================

    def bellman_backup(
        self,
        state
    ):

        if self.is_terminal_state(
            state
        ):

            return 0.0

        action_values = []

        for action in self.get_valid_actions(state):

            q_value = (
                self.compute_q_value(
                    state,
                    action
                )
            )

            action_values.append(
                q_value
            )

        return max(
            action_values
        )

    # ========================================================
    # ASYNCHRONOUS VALUE ITERATION
    # ========================================================

    def run_value_iteration(
        self
    ):

        iteration = 0

        while True:

            delta = 0.0

            iteration += 1

            for state in self.states:

                old_value = (
                    self.V[state]
                )

                new_value = (
                    self.bellman_backup(
                        state
                    )
                )

                self.V[state] = (
                    new_value
                )

                delta = max(

                    delta,

                    abs(
                        old_value -
                        new_value
                    )
                )

            print(

                f"Iteration={iteration}  "

                f"Delta={delta:.6f}"
            )

            if delta < self.theta:

                print(
                    "\nValue Function Converged\n"
                )

                break

        self.extract_policy()

    # =======================================================
    # VALID ACTION SELECTION
    # =======================================================
    def get_valid_actions(self, state):

        row, col = state[0], state[1]

        valid_actions = [a_sel.HOVER]

        if row > 0:
            valid_actions.append(a_sel.MOVE_UP)

        if row < self.env.grid_rows - 1:
            valid_actions.append(a_sel.MOVE_DOWN)

        if col > 0:
            valid_actions.append(a_sel.MOVE_LEFT)

        if col < self.env.grid_cols - 1:
            valid_actions.append(a_sel.MOVE_RIGHT)

        return valid_actions

    # ========================================================
    # POLICY EXTRACTION
    # ========================================================

    def extract_policy(
        self
    ):

        print(
            "Extracting Optimal Policy ..."
        )

        self.policy = {}

        for state in self.states:

            if self.is_terminal_state(
                state
            ):

                continue

            best_action = None

            best_value = -999999

            for action in self.get_valid_actions(state):

                q_value = (
                    self.compute_q_value(
                        state,
                        action
                    )
                )

                if q_value > best_value:

                    best_value = q_value

                    best_action = action

            self.policy[state] = (
                best_action
            )

        print(
            "Policy Extraction Complete"
        )

    # ========================================================
    # ACTION ICON
    # ========================================================

    def get_action_icon(
        self,
        action
    ):

        if action == a_sel.MOVE_UP:

            return "↑"

        elif action == a_sel.MOVE_DOWN:

            return "↓"

        elif action == a_sel.MOVE_LEFT:

            return "←"

        elif action == a_sel.MOVE_RIGHT:

            return "→"

        elif action == a_sel.HOVER:

            return "↺"

        return ""

    # ========================================================
    # POLICY GRID VISUALIZATION
    # ========================================================

    def render_policy_grid(
        self,
        battery_level=4,
        rescue_status=(0,0,0)
    ):

        pygame.init()

        window_size = 720

        cell_size = (
            window_size //
            self.env.grid_rows
        )

        screen = pygame.display.set_mode(
            (
                window_size,
                window_size
            )
        )

        pygame.display.set_caption(
            "Optimal Policy"
        )

        font = pygame.font.SysFont(
            "Segoe UI Emoji",
            36
        )

        running = True

        while running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    running = False

            screen.fill(
                (255,255,255)
            )

            for row in range(
                self.env.grid_rows
            ):

                for col in range(
                    self.env.grid_cols
                ):

                    x = (
                        col *
                        cell_size
                    )

                    y = (
                        row *
                        cell_size
                    )

                    position = (
                        row,
                        col
                    )

                    color = (
                        200,
                        200,
                        200
                    )

                    icon = ""

                    if (
                        position
                        in
                        self.env.charging_stations
                    ):

                        color = (
                            0,
                            200,
                            0
                        )

                        icon = "⚡"

                    elif (
                        position
                        in
                        self.env.danger_zones
                    ):

                        color = (
                            255,
                            0,
                            0
                        )

                        icon = "🔥"

                    elif (
                        position
                        in
                        self.env.wind_zones
                    ):

                        color = (
                            255,
                            255,
                            0
                        )

                        icon = "☁"

                    elif (
                        position
                        in
                        self.env.blocked_cells
                    ):

                        color = (
                            0,
                            0,
                            0
                        )

                    elif (
                        position
                        in
                        self.env.targets
                    ):

                        color = (
                            0,
                            100,
                            255
                        )

                        icon = "✋"

                    pygame.draw.rect(

                        screen,

                        color,

                        (
                            x,
                            y,
                            cell_size,
                            cell_size
                        )
                    )

                    pygame.draw.rect(

                        screen,

                        (50,50,50),

                        (
                            x,
                            y,
                            cell_size,
                            cell_size
                        ),

                        2
                    )

                    state = (

                        row,
                        col,

                        battery_level,

                        rescue_status[0],
                        rescue_status[1],
                        rescue_status[2]
                    )

                    if (
                        state
                        in
                        self.policy
                    ):

                        action_icon = (

                            self.get_action_icon(

                                self.policy[
                                    state
                                ]
                            )
                        )

                        action_surface = (

                            font.render(

                                action_icon,

                                True,

                                (0,0,0)
                            )
                        )

                        action_rect = (

                            action_surface.get_rect(

                                center=(

                                    x +
                                    cell_size//2,

                                    y +
                                    cell_size//2
                                )
                            )
                        )

                        screen.blit(
                            action_surface,
                            action_rect
                        )

            pygame.display.update()

        pygame.quit()
    
    # ========================================================
# POLICY ACTION LOOKUP
# ========================================================

    def get_policy_action(
        self,
        state
    ):

        if state in self.policy:

            return self.policy[state]

        return a_sel.HOVER


    # ========================================================
    # POLICY EXECUTION DEMO
    # ========================================================

    def simulate_optimal_rescue(
        self,
        delay=0.75
    ):
        if not pygame.get_init():
            pygame.init()
        
        env = self.env

        observation, info = env.reset()

        total_reward = 0

        print("\nExecuting Optimal Policy\n")

        while True:

            state = (

                env.position[0],
                env.position[1],

                env.battery_level,

                env.rescued_targets[0],
                env.rescued_targets[1],
                env.rescued_targets[2]
            )

            action = self.get_policy_action(
                state
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            total_reward += reward

            print(
                f"Pos={tuple(env.position)} "
                f"Battery={env.battery_level} "
                f"Reward={reward} "
                f"Total={total_reward}"
            )

            pygame.display.set_caption(

                f"Optimal Rescue Mission | "

                f"Battery={env.battery_level} | "

                f"Reward={total_reward} | "

                f"Targets Rescued="
                f"{sum(env.rescued_targets)}/3"
            )

            time.sleep(delay)

            if terminated or truncated:

                break

        print(
            f"\nMission Complete "
            f"Total Reward={total_reward}"
        )

    # ========================================================
# STATE VALUE HEATMAP ANALYSIS
# ========================================================

    def plot_value_heatmap(
    self,
    battery_level=4,
    rescue_status=(0,0,0)
):
        """
        State Value Analysis

        Fixed:
            Battery Level
            Rescue Status

        Variable:
            Drone Position

        Visualizes:
            V*(s)
        """

        value_grid = np.full(

        (
            self.env.grid_rows,
            self.env.grid_cols
        ),

        np.nan
        )

        for row in range(
            self.env.grid_rows
        ):

            for col in range(
                self.env.grid_cols
            ):

                if (
                    row,
                    col
                ) in self.env.blocked_cells:

                    continue

                state = (

                    row,
                    col,

                    battery_level,

                    rescue_status[0],
                    rescue_status[1],
                    rescue_status[2]
                )

                if state in self.V:

                    value_grid[row][col] = (
                        self.V[state]
                    )

        plt.figure(
            figsize=(8,6)
        )

        heatmap = plt.imshow(
            value_grid,
            cmap="viridis"
        )

        plt.colorbar(
            heatmap,
            label="V*(s)"
        )

        plt.title(

            f"State Value Heatmap\n"

            f"Battery={battery_level}, "

            f"Rescue={rescue_status}"
        )

        # ----------------------------------------
        # Overlay Special Cells
        # ----------------------------------------

        for target in self.env.targets:

            plt.text(

                target[1],
                target[0],

                "R",

                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

        for station in self.env.charging_stations:

            plt.text(

                station[1],
                station[0],

                "C",

                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

        for danger in self.env.danger_zones:

            plt.text(

                danger[1],
                danger[0],

                "D",

                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

        for wind in self.env.wind_zones:

            plt.text(

                wind[1],
                wind[0],

                "W",

                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

        plt.xlabel("Column")
        plt.ylabel("Row")

        plt.tight_layout()

        plt.show()

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

    print("Start Value Iteration Planner ...")
    planner = RescueDroneValueIteration(env)

    planner.run_value_iteration()

    # planner.render_policy_grid(
    # battery_level=4,
    # rescue_status=(0,0,0)
    # )
    planner.plot_value_heatmap(
    battery_level=4,
    rescue_status=(0,0,0)
    )

    planner.plot_value_heatmap(
    battery_level=10,
    rescue_status=(0,0,0)
    )

    planner.plot_value_heatmap(
    battery_level=10,
    rescue_status=(1,1,0)
    )
    
    planner.simulate_optimal_rescue()
    env.close()