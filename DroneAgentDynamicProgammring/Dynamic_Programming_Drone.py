'''
model a grid environment in open ai gym with the 
following considerations. 
Nature of Agent - Drone Agent

State Space - 
a) Position - 6 x 6 grid with 
              2 cells marked for charging station 
              4 cells marked danger zone 
              3 cells marked rescue target 
              2 cells marked windzone 
              3 cells marked as blocked cell 
              Top left most corner as start position 
              Other unmarked cells as freezones 
b) Battery Level max of 10 units with level depleting as
per reward system due to navigation or reaching charging 
station etc 
c) Status of rescued target - Rescued / Not Rescued 

Action space - Move up, down, left, right Hover
               The drone can only move within the grid and cannot move into blocked cells.
               - If an action would move the drone into a blocked cell, 
               the drone remains in its current position and consumes 1 battery unit and a
               reward of -1 for the attempted move.
               - Wind cells introduce stochastic transitions.
                 If the drone is currently on a wind cell (W) 
                 and attempts a movement action: there is a 
                 probability that the movement direction changes 
                 randomly.
                 If wind disturbance occurs: the actual movement 
                 direction must be selected uniformly from:            
                 Up, Down, Left, Right (excluding Hover).

Reward System - 
a) Rescue target reached - +20 reward
b) Charging station reached - +5 reward
c) Enter danger zone - -10 reward
d) Battery Exhausted - -20 reward
e) Regular Movement - -1 reward

Episode Termination
An episode ends when:
● Battery becomes zero
● All rescue targets are rescued
● Maximum step limit exceeded: 75 for 6×6 grids.
'''
import numpy as np
import random
class DroneEnv:
    def __init__(self):
        self.grid_size = 6
        self.battery_capacity = 10
        self.max_steps = 75
        
        # Define the grid with special cells
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.grid[0, 0] = 1  # Start position
        self.grid[1, 1] = 2  # Charging station
        self.grid[4, 4] = 2  # Charging station
        self.grid[2, 2] = 3  # Rescue target
        self.grid[3, 3] = 3  # Rescue target
        self.grid[5, 5] = 3  # Rescue target
        self.grid[1, 3] = -3 # Danger zone
        self.grid[4, 1] = -3 # Danger zone
        self.grid[2, 4] = -3 # Danger zone
        self.grid[0, 4] = -3 # Danger zone
        self.grid[3, 0] = -2 # Blocked cell
        self.grid[5, 0] = -2 # Blocked cell
        self.grid[0, 5] = -2 # Blocked cell
        # Wind cells
        self.grid[1, 0] = -1 # Wind cell
        self.grid[4, 3] = -1 # Wind cell
        
        self.reset()
    
    def reset(self):
        self.position = (0, 0)  # Start position
        self.battery_level = self.battery_capacity
        self.rescued_targets = set()
        self.steps_taken = 0
        return self._get_state()
    
    def _get_state(self):
        return (self.position, self.battery_level, len(self.rescued_targets))
    
    def step(self, action):
        if action not in ['up', 'down', 'left', 'right', 'hover']:
            raise ValueError("Invalid action")
        
        x, y = self.position
        
        if action == 'up':
            new_position = (x - 1, y)
        elif action == 'down':
            new_position = (x + 1, y)
        elif action == 'left':
            new_position = (x, y - 1)
        elif action == 'right':
            new_position = (x, y + 1)
        else: # hover
            new_position = (x, y)
        # Check for boundaries and blocked cells
        if (0 <= new_position[0] < self.grid_size and
            0 <= new_position[1] < self.grid_size and
            self.grid[new_position] != -2):
            self.position = new_position
        else:
            # If move is invalid, stay in place and consume battery
            self.battery_level -= 1
            return self._get_state(), -1, self._check_done()
        # Check for wind disturbance
        if self.grid[self.position] == -1:  # Wind cell
            if random.random() <= 0.3:  # 30% chance of wind disturbance
                action = random.choice(['up', 'down', 'left', 'right'])
                return self.step(action)  # Re-attempt move with new action
        # Update battery level        
        self.battery_level -= 1
        # Calculate reward
        reward = -1  # Default movement penalty
        cell_value = self.grid[self.position]
        if cell_value == 2:  # Charging station
            reward = 5
            self.battery_level = self.battery_capacity  # Recharge battery
        elif cell_value == 3:  # Rescue target
            if self.position not in self.rescued_targets:
                reward = 20
                self.rescued_targets.add(self.position)
        elif cell_value == -3:  # Danger zone
            reward = -10
        # Check for episode termination
        done = self._check_done()
        return self._get_state(), reward, done
    def _check_done(self):
        if self.battery_level <= 0:
            return True  # Battery exhausted
        if len(self.rescued_targets) == 3:
            return True  # All rescue targets rescued
        if self.steps_taken >= self.max_steps:
            return True  # Maximum step limit exceeded
        return False
# Example usage
env = DroneEnv()
state = env.reset()
done = False
while not done:
    action = random.choice(['up', 'down', 'left', 'right', 'hover'])  # Random action for testing
    state, reward, done = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")    
#Write code to render the grid environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def render_grid(env):
    fig, ax = plt.subplots()
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            cell_value = env.grid[i, j]
            if cell_value == 1:  # Start position
                color = 'lightblue'
            elif cell_value == 2:  # Charging station
                color = 'green'
            elif cell_value == 3:  # Rescue target
                color = 'yellow'
            elif cell_value == -3:  # Danger zone
                color = 'red'
            elif cell_value == -2:  # Blocked cell
                color = 'black'
            elif cell_value == -1:  # Wind cell
                color = 'cyan'
            else:  # Free zone
                color = 'white'
            rect = patches.Rectangle((j, env.grid_size - i - 1), 1, 1, edgecolor='gray', facecolor=color)
            ax.add_patch(rect)
    # Draw the drone's current position
    drone_x, drone_y = env.position
    drone_rect = patches.Rectangle((drone_y, env.grid_size - drone_x - 1), 1, 1, edgecolor='blue', facecolor='blue')
    ax.add_patch(drone_rect)
    plt.xlim(0, env.grid_size)
    plt.ylim(0, env.grid_size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.arange(0, env.grid_size + 1))
    plt.yticks(np.arange(0, env.grid_size + 1))
    plt.grid(True)
    plt.show()
# Example usage of rendering
env = DroneEnv()
render_grid(env)

            