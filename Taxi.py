# Import gymnasium (preferred) or fall back to gym if not installed
import gymnasium as gym
import time
import numpy as np
from collections import namedtuple


# Create a training environment - a cart with a pole that needs balancing
''' 
The Taxi is modelled as a Markov Decision Process such that the 

1) Action space A constitutes
    1. Discrete - {0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff}

2) State space / Observation space S constitutes
    1. Taxi row: 5 discrete values (0-4)        

    2. Taxi column: 5 discrete values (0-4)

    3. Passenger location: 5 discrete values (0-4) (4 means the passenger is in the taxi) 
        0: Red 
        1: Green 
        2: Yellow 
        3: Blue 
        4: In Taxi (passenger is in the taxi)

    4. Destination location: 4 discrete values (0-3)    
        0: Red 
        1: Green    
        2: Yellow
        3: Blue
'''
# print ("AGENT NOT TRAINED YET! - Random Action Selection")
env_random = gym.make("Taxi-v3", render_mode="human")
time.sleep(10)

# Start a new episode
observation,info = env_random.reset()
# cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
# print (type(cart_position), type(cart_velocity), type(pole_angle), type(pole_angular_velocity))

# First Observation
print(f"Start Observation, info : {observation}," f" {info}")

#Run a full episode. The action selection is random
episode_over = False
total_reward = 0
episode_reward_random = []

# while not episode_over:
#     #Choose an action: 0 - push cart left, 1 - push cart right
#     #Action Selection Stage
#     action = env_random.action_space.sample() # Random action. This is not an MDP model 
#     # Outcome of the action. New State, Reward, At Terminal Step, Truncated, additional information
#     observation, reward, terminated, truncated, info = env_random.step(action)
    
#     # Decode observation to extract state details
#     taxi_row, taxi_col, passenger_location, destination_location = env_random.unwrapped.decode(observation)
#     print(f"Passenger Location: {passenger_location}, Destination Location: {destination_location}")
#     print(f"Taxi Position: ({taxi_row}, {taxi_col})")
#     print(f"Observation decoded: taxi_row={taxi_row}, taxi_col={taxi_col}, passenger_loc={passenger_location}, dest_loc={destination_location}")
#     if truncated:
#         print("Episode truncated due to time limit.")
#     if terminated:
#         print("Episode terminated successfully.")
#     #reward +1 for each step the pole stays upright
#     #Terminated / At Terminal Step : True if pole falls too far
#     #Truncated : True if the time limit is hit
#     total_reward += reward
#     episode_over = terminated or truncated
#     #Introduce a small delay to make the rendering visible (optional)
#     time.sleep(0.1)
#     episode_reward_random.append(reward)
# print(f"Episode finished! Total reward: {total_reward}")
# print(f"Rewards per step: {episode_reward_random}")
env_random.close()



# print("QLEARNING TRAINED AGENT CREATED! - Action Selection based on Q-Values")

# from collections import defaultdict
# import numpy as np

MOVE_SOUTH = 0
MOVE_NORTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
PICKUP = 4
DROPOFF = 5


PASSENGER_LOCATION = namedtuple('PassengerLocation', ['RED', 'GREEN', 'YELLOW', 'BLUE', 'IN_TAXI'])
p_loc = PASSENGER_LOCATION(0, 1, 2, 3, 4)

DESTINATION_LOCATION = namedtuple('DestinationLocation', ['RED', 'GREEN', 'YELLOW', 'BLUE'])
d_loc = DESTINATION_LOCATION(0, 1, 2, 3)

ACTION_SELECTION = namedtuple('ActionSelection', ['MOVE_SOUTH', 'MOVE_NORTH', 'MOVE_EAST', 'MOVE_WEST', 'PICKUP', 'DROPOFF'])
a_sel = ACTION_SELECTION(0, 1, 2, 3, 4, 5)
# env_rl_agent = gym.make("CartPole-v1", render_mode="human")

# def heuristic_action(obs:tuple[np.float32,np.float32,np.float32,np.float32]) -> int:
#         '''
#         A smarter heuristic policy that uses pole angle plus angular velocity.
#         Args:
#             obs: The current observation (state)
#         Returns:
#             action: The heuristic action (0 or 1)
#         '''
#         cart_position, cart_velocity, pole_angle, pole_angular_velocity = obs

#         # Base decision: push in the direction the pole is leaning.
#         # Add a small angular velocity correction to account for pole motion.
#         velocity_correction = 0.5 * pole_angular_velocity
#         control_signal = pole_angle + velocity_correction
#         return MOVE_RIGHT if control_signal > 0 else MOVE_LEFT

# # Start a new episode
# observation,info = env_rl_agent.reset()
# cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
# print (type(cart_position), type(cart_velocity), type(pole_angle), type(pole_angular_velocity))

# # First Observation
# print(f"Start Observation: {observation}")

# #Run a full episode. The action selection is random
# episode_over = False
# total_reward = 0
# episode_reward_heuristic = []

# while not episode_over:
#     #Choose an action: 0 - push cart left, 1 - push cart right
#     #Action Selection Stage
#     action = heuristic_action(observation) # Smarter heuristic action selection based on pole angle and angular velocity. This is not an MDP model
#     # Outcome of the action. New State, Reward, At Terminal Step, Truncated, additional information
#     observation, reward, terminated, truncated, info = env_rl_agent.step(action)
#     #reward +1 for each step the pole stays upright
#     #Terminated / At Terminal Step : True if pole falls too far
#     #Truncated : True if the time limit is hit
#     total_reward += reward
#     episode_over = terminated or truncated
#     episode_reward_heuristic.append(reward)
#     #Introduce a small delay to make the rendering visible (optional)
#     time.sleep(0)
# print(f"Episode finished! Total reward: {total_reward}")
# print(f"Rewards per step: {episode_reward_heuristic}")
# env_rl_agent.close()

# import matplotlib.pyplot as plt
# import numpy as np

# # Example: replace these with your collected lists
# # episode_reward_random = [ ... ]
# # episode_reward_heuristic = [ ... ]

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_rewards_separate(random_rewards, heuristic_rewards):
#     episodes_r = np.arange(1, len(random_rewards) + 1)
#     episodes_h = np.arange(1, len(heuristic_rewards) + 1)

#     plt.figure(figsize=(7,4))
#     plt.plot(episodes_r, random_rewards, '-o', color='C0')
#     plt.title('Random Policy: Episode Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(7,4))
#     plt.plot(episodes_h, heuristic_rewards, '-o', color='C1')
#     plt.title('Heuristic Policy: Episode Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Usage:
# plot_rewards_separate(episode_reward_random, episode_reward_heuristic)
# #     episodes_r = np.arange(1, len(random_rewards) + 1)
# #     episodes_h = np.arange(1, len(random_rewards) + 1)

# #     plt.figure(figsize=(10,5))
# #     plt.plot(episodes_r, random_rewards, '-o', label='Random Policy')
# #     plt.plot(episodes_h, heuristic_rewards, '-o', label='Heuristic Policy')
# #     plt.xlabel('Episode')
# #     plt.ylabel('Total Reward')
# #     plt.title('Episode Rewards: Random vs Heuristic')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.show()

# #     print(f'Random: mean={np.mean(random_rewards):.2f}, std={np.std(random_rewards):.2f}')
# #     print(f'Heuristic: mean={np.mean(heuristic_rewards):.2f}, std={np.std(heuristic_rewards):.2f}')

# # # Usage:
# # plot_rewards(episode_reward_random, episode_reward_heuristic)

# #INTIALIZE Q-TABLE TO ZERO FOR ALL STATE-ACTION PAIRS
def initialize_q_table(p_loc,d_loc,a_sel):
    q_table={
        (taxi_row,taxi_col,p,d,a):0
        for taxi_row in range(5)
        for taxi_col in range(5)
        for p in p_loc
        for d in d_loc
        for a in a_sel
    }
    return q_table

class QLearningAgent:
    def __init__(self,env:gym.Env,learning_rate:float,initial_epsilon:float,\
                 epsilon_decay:float,final_epsilon:float,discount_factor:float=0.9):
        
        '''
        Initialize a Q-Learning agent.
        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        '''
        #Intialise the environment
        self.env = env
        #Learning rate (alpha)
        self.lr = learning_rate
        #Discount factor (gamma)
        self.gamma = discount_factor

        #Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay  
        self.final_epsilon = final_epsilon

        #Track Learning Progress
        self.training_error = []

        #Initialize Q-table as a dictionary of state-action pairs
        self.q_table = initialize_q_table(p_loc, d_loc, a_sel)
    
    #get q value
    def get_q_value(self,t_row,t_col,p_loc, d_loc, a_sel):
        return self.q_table[(t_row, t_col, p_loc, d_loc, a_sel)]

    def get_valid_actions(self,info:tuple[np.int8,np.int8,np.int8,np.int8,np.int8]) -> list:
        '''
        Get the list of valid actions based on the current state information.
        Args:
            info: The current information (state)
        Returns:
            A list of valid actions (0-5) that can be taken in the current state
        '''
        return [a for a in range(6) if info["action_mask"][a]]
    
    def get_best_action(self, taxi_row, taxi_col, passenger_location, destination_location, info):
        valid_actions = self.get_valid_actions(info)
        if not valid_actions:
            return self.env.action_space.sample()  # No valid actions, return a random action
        best_action = valid_actions[0]
        best_q = self.get_q_value(taxi_row, taxi_col, passenger_location, destination_location, best_action)

        for action in valid_actions[1:]:
            q = self.get_q_value(taxi_row, taxi_col, passenger_location, destination_location, action)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action
    
    # #heuristic action selection method
    # def heuristic_action(self,info:tuple[np.int8,np.int8,np.int8,np.int8,np.int8],passenger_location:int,destination_location:int,taxi_row:int,taxi_col:int) -> int:
    #     # Extract the q_value for the valid values from info. ie call get_q_value for all locations where info["action_mask"]==1
    #     valid_q_values = [self.get_q_value(taxi_row, taxi_col, passenger_location, destination_location, a) for a in range(6) if info["action_mask"][a]]
    #     # Select the action with the highest Q-value among the valid actions
    #     if valid_q_values:
    #         max_q_value = max(valid_q_values)
    #         best_actions = [a for a in range(6) if info["action_mask"][a] and self.get_q_value(passenger_location, destination_location, a) == max_q_value]
    #         return np.random.choice(best_actions)  # Randomly select among the best actions
    #     pass
    # Action Selection Stage
    def get_action(self, taxi_row, taxi_col, passenger_location, destination_location, info):
        # Forced pickup heuristic
        valid_actions = self.get_valid_actions(info)
        if passenger_location != p_loc.IN_TAXI:
            passenger_row, passenger_col = self.get_passenger_location_coordinates(passenger_location,taxi_row,taxi_col)
            if (taxi_row, taxi_col) == (passenger_row, passenger_col):
                return a_sel.PICKUP  # Force pickup if the taxi is at the passenger's location and the passenger is not in the taxi
            
        if passenger_location == p_loc.IN_TAXI:  
            if (taxi_row, taxi_col) == self.get_destination_location_coordinates(destination_location):
                return a_sel.DROPOFF  # Force dropoff if the taxi is at the destination and the passenger is in the taxi
        

        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.choice(valid_actions)
        # Exploit
        return self.get_best_action(taxi_row, taxi_col, passenger_location, destination_location, info)

    def get_passenger_location_coordinates(self, passenger_location:int, taxi_row:int, taxi_col:int) -> tuple:
        '''
        Get the coordinates of the passenger location based on its discrete value.
        Args:
            passenger_location: The discrete value representing the passenger location
            taxi_row: The row position of the taxi
            taxi_col: The column position of the taxi
        Returns:
            A tuple of (row, column) coordinates for the passenger location
        '''
        if passenger_location == p_loc.RED:
            return 0, 0
        elif passenger_location == p_loc.GREEN:
            return 0, 4
        elif passenger_location == p_loc.YELLOW:
            return 4, 0
        elif passenger_location == p_loc.BLUE:
            return 4, 4
        else: #return taxi coordinates if passenger is in the taxi
            return taxi_row, taxi_col
        
    def get_destination_location_coordinates(self, destination_location:int) -> tuple:
        '''
        Get the coordinates of the destination location based on its discrete value.
        Args:
            destination_location: The discrete value representing the destination location
        Returns:
            A tuple of (row, column) coordinates for the destination location
        '''
        if destination_location == d_loc.RED:
            return 0, 0
        elif destination_location == d_loc.GREEN:
            return 0, 4
        elif destination_location == d_loc.YELLOW:
            return 4, 0
        elif destination_location == d_loc.BLUE:
            return 4, 4
        

    def get_next_state(self,taxi_row:int, taxi_col:int, passenger_location:int, destination_location:int, action:int) -> tuple:
        '''
        Get the next state resulting from taking an action in the current state.
        Args:
            taxi_row: The row position of the taxi
            taxi_col: The column position of the taxi
            passenger_location: The location of the passenger
            destination_location: The location of the destination
            action: The action taken
        '''
        if action == a_sel.MOVE_SOUTH:
            taxi_position_next = min(taxi_row + 1, 4), taxi_col
            passenger_location_next = passenger_location
        elif action == a_sel.MOVE_NORTH:
            taxi_position_next = max(taxi_row - 1, 0), taxi_col
            passenger_location_next = passenger_location
        elif action == a_sel.MOVE_EAST:
            taxi_position_next = taxi_row, min(taxi_col + 1, 4)
            passenger_location_next = passenger_location
        elif action == a_sel.MOVE_WEST:
            taxi_position_next = taxi_row, max(taxi_col - 1, 0)
            passenger_location_next = passenger_location
        elif action == a_sel.PICKUP:
            if passenger_location != p_loc.IN_TAXI and (taxi_row, taxi_col) == self.get_passenger_location_coordinates(passenger_location, taxi_row, taxi_col):
                passenger_location_next = p_loc.IN_TAXI
            else:
                passenger_location_next = passenger_location
            taxi_position_next = taxi_row, taxi_col
        elif action == a_sel.DROPOFF:
            if passenger_location == p_loc.IN_TAXI and (taxi_row, taxi_col) == self.get_destination_location_coordinates(destination_location):
                passenger_location_next = destination_location  # Passenger is dropped off at the destination
            else:
                passenger_location_next = passenger_location
            taxi_position_next = taxi_row, taxi_col 
        return taxi_position_next[0], taxi_position_next[1], passenger_location_next, destination_location, action       

    def update_q_table(self, observation:tuple, taxi_row:int, taxi_col:int, passenger_location:int, destination_location:int, action:int, reward:int, done:bool, info:tuple[np.int8,np.int8,np.int8,np.int8,np.int8]):
        '''
        Update the Q-table based on the observed transition.
        Args:
            taxi_row: The row position of the taxi
            taxi_col: The column position of the taxi
            passenger_location: The location of the passenger
            destination_location: The location of the destination
            action: The action taken
            reward: The reward received
            next_state: The resulting state after taking the action
            done: Whether the episode is done
        '''
        # Update Q-value using the Q-learning update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s,a))
        old_q_value = self.get_q_value(taxi_row, taxi_col, passenger_location, destination_location, action)
        
        if done:
            target = reward  # No future rewards if the episode is done
        else:             
            # Get all valid actions for the next state and their Q-values
            next_q_values = []
            for next_action in self.get_valid_actions(info):
                #taxi_row_next, taxi_col_next, passenger_location_next, destination_location_next, _ = self.get_next_state(taxi_row, taxi_col, passenger_location, destination_location, next_action)
                taxi_row_next, taxi_col_next, passenger_location_next, destination_location_next = self.env.unwrapped.decode(observation)
                next_q = self.get_q_value(taxi_row_next, taxi_col_next, passenger_location_next, destination_location_next, next_action)
                next_q_values.append(next_q)
            target = reward + self.gamma * max(next_q_values)
        # Update the Q-table with the new Q-value
        new_q_value = old_q_value + self.lr * (target - old_q_value)
        self.q_table[(taxi_row, taxi_col, passenger_location, destination_location, action)] = new_q_value


    


#Training Hyperparameters
learning_rate = 0.1
starting_epsilon = 0.2

env_rl_agent = gym.make("Taxi-v3", render_mode="human")
agent = QLearningAgent(env_rl_agent, learning_rate, starting_epsilon, epsilon_decay=0.99, final_epsilon=0.1)

episode_over = False
total_reward = 0
episode_reward_random = []
observation,info = env_rl_agent.reset()
for episodes in range(1):
    observation,info = env_rl_agent.reset()
    while not episode_over:
        #Choose an action: 0 - move south, 1 - move north, 2 - move east, 3 - move west, 4 - pickup, 5 - dropoff
        #Action Selection Stage
        taxi_row, taxi_col, passenger_location, destination_location = env_rl_agent.unwrapped.decode(observation)
        action = agent.get_action(taxi_row, taxi_col, passenger_location, destination_location, info)  # Use the Q-learning agent to select an action
        # Outcome of the action. New State, Reward, At Terminal Step, Truncated, additional information
        observation, reward, terminated, truncated, info = env_rl_agent.step(action)
        print(f"Passenger Location: {passenger_location}, Destination Location: {destination_location}")
        print(f"Taxi Position: ({taxi_row}, {taxi_col})")
        print(f"Observation decoded: taxi_row={taxi_row}, taxi_col={taxi_col}, passenger_loc={passenger_location}, dest_loc={destination_location}")
        if truncated:
            print("Episode truncated due to time limit.")
        if terminated:
            print("Episode terminated successfully.")
        episode_over = terminated or truncated
        agent.update_q_table(observation,taxi_row, taxi_col, passenger_location, destination_location, action, reward, episode_over, info)  # Update Q-table based on the transition
        #reward +1 for each step the pole stays upright
        #Terminated / At Terminal Step : True if pole falls too far
        #Truncated : True if the time limit is hit
        total_reward += reward
        
        #Introduce a small delay to make the rendering visible (optional)
        time.sleep(0.1)
        episode_reward_random.append(reward)
    episode_over = False
print(f"Episode finished! Total reward: {total_reward}")
print(f"Rewards per step: {episode_reward_random}")
env_rl_agent.close()