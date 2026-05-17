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

while not episode_over:
    #Choose an action: 0 - push cart left, 1 - push cart right
    #Action Selection Stage
    action = env_random.action_space.sample() # Random action. This is not an MDP model 
    # Outcome of the action. New State, Reward, At Terminal Step, Truncated, additional information
    observation, reward, terminated, truncated, info = env_random.step(action)
    
    # Decode observation to extract state details
    taxi_row, taxi_col, passenger_location, destination_location = env_random.unwrapped.decode(observation)
    print(f"Passenger Location: {passenger_location}, Destination Location: {destination_location}")
    print(f"Taxi Position: ({taxi_row}, {taxi_col})")
    print(f"Observation decoded: taxi_row={taxi_row}, taxi_col={taxi_col}, passenger_loc={passenger_location}, dest_loc={destination_location}")
    if truncated:
        print("Episode truncated due to time limit.")
    if terminated:
        print("Episode terminated successfully.")
    #reward +1 for each step the pole stays upright
    #Terminated / At Terminal Step : True if pole falls too far
    #Truncated : True if the time limit is hit
    total_reward += reward
    episode_over = terminated or truncated
    #Introduce a small delay to make the rendering visible (optional)
    time.sleep(0.1)
    episode_reward_random.append(reward)
print(f"Episode finished! Total reward: {total_reward}")
print(f"Rewards per step: {episode_reward_random}")
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

# #NOT REQUIRED
def initialize_q_table(p_loc,d_loc,a_sel):
    q_table={
        (p,d,a):0
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

    #heuristic action selection method
    def heuristic_action(self,obs:tuple[np.float32,np.float32,np.float32,np.float32]) -> int:
        pass
    # Action Selection Stage

    def get_action(self,info:tuple[np.int8,np.int8,np.int8,np.int8,np.int8]) -> int:
        '''
        Select an action where a smarter heuristic policy is used during exploration instead of pure random actions.
        Args:
            info: The current information (state)
        Returns:
            action: The action to take (0-5)
#         '''

        if np.random.rand() < self.epsilon:
             # Explore: use the heuristic policy instead of pure random selection.
             return self.heuristic_action(info)
        else:
            # Exploit: choose a random action (since Q-values are not trained yet, this is effectively random).
             return self.env.action_space.sample()  # Placeholder for actual Q-value based action selection
            