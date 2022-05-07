import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from read_maze import load_maze, get_local_maze_information

####################################################################################################################################################

# Load the maze (This should only be called ONCE in the entire program)
load_maze()

####################################################################################################################################################

class FeatureNetwork(nn.Module):
    """
    Neural network for mapping agent states to a feature vector.
    Input: agent state
    Output: feature vector
    """
    def __init__(self, in_channel, out_channel):
        """out_channel: number of output channels in the last convolutional layer before being flattened and returned"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding="same")
        self.conv2 = nn.Conv2d(16, out_channel, 3, padding="same")

    def forward(self, state):
        feature = F.relu(self.conv1(state))
        feature = F.relu(self.conv2(feature))
        feature = feature.view(feature.shape[0], -1)
        return feature


class QNetwork(nn.Module):
    """
    Neural network to estimate the q-values.
    Input: feature vector output from the feature network
    Output: q values
    """

    def __init__(self, feature_dim, num_action):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, num_action)

    def forward(self, feature):
        output = F.relu(self.fc1(feature))
        output = self.fc2(output)
        return output

####################################################################################################################################################

class Environment:
    """
    Provides all functions to interact with the environment
    """
    def __init__(self, goal_x=199, goal_y=199, start_x=1, start_y=1, fire=True):
        self.timestep = 0
        self.maze_size = 201
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.maze = torch.zeros((self.maze_size, self.maze_size, 2))
        self.x = start_x
        self.y = start_y
        self.num_action = 5
        self.fire = fire

        # Make initial observation
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()

    def update_maze(self):
        """Update the maze according to self.around and decrement fire"""
        self.maze[:, :, 1] = torch.where(self.maze[:, :, 1] > 0, self.maze[:, :, 1] - 1.0, self.maze[:, :, 1])
        self.maze[self.y-1:self.y+2, self.x-1:self.x+2] = self.around

    def get_legal_actions(self):
        """
        Return all legal actions from current state. 
        Illegal actions: Walk out of the maze, walk into the wall, walk into a fire.
        """
        # Stay
        legal_actions = [0]
        # Left
        if self.around[1][0][0] == 1 and self.around[1][0][1] == 0 and self.x - 1 >= 0 and self.x - 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            legal_actions.append(1)
        # Right
        if self.around[1][2][0] == 1 and self.around[1][2][1] == 0 and self.x + 1 >= 0 and self.x + 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            legal_actions.append(2)
        # Up
        if self.around[0][1][0] == 1 and self.around[0][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y - 1 >= 0 and self.y - 1 < self.maze_size:
            legal_actions.append(3)
        # Down
        if self.around[2][1][0] == 1 and self.around[2][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y + 1 >= 0 and self.y + 1 < self.maze_size:
            legal_actions.append(4)
        return legal_actions

    def get_next_position(self, action):
        """
        Return next position if action is taken. 
        Note that this is not really taking an action, the environment would not change.
        Also note that it does not care whether the action is legal or not.
        """
        if action == 0:  # Stay
            x = self.x
            y = self.y
        elif action == 1:  # Left
            x = self.x - 1
            y = self.y
        elif action == 2:  # Right
            x = self.x + 1
            y = self.y
        elif action == 3:  # Up
            x = self.x
            y = self.y - 1
        elif action == 4:  # Down
            x = self.x
            y = self.y + 1
        else:
            raise ValueError(f"Unknown Action: {action}")
        return x, y

    def make_action(self, action):
        """
        Take 1 of the following actions: stay, left, right, up, down. 
        Increment timestep.
        Update environment states (x, y, around, maze).

        Return a reward: 0 if game ends, -1 otherwise.
        """
        reward = -1.0

        if action not in self.get_legal_actions():  # If action is illegal, stay at current position and discount reward by 1
            action = 0
            reward -= 1

        self.x, self.y = self.get_next_position(action)

        # Update agent states
        self.timestep += 1
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()

        if self.game_end():
            return 0.0

        return reward

    def game_end(self):
        """Return True if agent reaches the bottom right corner"""
        if self.x == self.goal_x and self.y == self.goal_y:  # 201 - 1 - wall
            return True
        return False

    def restart(self, x=1, y=1):
        """
        Move the agent to the starting position.
        Note that some fire might remain in the maze because we cannot call load_maze() again, 
        but they should be far away from the starting point so it does not really matter.
        """
        self.timestep = 0
        self.x = x
        self.y = y

        # Make initial observation
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()

####################################################################################################################################################

class QLearningAgent:
    """
    Agent that learns the optimal path to solve the maze using Q-Learning.
    """

    def __init__(self, env, lr=1e-4, target_update=10, batch_size=128, input_size=9, buffer_size=torch.inf, device='cpu'):
        # Environment and initial position
        self.env = env
        self.x = env.x
        self.y = env.y

        # For Exploration
        self.visited_times = torch.zeros((self.env.maze_size, self.env.maze_size))  # Reset everytime after calling restart()
        self.last_visited_timestep = torch.zeros((self.env.maze_size, self.env.maze_size))  # Do NOT reset after calling restart()

        # Training set up
        self.batch_size = batch_size
        self.target_update = target_update  # Timesteps between target network update
        self.input_size = input_size  # Size of feature_net input
        self.out_channel = 32
        self.feature_dim = self.out_channel * self.input_size ** 2
        self.lr = lr
        self.device = device
        self.replay_buffer = []  # list of tuples: (state, action, next_state, reward, done)
        self.buffer_size = buffer_size

        # Define networks
        self.feature_net = FeatureNetwork(self.get_state(self.y, self.x).shape[0], self.out_channel).to(device)
        self.feature_optimizer = optim.Adam(self.feature_net.parameters(), lr=self.lr)
        self.q_net = QNetwork(feature_dim=self.feature_dim, num_action=self.env.num_action).to(device)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.target_feature_net = FeatureNetwork(self.get_state(self.y, self.x).shape[0], self.out_channel).to(device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        self.target_feature_net.eval()
        self.target_q_net = QNetwork(feature_dim=self.feature_dim, num_action=self.env.num_action).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

    def load_model(self, path):
        """Load model state dict from file"""
        print(f"Loading models from {path}")
        model_info = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(model_info["feature_net"])
        self.feature_optimizer.load_state_dict(model_info["feature_optimizer"])
        self.q_net.load_state_dict(model_info["q_net"])
        self.q_optimizer.load_state_dict(model_info["q_optimizer"])

        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        self.target_feature_net.eval()
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
    
    def load_replay_buffer(self, path):
        """Load replay buffer to the agent. Multiple replay buffers can be loaded by calling this function multiple times with different paths."""
        # Load previoius replay buffer
        print(f"Loading replay buffer from: {path}")
        previous_replay_buffer = torch.load(path, map_location=self.device)
        self.replay_buffer.extend(previous_replay_buffer["buffer"])
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, self.buffer_size)
        print("Number of data in replay buffer:", len(self.replay_buffer))

    def save_maze_policy(self, path):
        """Save the current policy as an image"""
        policy_image = torch.zeros((self.env.maze_size, self.env.maze_size))
        for y in range(self.env.maze_size):
            for x in range(self.env.maze_size):
                if self.env.maze[y, x, 0]:  # not a wall
                    # Comput the best action
                    state = self.get_state(y, x).to(self.device)
                    action_values = self.q_net(self.feature_net(state.unsqueeze(0)))[0]
                    best_action = torch.argmax(action_values)
                    policy_image[y, x] = best_action

        # Plot heatmap to show frequency of visiting each position
        newcmap = ListedColormap(['black', 'blue', 'red', 'yellow', 'white'])
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(policy_image, cmap=newcmap, vmin=0, vmax=4)
        fig.colorbar(im)
        plt.savefig(path)
        plt.close(fig)

    def save_visit_frequency(self, path):
        # Save a heatmap showing frequency of visiting each position
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(self.visited_times, cmap='gray')
        fig.colorbar(im)
        plt.savefig(path)
        plt.close(fig)

    def save_explored_maze(self, path):
        """Save the explored positions as an image"""
        plt.figure(figsize=(20, 20))
        plt.imshow((self.visited_times > 0), cmap="gray")
        plt.savefig(path)
        plt.close()

    def get_state(self, y, x):
        """
        Return a tensor representing the current state 
        A state is a stack of 2D tensors with dimension (self.input_size, self.input_size) representing 
        the agent's position, walls, fires, y and x coordinates of the top left corner of the input.
        """
        # Find the top left corner of the window, such that it is not out of the maze
        x_left = max(0, x - int((self.input_size+1)/2))
        x_left = min(x_left, self.env.maze_size - self.input_size)
        y_top = max(0, y - int((self.input_size+1)/2))
        y_top = min(y_top, self.env.maze_size - self.input_size)

        state = torch.zeros((6, self.input_size, self.input_size))
        state[0, y - y_top, x - x_left] = 1  # Current position
        state[1] = self.env.maze[y_top:y_top+self.input_size, x_left:x_left+self.input_size, 0]  # Walls
        state[2] = self.env.maze[y_top:y_top+self.input_size, x_left:x_left+self.input_size, 1]  # Fires
        state[3] = torch.full((self.input_size, self.input_size), y_top)
        state[4] = torch.full((self.input_size, self.input_size), x_left)

        return state

    def restart(self, x=1, y=1):
        """Restart the environment. Called everytime before starting a new episode."""
        self.env.restart(x=x, y=y)
        self.x = x
        self.y = y
        self.visited_times = torch.zeros((self.env.maze_size, self.env.maze_size))

    def get_action(self, state, mode="full_explore", epsilon=0.8):
        """Given a state, return an action according to the policy."""
        state = state.to(self.device)

        legal_actions = self.env.get_legal_actions()

        if mode == "full_explore":
            action_to_least_visited_neighbour = None
            min_visited_times = torch.inf
            for action in range(5):
                next_x, next_y = self.env.get_next_position(action)
                if self.visited_times[next_y, next_x] <= min_visited_times:
                    # Legal action, record it
                    if action in legal_actions:
                        min_visited_times = self.visited_times[next_y, next_x]
                        action_to_least_visited_neighbour = action
                    # Illegal action but it is due to fire: wait there
                    elif self.env.maze[next_y, next_x, 0]:
                        action_to_least_visited_neighbour = 0
                        min_visited_times = self.visited_times[next_y, next_x]

            return action_to_least_visited_neighbour

        elif mode == "warm_up":  # Go to a position closest to the goal most of the time, pick random action otherwise
            action_closest_to_goal = None
            max_visited_timestep = -torch.inf
            for action in range(5):
                if action in legal_actions:
                    next_x, next_y = self.env.get_next_position(action)
                    if self.last_visited_timestep[next_y, next_x] > max_visited_timestep:
                        max_visited_timestep = self.last_visited_timestep[next_y, next_x]
                        action_closest_to_goal = action
            if random.random() < 0.5:
                return random.randint(0, 4)
            return action_closest_to_goal

        elif mode == "q_learning":
            # Epsilon Greedy
            if random.random() < epsilon:
                # Return random action
                return random.randint(0, 4)
            else:
                # Return Best Action
                action_values = self.q_net(self.feature_net(state.unsqueeze(0)))[0]
                max_action = torch.argmax(action_values).item()
                return max_action
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def update_network(self, gamma=0.99):
        """Update network weights"""
        batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
        state_batch = torch.stack([data[0].to(self.device) for data in batch])
        action_batch = torch.tensor([data[1] for data in batch]).to(self.device)
        next_state_batch = torch.stack([data[2].to(self.device) for data in batch])
        reward_batch = torch.tensor([data[3] for data in batch]).to(self.device)
        done = torch.tensor([data[4] for data in batch]).to(self.device)

        #----- Compute the loss of q values prediction -----#
        # Prediction (Only update against the action that the agent took)
        q_values_prediction = self.q_net(self.feature_net(state_batch))  # q value for all actions
        q_values_prediction = q_values_prediction[torch.arange(q_values_prediction.shape[0]), action_batch]  # q value for the selected action

        # Target (Next state's best q value)
        next_q_values = self.target_q_net(self.target_feature_net(next_state_batch))
        next_best_q_values, _ = torch.max(next_q_values, dim=1)
        next_best_q_values = next_best_q_values.detach()  # Stop Gradient

        q_criterion = nn.SmoothL1Loss()
        q_loss = q_criterion(q_values_prediction,  reward_batch + gamma * next_best_q_values * (1 - done))  # Multiply by (1 - done) to set target value of goal state to 0

        #----- Backprop all losses -----#
        self.feature_optimizer.zero_grad()
        self.q_optimizer.zero_grad()

        loss = q_loss
        loss.backward()

        self.feature_optimizer.step()
        self.q_optimizer.step()

        return loss.cpu().item()

    def run_episode(self, start_x=1, start_y=1, mode="q_learning", epsilon=0.8, gamma=0.99, max_timestep=20000, update=True, save=True, display=None):
        """
        Run a single episode.
        If both "full_explore" and "warm_up" modes are called, they should have the same start_x and start_y.
        
        Parameters: 
            - start_x: x-coordinate of the starting position
            - start_y: y-coordinate of the starting position
            - mode: controls how to run the current episode. ("full_explore", "warm_up" or "q_learning")
            - epsilon: parameter for epsilon-greedy
            - gamma: discount factor
            - max_timestep: timestep before terminating the episode
            - buffer_size: maximum size of replay buffer
            - update: update network weights during the episode if True.
            - save: save trajectory to replay buffer if True
            - display: timesteps between outputing each progress message
        
        Return:
            - timestep: timestep used to complete the episode
            - reward: total reward
            - loss: a list of losses during every timestep
            - done: True if the episode ends with agent reaching goal state, False otherwise.
        """
        # Start from start state
        self.restart(x=start_x, y=start_y)

        total_reward = 0
        losses = []

        # For displaying the first timestep of an episode only
        action = 0
        reward = 0

        start_time = time.perf_counter()
        while not self.env.game_end() and self.env.timestep < max_timestep:
            # Display training log
            if display is not None and self.env.timestep % display == 0:
                # Print current state
                print(f"Timestep: {self.env.timestep} | Position: ({self.x},{self.y}) | Epsilon: {epsilon:.4f} | Last Action: {action} | Reward: {reward:4f} | Time Used: {time.perf_counter() - start_time:.2f}")
                state = self.get_state(self.y, self.x).to(self.device)
                action_values = self.q_net(self.feature_net(state.unsqueeze(0)))[0]
                print("Q-values: ", action_values.detach().cpu().numpy())

            # Store current state info
            previous_y, previous_x = self.y, self.x
            no_fire_around = self.env.around[[0, 1, 1, 2], [1, 0, 2, 1], 1].sum().item() == 0

            # Select action
            state = self.get_state(self.y, self.x)
            action = self.get_action(state, mode=mode)

            # Make an action, update agent's position and exploration info
            reward = self.env.make_action(action)
            self.y, self.x = self.env.y, self.env.x
            self.visited_times[self.y, self.x] += 1
            if mode == "full_explore":
                self.last_visited_timestep[self.y, self.x] = self.env.timestep
            # Set timestep of unvisited position as the visited position - 1. The start state should have value 0.
            if mode == "warm_up" and self.last_visited_timestep[self.y, self.x] == 0 and (self.y != start_x or self.x != start_y):
                self.last_visited_timestep[self.y, self.x] = self.last_visited_timestep[previous_y, previous_x] - 1

            # Reward Shaping
            # # Penalize going to previous state unless the action is stay
            # if trajectory and torch.equal(self.get_state(self.y, self.x)[0], trajectory[-1][0][0]) and action != 0:
            #     reward -= 1

            # If agent stays while there is no fire around, agent gets -0.5 reward. This is to solve the problem where the agent tends to stay to get more reward.
            # For example, q value of staying is 1, but max q of all other next states are < 1. Both rewards are -1 because they are not goal state, so the agent prefers to stay.
            if action == 0 and no_fire_around:
                reward -= 0.5

            total_reward += reward

            # Get next state
            next_state = self.get_state(self.y, self.x)

            # Save to replay buffer
            if save:
                self.replay_buffer.append([state, action, next_state, reward, int(self.env.game_end())])

                if len(self.replay_buffer) > self.buffer_size:
                    self.replay_buffer = random.sample(self.replay_buffer, self.buffer_size)

            # Update networks
            if update:
                loss = self.update_network(gamma=gamma)
                losses.append(loss)

                if self.env.timestep % self.target_update == 0:  # Update target network
                    self.target_feature_net.load_state_dict(self.feature_net.state_dict())
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Make sure target nets are up to date after an episode
        if update:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        print(f"Timestep: {self.env.timestep} | Position: ({self.x},{self.y}) | Epsilon: {epsilon:.4f} | Last Action: {action} | Reward: {reward:4f} | Time Used: {time.perf_counter() - start_time:.2f}")

        # Return total reward
        return self.env.timestep, total_reward, losses, self.env.game_end()

    def train(self, full_explore=1, warm_up=50, q_learning=5, epsilon=0.8, min_epsilon=0.1, epsilon_decay=1e-3, move_start_position=True, previous_training_history_path=None, previous_eval_path=None, previous_episode_path=None, starting_position_path=None):
        """
        Training loop.

        Paremeters:
            - full_explore: number of times to run an episode in "full_explore" mode.
            - warm_up: number of times to run an episode in "warm_up" mode.
            - q_learning: number of times to run an episode in "q_learning" mode per starting points. (If move_start_position is True and there are 5 starting points, (5 * q_learning) episodes will be executed in q_learning mode.)
            - epsilon: initial value of epsilon for epsilon greedy exploration.
            - min_epsilon: minimum epsilon value.
            - epsilon_decay: epsilon decay constant subtracted from epsilon after each q_learning episode.
            - move_start_position: move the starting point further away from the goal during training.
            - previous_training_history_path: file path storing previous training history.
            - previous_replay_buffer_path: file path storing previous replay buffer.
        
        Return:
            - timestep: list of timestep used in q_learning episodes
            - reward: list of rewards from q_learning episodes
            - loss: a list of losses from q_learning episodes
        """
        timesteps = []
        rewards = []
        losses = []

        eval_timesteps = []
        eval_rewards = []
        eval_done = []

        # Load previous training history
        if previous_training_history_path is not None:
            print(f"Loading previous training history from: {previous_training_history_path}")
            previous_training_history = torch.load(previous_training_history_path, map_location=self.device)
            timesteps = previous_training_history["timesteps"]
            rewards = previous_training_history["rewards"]
            losses = previous_training_history["losses"]

        # Load previous evaluation history
        if previous_eval_path is not None:
            print(f"Loading previous evaluation history from: {previous_eval_path}")
            previous_eval_history = torch.load(previous_eval_path, map_location=self.device)
            eval_timesteps = previous_eval_history["timesteps"]
            eval_rewards = previous_eval_history["rewards"]
            eval_done = previous_eval_history["done"]

        #----- Exploration Phase -----#
        for i in range(full_explore):  # To explore the maze, initialize self.last_visited_timestep, update the maze
            st = time.perf_counter()
            print(f"Starting Full Explore Episode {i}")
            timestep, reward, loss, done = self.run_episode(mode="full_explore", display=500, save=False, update=False)

            # Save exploration path as images
            self.save_visit_frequency(f"maze/frequency/f_{i}.png")
            self.save_explored_maze(f"maze/explored/f_{i}.png")

            print("Time used for this episode:", time.perf_counter() - st)
            print()

        #----- Warm Up Phase -----#
        print(f"Starting {warm_up} warm-up episodes.\n")
        for i in range(warm_up):
            st = time.perf_counter()
            print(f"Starting Warm Up Episode {i}")
            timestep, reward, loss, done = self.run_episode(mode="warm_up", display=500, save=True, update=False)

            # Save exploration path as images
            if i+1 % 10 == 0:
                self.save_visit_frequency(f"maze/frequency/w_{i}.png")
                self.save_explored_maze(f"maze/explored/w_{i}.png")
            
            torch.save({"buffer": self.replay_buffer}, "replay_buffer.pth")

            print("Time used for this episode:", time.perf_counter() - st)
            print()

        #----- Q-Learning Phase -----#
        # Initialize the starting positions (starting from positions closer to the goal)
        starting_position = []
        if starting_position_path is not None:
            starting_position = torch.load(starting_position_path, map_location=self.device)["starting_position"]
        elif move_start_position:
            for y in range(self.env.maze_size):
                for x in range(self.env.maze_size):
                    starting_position.append((y, x))
            # Sort according to the distance to goal state, in ascending order (goal position is the first element)
            starting_position.sort(key=lambda p: self.last_visited_timestep[p[0], p[1]], reverse=True)
            for i, p in enumerate(starting_position):  # Loop until last visited time step is zero
                if self.last_visited_timestep[p[0], p[1]] == 0:
                    break
            starting_position = starting_position[1:i]  # Ignore goal state and all unexplored positions
            starting_position.append((1, 1))  # Add the starting point back because it has a 0 last visited timestep as well
        else:
            starting_position = [(1, 1)]
        
        # Save remaining startining positions for resume training
        torch.save({"starting_position": starting_position}, "starting_position.pth")

        print(f"Starting {len(starting_position) * q_learning} q-learning episodes. {'With' if move_start_position else 'Without'} moving start position.\n")

        # Load previous episode index and epsilon
        if previous_episode_path is not None:
            previous_episode_info = torch.load(previous_episode_path)
            previous_episode_index = previous_episode_info["episode"]
            epsilon = previous_episode_info["epsilon"]
            print(f"Resuming training from episode {previous_episode_index}\n")

        # Run q_learning episodes
        for i, (y, x) in enumerate(starting_position):
            if previous_episode_path is not None and (i+1)*q_learning <= previous_episode_index: # Skip previously trained episodes
                continue
            # max_timestep = 20000
            max_timestep = 5 * (self.last_visited_timestep[self.env.goal_y, self.env.goal_x] - self.last_visited_timestep[y, x])

            for j in range(q_learning):
                if previous_episode_path is not None and i*q_learning+j <= previous_episode_index: # Skip previously trained episodes
                    continue

                st = time.perf_counter()
                print(f"Starting Q-Learning Episode {i*q_learning+j}. Starting point: ({x},{y})")

                timestep, reward, loss, done = self.run_episode(start_x=x, start_y=y, mode="q_learning",
                                                                epsilon=epsilon, display=500, max_timestep=max_timestep, update=True, save=True)

                # Save training histories and replay buffer
                timesteps.append(timestep)
                rewards.append(reward)
                losses.append(loss)

                if (i*q_learning+j+1) % 20 == 0:
                    torch.save({"timesteps": timesteps, "rewards": rewards, "losses": losses}, "training_histories.pth")

                if (i*q_learning+j+1) % 100 == 0:
                    torch.save({"buffer": self.replay_buffer}, "replay_buffer.pth")

                # Save episode index for resume training
                torch.save({"episode": i*q_learning+j, "epsilon": epsilon}, "previous_episode.pth")

                # Epsilon decay
                if epsilon > min_epsilon:
                    epsilon -= epsilon_decay

                print("Time used for this episode: ", time.perf_counter() - st)
                print()

            # Save exploration path and current policy as images
            self.save_visit_frequency(f"maze/frequency/q_{i*q_learning+j+1}.png")
            self.save_explored_maze(f"maze/explored/q_{i*q_learning+j+1}.png")
            self.save_maze_policy(f"maze/policy/q_{i*q_learning+j+1}.png")

            if (i+1)*q_learning % 100 == 0:
                # Evaluate policy by acting almost greedily
                print(f"Running evaluation episode")
                timestep, reward, _, done = self.run_episode(mode="q_learning", epsilon=0.05, display=500,
                                                                max_timestep=6000, update=False, save=False)
                eval_timesteps.append(timestep)
                eval_rewards.append(reward)
                eval_done.append(done)
                torch.save({"timesteps": timesteps, "rewards": rewards, "done": eval_done}, "eval_histories.pth")
                print()
            
            # Save model
            torch.save({
                "feature_net": self.feature_net.state_dict(),
                "q_net": self.q_net.state_dict(),
                "feature_optimizer": self.feature_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict()
            }, "models/q_learning.pth")

        # Save model
        torch.save({
            "feature_net": self.feature_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "feature_optimizer": self.feature_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict()
        }, "models/q_learning.pth")

        torch.save({"timesteps": timesteps, "rewards": rewards, "losses": losses}, "training_histories.pth")
        torch.save({"buffer": self.replay_buffer}, "replay_buffer.pth")

        return timesteps, rewards, losses

####################################################################################################################################################

def plot_graphs(training_history, eval_history):
    training_history = torch.load('training_histories.pth')
    timesteps = training_history["timesteps"]
    rewards = training_history["rewards"]
    losses = training_history["losses"]

    eval_history = torch.load('eval_histories.pth')
    eval_timesteps = eval_history["timesteps"]
    eval_rewards = eval_history["rewards"]
    eval_done = eval_history["done"]

    #----- Training graphs -----#
    fig, ax = plt.subplots(figsize=(5, 5))
    timesteps = np.array(timesteps)
    average_timesteps = [timesteps[max(0, i-10):i+11].mean() for i in range(len(timesteps))]
    ax.plot(average_timesteps)
    ax.set_ylabel("Time steps")
    ax.set_xlabel("Episode")
    ax.set_title("Training Timesteps")
    plt.savefig("graphs/training_timesteps.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    rewards = np.array(rewards)
    average_rewards = [rewards[max(0, i-10):i+11].mean() for i in range(len(rewards))]
    ax.plot(average_rewards)
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Episode")
    ax.set_title("Training Rewards")
    plt.savefig("graphs/training_rewards.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    flattened_loss = [loss for episode_loss in losses for loss in episode_loss]
    flattened_loss = np.array(flattened_loss)
    average_loss = [flattened_loss[max(0, i-250):i+251].mean() for i in range(len(flattened_loss))]
    ax.plot(average_loss)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Episode")
    ax.set_title("Training Loss")
    plt.savefig("graphs/training_losses.png")

    #----- Evaluation graphs -----#
    # Scale the x_axis because the policy is only evaluated after running all episodes for each starting point
    x_axis = np.arange(len(eval_timesteps)) * 5

    fig, ax = plt.subplots(figsize=(5, 5))
    eval_timesteps = np.array(eval_timesteps)
    average_eval_timesteps = [eval_timesteps[max(0, i-10):i+11].mean() for i in range(len(eval_timesteps))]
    ax.plot(x_axis, average_eval_timesteps)
    ax.set_ylabel("Time steps")
    ax.set_xlabel("Episode")
    ax.set_title("Evaluation Timesteps")
    plt.savefig("graphs/eval_timesteps.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    eval_rewards = np.array(eval_rewards)
    average_eval_rewards = [eval_rewards[max(0, i-10):i+11].mean() for i in range(len(eval_rewards))]
    ax.plot(x_axis, average_eval_rewards)
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Episode")
    ax.set_title("Evaluation Rewards")
    plt.savefig("graphs/eval_rewards.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    eval_done = np.array(eval_done)
    average_eval_done = [eval_done[max(0, i-10):i+11].mean() for i in range(len(eval_done))]
    ax.plot(x_axis, average_eval_done)
    ax.set_ylabel("Reaching goal state")
    ax.set_xlabel("Episode")
    ax.set_title("Evaluation reaches goal state")
    plt.savefig("graphs/eval_done.png")


####################################################################################################################################################
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = Environment(199, 199, fire=False)
    # env = Environment(28, 135, fire=False)
    # env = Environment(48, 31, fire=False)
    # env = Environment(23, 9, fire=False)
    # env = Environment(10, 3, fire=False)
    # env = Environment(1, 3, fire=False)

    agent = QLearningAgent(env, device=device)
    # agent.load_model("models/q_learning.pth")

    timesteps, rewards, losses = agent.train(full_explore=1, warm_up=50, q_learning=5, epsilon=0.8, min_epsilon=0.3, move_start_position=True)
    # timesteps, rewards, losses = agent.train(full_explore=1, warm_up=50, q_learning=5, epsilon=0.8, min_epsilon=0.3, move_start_position=True, previous_training_history_path='training_histories.pth', previous_replay_buffer_path='replay_buffer.pth', previous_eval_path='eval_histories.pth')

    plot_graphs('training_histories.pth', 'eval_histories.pth')