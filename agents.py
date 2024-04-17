import numpy as np
import json
import random
import csv
import matplotlib.pyplot as plt
from constants import GRID_SIZE, CAPACITY, PICKUP_LOCATIONS, DROPOFF_LOCATIONS, ACTIONS

class Agent:
    """
    Agent class represents an entity in the environment capable of performing actions 
    based on policies using a Q-table for decision making.
    """
    def __init__(self, alpha, gamma):
        """
        Initializes an agent with specific learning parameters alpha and gamma.
        It also initializes the Q-table for storing the value of each state-action pair.
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        """
        Initializes the Q-table with all state-action pairs set to a default value (0.0).
        States are defined by the grid position and whether the agent is carrying a block.
        """
        q_table = {}
        for x in range(1, GRID_SIZE + 1):
            for y in range(1, GRID_SIZE + 1):
                for carrying in [False, True]:
                    state = (x, y, carrying)
                    q_table[state] = {a: 0.0 for a in ACTIONS}
        return q_table

    def select_action(self, state, agent, policy):
        """
        Selects an action to perform based on the current state and specified policy.
        Actions can include moving directions, pickup, and dropoff, with decisions influenced by the Q-table.
        """
        valid_actions = self.get_valid_actions(state, agent)
        current_pos = state['positions'][agent]
        carrying = state['carrying'][agent]
        state_as_tuple = (current_pos[0], current_pos[1], carrying)

        if not valid_actions:
            return None  # No valid actions available

        if 'pickup' in valid_actions or 'dropoff' in valid_actions:
            if 'pickup' in valid_actions:
                return 'pickup'
            if 'dropoff' in valid_actions:
                return 'dropoff'

        if policy == 'PRandom':
            return random.choice(valid_actions)
        elif policy == 'PGreedy':
            valid_q_values = {action: self.q_table[state_as_tuple][action] for action in valid_actions}
            max_q_value = max(valid_q_values.values())
            max_actions = [action for action, q in valid_q_values.items() if q == max_q_value]
            return random.choice(max_actions)
        elif policy == 'PExploit':
            if random.random() < 0.8:
                valid_q_values = {action: self.q_table[state_as_tuple][action] for action in valid_actions}
                max_q_value = max(valid_q_values.values())
                max_actions = [action for action, q in valid_q_values.items() if q == max_q_value]
                return random.choice(max_actions)
            else:
                return random.choice(valid_actions)

        return None

    def get_valid_actions(self, state, agent):
        """
        Determines the actions that are valid for the agent to perform from its current position,
        considering both the grid boundaries and the positions of other agents.
        """
        col, row = state['positions'][agent]
        carrying = state['carrying'][agent]
        other_agents_positions = {a: pos for a, pos in state['positions'].items() if a != agent}
        valid_actions = []

        if row > 1 and (col, row - 1) not in other_agents_positions.values():
            valid_actions.append('north')
        if row < GRID_SIZE and (col, row + 1) not in other_agents_positions.values():
            valid_actions.append('south')
        if col < GRID_SIZE and (col + 1, row) not in other_agents_positions.values():
            valid_actions.append('east')
        if col > 1 and (col - 1, row) not in other_agents_positions.values():
            valid_actions.append('west')

        if (col, row) in PICKUP_LOCATIONS and not carrying and state['blocks'][(col, row)] > 0:
            valid_actions.append('pickup')
        if (col, row) in DROPOFF_LOCATIONS and carrying and state['blocks'][(col, row)] < CAPACITY:
            valid_actions.append('dropoff')

        return valid_actions

    def update_q_table(self, state, action, reward, new_state, agent):
        """
        Updates the Q-table based on the agent's experience (state, action, reward, new_state)
        using the Q-learning update formula.
        """
        old_pos = state['positions'][agent]
        new_pos = new_state['positions'][agent]
        old_carrying = state['carrying'][agent]
        new_carrying = new_state['carrying'][agent]
        old_state = (old_pos[0], old_pos[1], old_carrying)
        new_state_tuple = (new_pos[0], new_pos[1], new_carrying)

        old_q_value = self.q_table[old_state][action]
        new_actions = self.get_valid_actions(new_state, agent)
        max_future_q = max([self.q_table[new_state_tuple][new_action] for new_action in new_actions]) if new_actions else 0

        self.q_table[old_state][action] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_future_q)

    def update_q_table_sarsa(self, state, action, reward, new_state, a_prime, agent):
        """
        Updates the Q-table using the SARSA method, which includes the action a_prime that the agent will take next.
        """
        old_pos = state['positions'][agent]
        new_pos = new_state['positions'][agent]
        old_carrying = state['carrying'][agent]
        new_carrying = new_state['carrying'][agent]
        old_state = (old_pos[0], old_pos[1], old_carrying)
        new_state_tuple = (new_pos[0], new_pos[1], new_carrying)

        old_q_value = self.q_table[old_state][action]
        next_q_value = self.q_table[new_state_tuple][a_prime] if a_prime else 0

        self.q_table[old_state][action] = old_q_value + self.alpha * (reward + self.gamma * next_q_value - old_q_value)

    def save_q_table_to_csv(self):
        """
        Saves the Q-table to CSV files, separating the data by whether the agent is carrying or not.
        """
        filename_carrying = "q_table_carrying.csv"
        filename_not_carrying = "q_table_not_carrying.csv"
        headers = ['State (x, y, carrying)', 'North', 'South', 'East', 'West', 'Pickup', 'Dropoff']

        with open(filename_carrying, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for state, actions in self.q_table.items():
                if state[2]:  # Check if the carrying flag is True
                    row = [state] + [actions[action] for action in ACTIONS]
                    writer.writerow(row)

        with open(filename_not_carrying, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for state, actions in self.q_table.items():
                if not state[2]:  # Check if the carrying flag is False
                    row = [state] + [actions[action] for action in ACTIONS]
                    writer.writerow(row)

    def plot_q_table(self):
        """
        Plots the Q-values for each action from the Q-table, showing how they vary across the grid.
        Separate plots are made for when the agent is carrying or not carrying a block.
        """
        for carrying in [False, True]:
            fig, axs = plt.subplots(1, len(ACTIONS), figsize=(20, 4))
            fig.suptitle(f'Q-values with carrying={carrying}')
            for i, action in enumerate(ACTIONS):
                data = np.zeros((GRID_SIZE, GRID_SIZE))
                for x in range(1, GRID_SIZE + 1):
                    for y in range(1, GRID_SIZE + 1):
                        data[GRID_SIZE - y, x - 1] = self.q_table[(x, y, carrying)][action]
                ax = axs[i]
                cax = ax.matshow(data, interpolation='nearest')
                fig.colorbar(cax, ax=ax)
                ax.set_title(action)
                ax.set_xlabel('Grid X')
                ax.set_ylabel('Grid Y')
                ax.invert_yaxis()  # Invert the Y-axis to make the origin at the bottom left
            plt.savefig(f'q_table_{carrying}.png')
            plt.close()

    def save_q_table(self, filename="q_table.json"):
        """
        Saves the entire Q-table to a JSON file for later use or analysis.
        """
        serializable_q_table = {str(key): {str(k): v for k, v in value.items()} for key, value in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f, indent=4)
