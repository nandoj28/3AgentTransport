import numpy as np
import json
import random
import csv
import matplotlib.pyplot as plt
from constants import GRID_SIZE, CAPACITY, PICKUP_LOCATIONS, DROPOFF_LOCATIONS, ACTIONS

class Agent:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        q_table = {}
        for x in range(1, GRID_SIZE + 1):
            for y in range(1, GRID_SIZE + 1):
                for carrying in [False, True]:
                    state = (x, y, carrying)
                    q_table[state] = {a: 0.0 for a in ACTIONS}
        return q_table

    def select_action(self, state, agent, policy):
        valid_actions = self.get_valid_actions(state, agent)
        current_pos = state['positions'][agent]
        carrying = state['carrying'][agent]
        state_as_tuple = (current_pos[0], current_pos[1], carrying)

        if not valid_actions:
            return None  # No valid actions available

        # Check if 'pickup' or 'dropoff' is possible and applicable first
        if 'pickup' in valid_actions or 'dropoff' in valid_actions:
            if 'pickup' in valid_actions:
                return 'pickup'
            if 'dropoff' in valid_actions:
                return 'dropoff'

        # Handling different policies
        if policy == 'PRandom':
            return random.choice(valid_actions)
        elif policy == 'PGreedy':
            # Select the action with the highest Q-value from the valid actions
            # Breaking ties by rolling a dice (random choice among highest Q-values)
            valid_q_values = {action: self.q_table[state_as_tuple][action] for action in valid_actions}
            max_q_value = max(valid_q_values.values())
            max_actions = [action for action, q in valid_q_values.items() if q == max_q_value]
            return random.choice(max_actions)
        elif policy == 'PExploit':
            # Apply the highest Q-value action with a probability of 0.8
            if random.random() < 0.8:
                valid_q_values = {action: self.q_table[state_as_tuple][action] for action in valid_actions}
                max_q_value = max(valid_q_values.values())
                max_actions = [action for action, q in valid_q_values.items() if q == max_q_value]
                return random.choice(max_actions)
            else:
                # Choose randomly among other valid actions
                return random.choice(valid_actions)

        return None

    def get_valid_actions(self, state, agent):
        col, row = state['positions'][agent]
        carrying = state['carrying'][agent]
        other_agents_positions = {a: pos for a, pos in state['positions'].items() if a != agent}
        valid_actions = []

        # Movement actions are valid unless at grid edges or another agent occupies the cell
        if row > 1 and (col, row - 1) not in other_agents_positions.values():
            valid_actions.append('north')
        if row < GRID_SIZE and (col, row + 1) not in other_agents_positions.values():
            valid_actions.append('south')
        if col < GRID_SIZE and (col + 1, row) not in other_agents_positions.values():
            valid_actions.append('east')
        if col > 1 and (col - 1, row) not in other_agents_positions.values():
            valid_actions.append('west')

        # Pickup and dropoff checks remain unchanged
        if (col, row) in PICKUP_LOCATIONS and not carrying and state['blocks'][(col, row)] > 0:
            valid_actions.append('pickup')
        if (col, row) in DROPOFF_LOCATIONS and carrying and state['blocks'][(col, row)] < CAPACITY:
            valid_actions.append('dropoff')

        return valid_actions

    def update_q_table(self, state, action, reward, new_state, agent):
        # Extract relevant state details
        old_pos = state['positions'][agent]
        new_pos = new_state['positions'][agent]
        old_carrying = state['carrying'][agent]
        new_carrying = new_state['carrying'][agent]

        # Convert positions and carrying status to state tuples
        old_state = (old_pos[0], old_pos[1], old_carrying)
        new_state_tuple = (new_pos[0], new_pos[1], new_carrying)

        # Compute Q-value update
        old_q_value = self.q_table[old_state][action]

        # Get the maximum Q-value for applicable actions in the new state
        new_actions = self.get_valid_actions(new_state, agent)
        max_future_q = max([self.q_table[new_state_tuple][new_action] for new_action in new_actions]) if new_actions else 0

        # Q-Learning update formula
        self.q_table[old_state][action] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_future_q)

    def update_q_table_sarsa(self, state, action, reward, new_state, a_prime, agent):
        # Extract relevant state details
        old_pos = state['positions'][agent]
        new_pos = new_state['positions'][agent]
        old_carrying = state['carrying'][agent]
        new_carrying = new_state['carrying'][agent]

        # Convert positions and carrying status to state tuples
        old_state = (old_pos[0], old_pos[1], old_carrying)
        new_state_tuple = (new_pos[0], new_pos[1], new_carrying)

        # Compute Q-value update
        old_q_value = self.q_table[old_state][action]
        # For SARSA, use the Q-value of the next action the agent is going to take
        next_q_value = self.q_table[new_state_tuple][a_prime] if a_prime else 0

        # SARSA update formula
        self.q_table[old_state][action] = old_q_value + self.alpha * (reward + self.gamma * next_q_value - old_q_value)

    def save_q_table_to_csv(self):
        # File paths for the two states
        filename_carrying = "q_table_carrying.csv"
        filename_not_carrying = "q_table_not_carrying.csv"

        # Headers for the CSV files
        headers = ['State (x, y, carrying)', 'North', 'South', 'East', 'West', 'Pickup', 'Dropoff']

        # Save Q-table for carrying state
        with open(filename_carrying, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for state, actions in self.q_table.items():
                if state[2]:  # Check if the carrying flag is True
                    row = [state] + [actions[action] for action in ACTIONS]
                    writer.writerow(row)

        # Save Q-table for not carrying state
        with open(filename_not_carrying, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for state, actions in self.q_table.items():
                if not state[2]:  # Check if the carrying flag is False
                    row = [state] + [actions[action] for action in ACTIONS]
                    writer.writerow(row)

    def plot_q_table(self):
        for carrying in [False, True]:
            fig, axs = plt.subplots(1, len(ACTIONS), figsize=(20, 4))
            fig.suptitle(f'Q-values with carrying={carrying}')
            for i, action in enumerate(ACTIONS):
                data = np.zeros((GRID_SIZE, GRID_SIZE))
                for x in range(1, GRID_SIZE + 1):
                    for y in range(1, GRID_SIZE + 1):
                        # Adjust the y-index for inversion here by using GRID_SIZE - y
                        data[GRID_SIZE - y, x - 1] = self.q_table[(x, y, carrying)][action]
                ax = axs[i]
                cax = ax.matshow(data, interpolation='nearest')
                fig.colorbar(cax, ax=ax)
                ax.set_title(action)
                ax.set_xlabel('Grid X')
                ax.set_ylabel('Grid Y')
                # Invert the Y-axis to make the origin at the bottom left
                ax.invert_yaxis()
            plt.savefig(f'q_table_{carrying}.png')
            plt.close()

    def save_q_table(self, filename="q_table.json"):
        # Convert Q-table to a savable format
        # Creating a dictionary that is serializable with custom formatting for better readability
        serializable_q_table = {str(key): {str(k): v for k, v in value.items()} for key, value in self.q_table.items()}
        with open(filename, 'w') as f:
            # Dump the dictionary to a JSON file with indentation for readability
            json.dump(serializable_q_table, f, indent=4)
