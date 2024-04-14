import numpy as np
import cv2
import json
import random
import csv

import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 5
ACTIONS = ['north', 'south', 'east', 'west', 'pickup', 'dropoff']
AGENT_NAMES = ['red', 'blue', 'black']
PICKUP_LOCATIONS = {(5, 1), (4, 2), (2, 5)}
DROPOFF_LOCATIONS = {(1, 1), (1, 3), (5, 4)}
BLOCKS_INITIAL = 5
CAPACITY = 5
REWARDS = {'pickup': 13, 'dropoff': 13, 'move': -1}
AGENT_COLORS = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'black': (0, 0, 0)
}
alpha = 0.3
gamma = 0.5

class Environment:
    def __init__(self):
        self.positions = {'red': (3, 3), 'blue': (3, 5), 'black': (3, 1)}
        self.blocks = {(5, 1): 5, (4, 2): 5, (2, 5): 5, (1, 1): 0, (1, 3): 0, (5, 4): 0}
        self.carrying = {'red': False, 'blue': False, 'black': False}

    def reset(self):
        self.positions = {'red': (3, 3), 'blue': (3, 5), 'black': (3, 1)}
        self.blocks = {(5, 1): 5, (4, 2): 5, (2, 5): 5, (1, 1): 0, (1, 3): 0, (5, 4): 0}
        self.carrying = {'red': False, 'blue': False, 'black': False}

    def get_state(self):
        return {
            'positions': self.positions,
            'blocks': self.blocks,
            'carrying': self.carrying
        }

    def apply_action(self, agent, action):
        col, row = self.positions[agent]
        new_col, new_row = col, row

        if action in ['north', 'south', 'east', 'west']:
            if action == 'north':
                new_row -= 1
            elif action == 'south':
                new_row += 1
            elif action == 'east':
                new_col += 1
            elif action == 'west':
                new_col -= 1

            # Check if new position is on grid and not occupied
            if 1 <= new_col <= GRID_SIZE and 1 <= new_row <= GRID_SIZE and (new_col, new_row) not in self.occupied_positions():
                self.positions[agent] = (new_col, new_row)
            else:
                # If invalid move, do nothing
                return

        if action == 'pickup' and (col, row) in PICKUP_LOCATIONS and not self.carrying[agent] and self.blocks[(col, row)] > 0:
            self.carrying[agent] = True
            self.blocks[(col, row)] -= 1
        elif action == 'dropoff' and (col, row) in DROPOFF_LOCATIONS and self.carrying[agent] and self.blocks[(col, row)] < CAPACITY:
            self.carrying[agent] = False
            self.blocks[(col, row)] += 1

    def occupied_positions(self):
        return set(self.positions.values())

class QLearningAgent:
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
    
    def save_q_table_to_csv(self, filename="q_table.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers
            headers = ['State (x, y, carrying)', 'North', 'South', 'East', 'West', 'Pickup', 'Dropoff']
            writer.writerow(headers)
            # Write data
            for state, actions in self.q_table.items():
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

class GridWorldSimulation:
    def __init__(self, total_steps, seed=None):
        self.total_steps = total_steps
        self.seed = seed
        self.environment = Environment()
        self.agent = QLearningAgent(alpha, gamma)
        # Metrics for plotting
        self.reward_history = []
        self.cumulative_rewards = 0
        self.reset_counts = []  # List to store the number of actions between resets
        self.action_count = 0

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.environment.reset()
        actions_since_last_reset = 0  # Track actions since last reset

        policies = [('PRandom', 500), ('PRandom', 8500), ('PGreedy', 8500), ('PExploit', 8500)]
        current_policy, steps_for_policy = policies.pop(0)
        current_step = 0

        while current_step < self.total_steps:
            if current_step >= steps_for_policy:
                if policies:
                    current_policy, steps = policies.pop(0)
                    steps_for_policy += steps
                else:
                    break

            img = self.create_grid_image()
            cv2.imshow('Grid World', img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            for agent in AGENT_NAMES:
                action = self.agent.select_action(self.environment.get_state(), agent, current_policy)
                if action:
                    self.action_count += 1
                    actions_since_last_reset += 1
                    new_state = self.environment.get_state()
                    self.environment.apply_action(agent, action)
                    reward = self.compute_reward(action)
                    self.cumulative_rewards += reward
                    self.agent.update_q_table(new_state, action, reward, self.environment.get_state(), agent)

            self.reward_history.append(self.cumulative_rewards)

            if all(self.environment.blocks[loc] == CAPACITY for loc in DROPOFF_LOCATIONS):
                self.environment.reset()
                self.reset_counts.append(actions_since_last_reset)
                actions_since_last_reset = 0  # Reset the action count after a terminal state is reached

            # if current_step % 1000 == 0 or current_step == self.total_steps - 1:
            #     filename = f"q_table_final_step_{current_step}_seed_{self.seed}.json"
            #     self.agent.save_q_table(filename)

            if current_step % 1000 == 0 or current_step == self.total_steps - 1:
                csv_filename = f"q_table_step__seed_{self.seed}.csv"
                image_filename_prefix = f"q_table_step_{current_step}_seed_{self.seed}"
                self.agent.save_q_table_to_csv(csv_filename)
                self.agent.plot_q_table()  # This will save images


            current_step += 1

        cv2.destroyAllWindows()
        return self.cumulative_rewards, self.action_count, self.reset_counts
    
    def create_grid_image(self, cell_size=50):
        img_size = GRID_SIZE * cell_size
        grid_img = np.full((img_size, img_size, 3), 255, np.uint8)

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            cv2.line(grid_img, (0, i * cell_size), (img_size, i * cell_size), (0, 0, 0), 1)
            cv2.line(grid_img, (i * cell_size, 0), (i * cell_size, img_size), (0, 0, 0), 1)

        # Plot pickup and dropoff locations and agents
        for loc in PICKUP_LOCATIONS | DROPOFF_LOCATIONS:
            text = 'P' if loc in PICKUP_LOCATIONS else 'D'
            color = (0, 255, 0) if loc in PICKUP_LOCATIONS else (0, 0, 255)
            count = self.environment.blocks[loc]
            cv2.putText(grid_img, f'{text} {count}', ((loc[0] - 1) * cell_size + 5, (loc[1] - 1) * cell_size + cell_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Display agents with direction arrows
        for agent, color in AGENT_COLORS.items():
            pos = self.environment.positions[agent]
            # Determine direction arrow based on last action (this requires storing last action per agent)
            # For now, just show position with a circle
            cv2.circle(grid_img, ((pos[0] - 1) * cell_size + cell_size // 2, (pos[1] - 1) * cell_size + cell_size // 2), cell_size // 4, color, -1)

        return grid_img


    def compute_reward(self, action):
        if action == 'pickup' or action == 'dropoff':
            return REWARDS[action]
        else:
            return REWARDS['move']
        
def plot_rewards(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Cumulative Rewards')
    plt.xlabel('Simulation Step')
    plt.ylabel('Cumulative Rewards')
    plt.title('Agent Learning Progress Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reset_statistics(reset_counts):
    plt.figure(figsize=(10, 5))
    plt.plot(reset_counts, marker='o', linestyle='-')
    plt.title('Number of Actions Between Resets')
    plt.xlabel('Reset Index')
    plt.ylabel('Number of Actions')
    plt.grid(True)
    plt.show()

def run_and_plot_simulation(total_steps, seed=None):
    # Initialize and run the simulation
    simulation = GridWorldSimulation(total_steps, seed)
    total_rewards, action_count, reset_counts  = simulation.run()
    print(f"Total Rewards: {total_rewards}, Total Actions Taken: {action_count}")

    # Plot the cumulative rewards history
    plot_rewards(simulation.reward_history)
    plot_reset_statistics(reset_counts)

# Run the simulation and visualize the results
run_and_plot_simulation(9000, seed=43)
