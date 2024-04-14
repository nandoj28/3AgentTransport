import numpy as np
import cv2
import json
import random

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
        x, y = self.positions[agent]
        new_x, new_y = x, y

        if action in ['north', 'south', 'east', 'west']:
            if action == 'north' and y > 1:
                new_y -= 1
            elif action == 'south' and y < GRID_SIZE:
                new_y += 1
            elif action == 'east' and x < GRID_SIZE:
                new_x += 1
            elif action == 'west' and x > 1:
                new_x -= 1

            if (new_x, new_y) not in self.occupied_positions():
                self.positions[agent] = (new_x, new_y)
                self.occupied_positions().add((new_x, new_y))

        if action == 'pickup' and (x, y) in PICKUP_LOCATIONS and not self.carrying[agent] and self.blocks[(x, y)] > 0:
            self.carrying[agent] = True
            self.blocks[(x, y)] -= 1
        elif action == 'dropoff' and (x, y) in DROPOFF_LOCATIONS and self.carrying[agent] and self.blocks[(x, y)] < CAPACITY:
            self.carrying[agent] = False
            self.blocks[(x, y)] += 1

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
        x, y = state['positions'][agent]
        carrying = state['carrying'][agent]
        valid_actions = []

        # Movement actions are always valid unless at grid edges
        if y > 1:
            valid_actions.append('north')
        if y < GRID_SIZE:
            valid_actions.append('south')
        if x < GRID_SIZE:
            valid_actions.append('east')
        if x > 1:
            valid_actions.append('west')

        # Check if pickup or dropoff can be performed
        if (x, y) in PICKUP_LOCATIONS and not carrying and state['blocks'][(x, y)] > 0:
            valid_actions.append('pickup')
        if (x, y) in DROPOFF_LOCATIONS and carrying and state['blocks'][(x, y)] < CAPACITY:
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

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.environment.reset()

        policies = [('PRandom', 500), ('PRandom', 8500), ('PGreedy', 8500), ('PExploit', 8500)]
        current_policy, steps_for_policy = policies.pop(0)
        current_step = 0
        total_rewards = 0
        action_count = 0

        while current_step < self.total_steps:
            if current_step >= steps_for_policy:
                if policies:
                    current_policy, steps = policies.pop(0)
                    steps_for_policy += steps
                else:
                    break

            img = self.create_grid_image()
            cv2.imshow('Grid World', img)
            key = cv2.waitKey(10)

            if key & 0xFF == ord('q'):
                break

            occupied_positions = self.environment.occupied_positions()
            for agent in AGENT_NAMES:
                action = self.agent.select_action(self.environment.get_state(), agent, current_policy)
                if action is not None:
                    action_count += 1
                    new_state = self.environment.get_state()
                    self.environment.apply_action(agent, action)
                    reward = self.compute_reward(action)
                    total_rewards += reward
                    self.agent.update_q_table(new_state, action, reward, self.environment.get_state(), agent)

            if all(self.environment.blocks[loc] == CAPACITY for loc in DROPOFF_LOCATIONS):
                self.environment.reset()

            current_step += 1
            if current_step % 1000 == 0 or current_step == self.total_steps - 1:
                self.agent.save_q_table(f"q_table_final_seed_{self.seed}.json")

        cv2.destroyAllWindows()
        return total_rewards, action_count

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
            cv2.putText(grid_img, f'{text} {count}', ((loc[0] - 1) * cell_size + int(cell_size/4), (loc[1] - 1) * cell_size + int(cell_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        for agent in AGENT_NAMES:
            pos = self.environment.positions[agent]
            color = AGENT_COLORS[agent]
            cv2.circle(grid_img, ((pos[0] - 1) * cell_size + int(cell_size/2), (pos[1] - 1) * cell_size + int(cell_size/2)), int(cell_size/4), color, -1)

        return grid_img

    def compute_reward(self, action):
        if action == 'pickup' or action == 'dropoff':
            return REWARDS[action]
        else:
            return REWARDS['move']

# Example usage
simulation = GridWorldSimulation(9000, seed=43)
total_rewards, action_count = simulation.run()
print(total_rewards, action_count)
