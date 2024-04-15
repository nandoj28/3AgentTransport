import numpy as np
import cv2
import json
import random
import matplotlib.pyplot as plt
from environment import Environment
from agents import Agent
from constants import GRID_SIZE, CAPACITY, PICKUP_LOCATIONS, DROPOFF_LOCATIONS
from constants import REWARDS, AGENT_COLORS, AGENT_NAMES, alpha, gamma

class GridWorldSimulation:
    def __init__(self, total_steps, initial_policy='PRandom', subsequent_policy='PGreedy', change_step=500, seed=None):
        self.total_steps = total_steps
        self.initial_policy = initial_policy
        self.subsequent_policy = subsequent_policy
        self.change_step = change_step
        self.seed = seed
        self.environment = Environment()
        self.agent = Agent(alpha, gamma)
        self.current_policy = initial_policy
        # Metrics for plotting
        self.reward_history = []
        self.cumulative_rewards = 0
        self.reset_counts = []
        self.action_count = 0

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.environment.reset()
        actions_since_last_reset = 0
        current_step = 0

        while current_step < self.total_steps:
            if current_step == self.change_step:
                self.current_policy = self.subsequent_policy

            img = self.create_grid_image()
            cv2.imshow('Grid World', img)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break

            # Process each agent's action as an individual step
            for agent in ['red', 'blue', 'black']:  # Explicit ordering
                if current_step >= self.total_steps:
                    break  # Exit if the total number of steps is reached

                action = self.agent.select_action(self.environment.get_state(), agent, self.current_policy)
                if action:
                    self.action_count += 1
                    actions_since_last_reset += 1
                    new_state = self.environment.get_state()
                    self.environment.apply_action(agent, action)
                    reward = self.compute_reward(action)
                    self.cumulative_rewards += reward

                    # Update the Q-table
                    self.agent.update_q_table(new_state, action, reward, self.environment.get_state(), agent)

                self.reward_history.append(self.cumulative_rewards)

                # Check if it's time to reset the environment based on the condition of all blocks delivered
                if all(self.environment.blocks[loc] == CAPACITY for loc in DROPOFF_LOCATIONS):
                    self.environment.reset()
                    self.reset_counts.append(actions_since_last_reset)
                    actions_since_last_reset = 0

                current_step += 1  # Increment the step counter after each action

                if current_step % 1000 == 0 or current_step == self.total_steps - 1:
                    # csv_filename = f"q_table_policy_{self.subsequent_policy}_seed_{self.seed}.csv"
                    self.agent.save_q_table_to_csv()
                    self.agent.plot_q_table()

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