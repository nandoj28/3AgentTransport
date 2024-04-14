import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from random import choice, random
import seaborn as sns
import cv2
import json

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

# return environment to original state
def reset_environment():
    return {
        'positions': {'red': (3, 3), 'blue': (3, 5), 'black': (3, 1)},
        'blocks': {(5, 1): 5, (4, 2): 5, (2, 5): 5, (1, 1): 0, (1, 3): 0, (5, 4): 0},
        'carrying': {'red': False, 'blue': False, 'black': False}
    }

# Initialize state
state = reset_environment()

# Initialize Q-table
def initialize_q_table():
    q_table = {}
    for x in range(1, GRID_SIZE + 1):
        for y in range(1, GRID_SIZE + 1):
            for carrying in [False, True]:
                state = (x, y, carrying)
                q_table[state] = {a: 0.0 for a in ACTIONS}
    return q_table

q_table = initialize_q_table()


def apply_action(agent, action, state, occupied_positions):
    x, y = state['positions'][agent]
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

        if (new_x, new_y) not in occupied_positions:
            state['positions'][agent] = (new_x, new_y)
            occupied_positions.add((new_x, new_y))

    if action == 'pickup' and (x, y) in PICKUP_LOCATIONS and not state['carrying'][agent] and state['blocks'][(x, y)] > 0:
        state['carrying'][agent] = True
        state['blocks'][(x, y)] -= 1
    elif action == 'dropoff' and (x, y) in DROPOFF_LOCATIONS and state['carrying'][agent] and state['blocks'][(x, y)] < CAPACITY:
        state['carrying'][agent] = False
        state['blocks'][(x, y)] += 1


def select_action(q_table, state, agent, policy):
    current_pos = state['positions'][agent]
    carrying = state['carrying'][agent]
    valid_actions = []

    # Validate possible actions
    x, y = current_pos
    if y > 1:  # North is possible
        valid_actions.append('north')
    if y < GRID_SIZE:  # South is possible
        valid_actions.append('south')
    if x < GRID_SIZE:  # East is possible
        valid_actions.append('east')
    if x > 1:  # West is possible
        valid_actions.append('west')
    if (x, y) in PICKUP_LOCATIONS and not carrying and state['blocks'][(x, y)] > 0:
        valid_actions.append('pickup')
    if (x, y) in DROPOFF_LOCATIONS and carrying and state['blocks'][(x, y)] < CAPACITY:
        valid_actions.append('dropoff')

    if policy == 'PRandom' or not valid_actions:
        return choice(valid_actions) if valid_actions else None  

    state_as_tuple = (x, y, carrying)
    valid_q_values = {action: q_table[state_as_tuple][action] for action in valid_actions}
    if policy == 'PGreedy':
        return max(valid_q_values, key=valid_q_values.get)
    elif policy == 'PExploit' and random() < 0.8:
        return max(valid_q_values, key=valid_q_values.get)
    else:
        return choice(valid_actions)


def update_q_table(q_table, state, action, reward, new_state, agent):
    # Extract relevant state details
    old_pos = state['positions'][agent]
    new_pos = new_state['positions'][agent]
    old_carrying = state['carrying'][agent]
    new_carrying = new_state['carrying'][agent]

    # Convert positions and carrying status to state tuples
    old_state = (old_pos[0], old_pos[1], old_carrying)
    new_state = (new_pos[0], new_pos[1], new_carrying)

    # Compute Q-value update
    old_q_value = q_table[old_state][action]
    future_q = max(q_table[new_state].values())  # Max Q-value for the new state
    q_table[old_state][action] = old_q_value + alpha * (reward + gamma * future_q - old_q_value)


def compute_reward(state, action, new_state, agent):
    if action == 'pickup' or action == 'dropoff':
        return REWARDS[action]
    else:
        return REWARDS['move']
def plot_q_values(q_table, position, carrying):
    actions = ACTIONS
    values = [q_table[(position, carrying)][action] for action in actions]
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x=actions, y=values, ax=ax)
    ax.set_title(f'Q-values at Position {position} Carrying: {"Yes" if carrying else "No"}')
    ax.set_ylabel('Q-value')
    plt.show()

def save_q_table(q_table, filename="q_table.json"):
    # Convert Q-table to a savable format
    # Creating a dictionary that is serializable with custom formatting for better readability
    serializable_q_table = {str(key): {str(k): v for k, v in value.items()} for key, value in q_table.items()}
    with open(filename, 'w') as f:
        # Dump the dictionary to a JSON file with indentation for readability
        json.dump(serializable_q_table, f, indent=4)


def create_grid_image(state, cell_size=50):
    img_size = GRID_SIZE * cell_size
    grid_img = np.full((img_size, img_size, 3), 255, np.uint8)

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        cv2.line(grid_img, (0, i * cell_size), (img_size, i * cell_size), (0, 0, 0), 1)
        cv2.line(grid_img, (i * cell_size, 0), (i * cell_size, img_size), (0, 0, 0), 1)

    # Plot pickup and dropoff locations and agents
    for loc in PICKUP_LOCATIONS:
        cv2.putText(grid_img, 'P', ((loc[0] - 1) * cell_size + int(cell_size/2), (loc[1] - 1) * cell_size + int(cell_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    for loc in DROPOFF_LOCATIONS:
        cv2.putText(grid_img, 'D', ((loc[0] - 1) * cell_size + int(cell_size/2), (loc[1] - 1) * cell_size + int(cell_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    for agent in AGENT_NAMES:
        pos = state['positions'][agent]
        color = AGENT_COLORS[agent]
        cv2.circle(grid_img, ((pos[0] - 1) * cell_size + int(cell_size/2), (pos[1] - 1) * cell_size + int(cell_size/2)), int(cell_size/4), color, -1)

    return grid_img


def run_simulation_with_opencv(steps, initial_policy, changes):
    state = reset_environment()
    cv2.namedWindow('Grid World', cv2.WINDOW_AUTOSIZE)
    policy = initial_policy
    policy_changes = iter(changes)
    change_step = 500

    for step in range(steps):
        img = create_grid_image(state)
        cv2.imshow('Grid World', img)
        key = cv2.waitKey(100)

        if key & 0xFF == ord('q'):
            break

        occupied_positions = set()
        for agent in ['red', 'black', 'blue']:  # Ensuring order
            action = select_action(q_table, state, agent, policy)
            new_state = dict(state)
            apply_action(agent, action, state, occupied_positions)
            reward = compute_reward(new_state, action, state, agent)
            update_q_table(q_table, new_state, action, reward, state, agent)

        if step == change_step:
            try:
                policy = next(policy_changes)
                change_step += 8500
            except StopIteration:
                pass  # No more changes

        if step % 1000 == 0 or step == steps - 1:
            save_q_table(q_table, "q_table.json")

    cv2.destroyAllWindows()


# Set up the experiments
run_simulation_with_opencv(9000, 'PRandom', ['PRandom', 'PGreedy', 'PExploit'])

