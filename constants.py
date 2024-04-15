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
alpha = 0.45
gamma = 0.5