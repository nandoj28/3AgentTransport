from constants import GRID_SIZE, CAPACITY, PICKUP_LOCATIONS, DROPOFF_LOCATIONS

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