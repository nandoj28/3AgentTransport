from constants import GRID_SIZE, CAPACITY, PICKUP_LOCATIONS, DROPOFF_LOCATIONS

class Environment:
    """
    Environment class handles the simulation space where agents interact.
    It keeps track of agent positions, the status of blocks at various locations,
    and whether each agent is currently carrying a block.
    """
    def __init__(self):
        """
        Initializes the environment with predefined positions of agents and blocks.
        Also sets the initial carrying status of each agent to False.
        """
        self.positions = {'red': (3, 3), 'blue': (3, 5), 'black': (3, 1)}
        self.blocks = {(5, 1): 5, (4, 2): 5, (2, 5): 5, (1, 1): 0, (1, 3): 0, (5, 4): 0}
        self.carrying = {'red': False, 'blue': False, 'black': False}

    def reset(self):
        """
        Resets the environment to the initial state with default positions,
        block statuses, and carrying bool.
        """
        self.positions = {'red': (3, 3), 'blue': (3, 5), 'black': (3, 1)}
        self.blocks = {(5, 1): 5, (4, 2): 5, (2, 5): 5, (1, 1): 0, (1, 3): 0, (5, 4): 0}
        self.carrying = {'red': False, 'blue': False, 'black': False}

    def get_state(self):
        """
        Returns the current state of the environment, including agent positions,
        the status of blocks at locations, and whether agents are carrying blocks.
        """
        return {
            'positions': self.positions,
            'blocks': self.blocks,
            'carrying': self.carrying
        }

    def apply_action(self, agent, action):
        """
        Applies an action taken by an agent, updating the environment state accordingly.
        This includes moving the agent on the grid and handling pickup or dropoff actions.
        """
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
        """
        Returns a set of all currently occupied positions on the grid by any agent,
        used to prevent agents from moving into the same grid cell.
        """
        return set(self.positions.values())