# import cv2
# import numpy as np

# class GridWorld:
#     def __init__(self, size=6, pickup_locations=None, dropoff_locations=None, agent_positions=None):
#         self.size = size
#         self.pickup_locations = pickup_locations if pickup_locations else []
#         self.dropoff_locations = dropoff_locations if dropoff_locations else []
#         self.agent_positions = agent_positions if agent_positions else {'red': (2, 2), 'blue': (4, 2), 'black': (5,2)}

#     def draw_grid(self):
#         img_size = 500  # Image size in pixels
#         cell_size = img_size // self.size

#         # Create a white square
#         img = np.full((img_size, img_size, 3), 255, np.uint8)

#         # Draw grid lines
#         for i in range(self.size + 1):
#             cv2.line(img, (0, i * cell_size), (img_size, i * cell_size), (0, 0, 0), 1)
#             cv2.line(img, (i * cell_size, 0), (i * cell_size, img_size), (0, 0, 0), 1)

#         # Mark pickup and dropoff locations
#         for loc in self.pickup_locations:
#             cv2.rectangle(img, (loc[1] * cell_size, loc[0] * cell_size), ((loc[1] + 1) * cell_size, (loc[0] + 1) * cell_size), (0, 255, 0), -1)
#         for loc in self.dropoff_locations:
#             cv2.rectangle(img, (loc[1] * cell_size, loc[0] * cell_size), ((loc[1] + 1) * cell_size, (loc[0] + 1) * cell_size), (0, 0, 255), -1)

#         # Draw agents
#         for color, position in self.agent_positions.items():
#             center = (position[1] * cell_size + cell_size // 2, position[0] * cell_size + cell_size // 2)
#             if color == 'red':
#                 cv2.circle(img, center, cell_size // 4, (0, 0, 0), -1)
#             elif color == 'blue':
#                 cv2.circle(img, center, cell_size // 4, (0, 0, 255), -1)
#             elif color == 'black':
#                 cv2.circle(img, center, cell_size // 4, (255, 0, 0), -1)

#         cv2.imshow("PD-World", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# # Example usage
# if __name__ == "__main__":
#     # Define the new pickup and dropoff locations with corrected coordinates
#     pickup_locations = [(0, 4), (1, 3), (4, 1)]
#     dropoff_locations = [(0, 0), (2, 0), (3, 4)]
#     agent_positions = {'red': (0, 2), 'blue': (2, 2), 'black': (4, 2)}
    
#     # Initialize and draw the grid world
#     world = GridWorld(size=5, pickup_locations=pickup_locations, dropoff_locations=dropoff_locations, agent_positions=agent_positions)
#     world.draw_grid()
