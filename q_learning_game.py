import pygame
import numpy as np

# Constants
GRID_SIZE = 4
NUM_ACTIONS = 4

# Function to load the Q-table from a file
def load_q_table(filename="q_table.npy"):
    return np.load(filename)

# Pygame setup for visualization
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * 50, GRID_SIZE * 50))
pygame.display.set_caption("Q-learning Game")

# Load the Q-table for testing
q_values = load_q_table()

# User input for the initial state
initial_row = int(input("Enter initial row (0-3): "))
initial_col = int(input("Enter initial column (0-3): "))
user_initial_state = (initial_row, initial_col)

# Define fire positions
fire_positions = [(1, 1), (2, 2), (3, 2)]

# Agent testing using the learned Q-values
current_position = user_initial_state
target_position = (GRID_SIZE - 1, GRID_SIZE - 1)
optimal_path = []

# Initialize Pygame window
while current_position != target_position:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * 50, i * 50, 50, 50)

            if (i, j) == current_position:
                pygame.draw.rect(screen, (0, 255, 0), rect)  # Agent (green)
                optimal_path.append(current_position)
            elif (i, j) == target_position:
                pygame.draw.rect(screen, (0, 0, 255), rect)  # Target (blue)
            elif (i, j) in fire_positions:
                pygame.draw.rect(screen, (255, 0, 0), rect)  # Fire (red)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)  # Empty (white)

    pygame.display.flip()

    # Choose an action based on the learned Q-values
    state = current_position[0] * GRID_SIZE + current_position[1]
    action = np.argmax(q_values[state])

    # Move to the next state
    if action == 0:  # Move up
        next_position = (max(current_position[0] - 1, 0), current_position[1])
    elif action == 1:  # Move down
        next_position = (min(current_position[0] + 1, GRID_SIZE - 1), current_position[1])
    elif action == 2:  # Move left
        next_position = (current_position[0], max(current_position[1] - 1, 0))
    else:  # Move right
        next_position = (current_position[0], min(current_position[1] + 1, GRID_SIZE - 1))

    current_position = next_position

    pygame.time.delay(300)  # Delay to visualize agent movement

# Display the target position
optimal_path.append(target_position)  # Include the target position in the optimal path
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        rect = pygame.Rect(j * 50, i * 50, 50, 50)
        if (i, j) == target_position:
            pygame.draw.rect(screen, (0, 0, 255), rect)  # Target (blue)
pygame.display.flip()

# Print the optimal path
print("Optimal Path:")
for step in optimal_path:
    print(step)

# Wait for the user to close the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
