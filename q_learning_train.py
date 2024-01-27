import numpy as np
import os
import pygame

GRID_SIZE = 4
NUM_ACTIONS = 4
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9
NUM_EPISODES = 5000

# Initialize Q-values
q_values = np.zeros((GRID_SIZE * GRID_SIZE, NUM_ACTIONS))

# Define the grid with new fire and target positions
grid = np.zeros((GRID_SIZE, GRID_SIZE))
fire_positions = [(1, 1), (2, 2), (3, 2)]  # Updated fire positions
target_position = (GRID_SIZE - 1, GRID_SIZE - 1)

for fire_pos in fire_positions:
    grid[fire_pos] = -100

grid[target_position] = 100

# Pygame setup for visualization
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * 50, GRID_SIZE * 50))
pygame.display.set_caption("Q-learning Training")

# Function to get the state index for a given position
def get_state_index(position):
    return position[0] * GRID_SIZE + position[1]

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(q_values[state])

# Function to update Q-values using Q-learning
def update_q_values(state, action, reward, next_state):
    best_next_action = np.argmax(q_values[next_state])
    q_values[state, action] = (1 - ALPHA) * q_values[state, action] + \
                             ALPHA * (reward + GAMMA * q_values[next_state, best_next_action])

# Function to save the Q-table to a file
def save_q_table(q_values, filename="q_table.npy"):
    np.save(filename, q_values)

# Function to display the grid, agent, and fire positions
def display_grid(agent_position):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * 50, i * 50, 50, 50)

            if (i, j) == agent_position:
                pygame.draw.rect(screen, (0, 255, 0), rect)  # Agent (green)
            elif (i, j) in fire_positions:
                pygame.draw.rect(screen, (255, 0, 0), rect)  # Fire (red)
            elif (i, j) == target_position:
                pygame.draw.rect(screen, (0, 0, 255), rect)  # Target (blue)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)  # Empty (white)

    pygame.display.flip()

# Main Q-learning training loop with visualization
for episode in range(NUM_EPISODES):
    # Choose a random initial position within the grid
    initial_position = np.random.randint(0, GRID_SIZE, size=2)
    current_position = tuple(initial_position)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    while current_position != target_position and current_position not in fire_positions:
        display_grid(current_position)
        pygame.time.delay(3)  # Delay to visualize agent movement
        pygame.event.pump()

        # Choose an action
        state = get_state_index(current_position)
        action = choose_action(state)

        # Move to the next state
        if action == 0:  # Move up
            next_position = (max(current_position[0] - 1, 0), current_position[1])
        elif action == 1:  # Move down
            next_position = (min(current_position[0] + 1, GRID_SIZE - 1), current_position[1])
        elif action == 2:  # Move left
            next_position = (current_position[0], max(current_position[1] - 1, 0))
        else:  # Move right
            next_position = (current_position[0], min(current_position[1] + 1, GRID_SIZE - 1))

        # Get the reward
        reward = grid[next_position]

        # Update Q-values
        next_state = get_state_index(next_position)
        update_q_values(state, action, reward, next_state)

        current_position = next_position

# Save the Q-table after training
save_q_table(q_values)
pygame.quit()
