import numpy as np
import random

# Define the environment
grid_size = 5
goal = (4, 4)
penalty = (2, 2)

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions (up, down, left, right)

# Define learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

# Define action mapping
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down

# Q-learning training loop
for episode in range(500):
    state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    
    while state != goal:
        # Choose action (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        # Get next state
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        
        # Keep within bounds
        next_state = (max(0, min(grid_size - 1, next_state[0])), max(0, min(grid_size - 1, next_state[1])))
        
        # Assign reward
        reward = 10 if next_state == goal else (-5 if next_state == penalty else -1)

        # Q-value update
        q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + alpha * (
            reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])

        state = next_state  # Move to the next state

# Display learned Q-table
print("Final Q-table:")
print(q_table)