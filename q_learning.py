import numpy as np
import random

# Testing code for Q-learning in a grid environment
# This code implements a simple Q-learning algorithm to navigate a 4x4 grid environment
# The agent learns to reach a goal state while avoiding penalties for non-goal states

# --- パラメータ設定 ---
grid_size = 4
goal_state = (3, 3)
alpha = 0.1       # 学習率
gamma = 0.9       # 割引率
epsilon = 0.2     # ε-greedyの探索率
episodes = 1000

actions = ["up", "down", "left", "right"]
q_table = np.zeros((grid_size, grid_size, len(actions)))

# --- 環境関数 ---
def get_next_state(state, action):
    x, y = state
    if action == "up" and x > 0:
        x -= 1
    elif action == "down" and x < grid_size - 1:
        x += 1
    elif action == "left" and y > 0:
        y -= 1
    elif action == "right" and y < grid_size - 1:
        y += 1
    return (x, y)

def get_reward(state):
    return 1 if state == goal_state else -0.01

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        x, y = state
        return actions[np.argmax(q_table[x, y])]

# --- Q学習アルゴリズム ---
def train():
    for episode in range(episodes):
        state = (0, 0)
        while state != goal_state:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)

            x, y = state
            a = actions.index(action)
            next_x, next_y = next_state

            best_next_q = np.max(q_table[next_x, next_y])
            q_table[x, y, a] += alpha * (reward + gamma * best_next_q - q_table[x, y, a])

            state = next_state

    print("Q学習完了")

# --- 実行テスト ---
def test():
    state = (0, 0)
    path = [state]
    while state != goal_state:
        x, y = state
        action_index = np.argmax(q_table[x, y])
        action = actions[action_index]
        state = get_next_state(state, action)
        path.append(state)
    print("テスト経路：", path)

# --- main関数 ---
def main():
    train()
    test()

if __name__ == "__main__":
    main()
