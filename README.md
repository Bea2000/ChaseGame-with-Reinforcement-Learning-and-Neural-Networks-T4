# ChaseGame with Reinforcement Learning and Neural Networks

This project is an implementation of a reinforcement learning (RL) and neural network (NN) based game where two agents—a cat (hunter) and a mouse (prey)—interact on a grid-based map. The objective is to train the agents to pursue (cat) or escape (mouse) while navigating through obstacles. The project is structured into various modules, with clear separation between neural network training and reinforcement learning implementation.

## Context

The game map consists of obstacles (black cells) and free spaces (white cells). The cat, represented by a red marker, aims to catch the mouse, represented by a green marker. Both agents must navigate the environment and avoid obstacles to achieve their objectives. The agents use 5 different possible actions to move:

- **0:** Up
- **1:** Down
- **2:** Left
- **3:** Right
- **4:** Stay (no movement)

### Code Overview

- **`test.py`:** Script to test the agents. You can modify specific lines to use your trained agent.
- **`chase_game.py`:** Contains the game logic (Do not modify).
- **`agents/baseline.py`:** Provides baseline agents to train against (Do not modify).
- **`agents/neural.py`:** Contains the classes `NNCat` and `NNMouse` which load the neural network models.
- **`train_neural_agent.ipynb`:** Jupyter Notebook for creating, training, and exporting neural networks for both agents.
- **`train_reinforced_agent.py`:** Script to train the RL agents.
- **`agents/reinforced.py`:** File where Q-learning is implemented for the agents.

## Requirements

To run the project, ensure you have the following dependencies installed:

```bash
pip install numpy pygame tensorflow keras
```

Additionally, you'll need Python 3.8 or higher.

## Running the Project

### Testing the Agents

To run a simulation with the trained agents, execute the following command:

```bash
python test.py
```

Make sure that you've trained the agents first or are using pre-trained models stored in the `agents/data/` folder.

### Training the Neural Network Agents

The neural network agents can be trained using the `train_neural_agent.ipynb` notebook. After training, the models should be saved in the `agents/data/` folder for use in the game.

1. Open the Jupyter Notebook:

    ```bash
    jupyter notebook train_neural_agent.ipynb
    ```

2. Train both the cat and mouse agents.

3. Save the models

### Training the Reinforcement Learning Agents

The RL agents (Q-Learning) can be trained with:

```bash
python train_reinforced_agent.py
```

This script will train both the cat and mouse agents using reinforcement learning and save the Q-tables for future use.

## Example Run

Here's an example of a simple simulation:

```bash
python test.py
```

You'll see the game window open with the cat trying to catch the mouse, both navigating through obstacles.

## Key Features

- *Reinforcement Learning:* Implemented using Q-Learning. The agents learn to improve their strategy over time based on rewards.
- *Neural Networks:* Separate networks are trained for both the cat and mouse, using labeled datasets.
- *Map with Obstacles:* Both agents must navigate a map with obstacles, making the task more challenging.