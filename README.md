# Sudoku Solver using Reinforcement Learning

This repository implements a **Sudoku Solver** using Reinforcement Learning (RL) techniques, specifically **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)**. The goal is to train an RL agent to solve Sudoku puzzles by learning the rules and strategies of the game through interaction with the environment.

## Table of Contents

1. [Introduction](#introduction)
2. [Reinforcement Learning Overview](#reinforcement-learning-overview)
3. [Dataset](#dataset)
4. [Implementation Details](#implementation-details)
   - [Environment](#environment)
   - [PPO Model](#ppo-model)
   - [DQN Model](#dqn-model)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Contributing](#contributing)


---

## Introduction

Sudoku is a classic logic-based number placement puzzle. The objective is to fill a 9x9 grid with digits so that each column, each row, and each of the nine 3x3 subgrids contain all of the digits from 1 to 9. This project uses Reinforcement Learning to train an agent to solve Sudoku puzzles by learning from interactions with the environment.

---

## Reinforcement Learning Overview

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards. In this project:

- The **environment** is a Sudoku puzzle.
- The **agent** is the RL model (PPO or DQN).
- The **actions** are the moves the agent makes to fill in the Sudoku grid.
- The **rewards** are based on whether the moves are correct or incorrect.

---

## Dataset

The dataset used in this project is the **Sudoku Dataset** from Kaggle, which contains:
- **Quizzes**: Unsolved Sudoku puzzles.
- **Solutions**: Corresponding solved Sudoku puzzles.

The dataset is loaded in chunks to handle memory efficiently during training.

---

## Implementation Details

### Environment

The Sudoku environment is implemented using the `gymnasium` library. Key features include:
- **Observation Space**: A 9x9 grid representing the current state of the Sudoku puzzle.
- **Action Space**: Discrete actions representing possible moves (row, column, and number).
- **Rewards**: Positive rewards for correct moves and negative rewards for incorrect moves.
- **Termination**: The episode ends when the puzzle is solved or the maximum number of steps is reached.

### PPO Model

The **Proximal Policy Optimization (PPO)** model is implemented using the `stable-baselines3` library. Key features include:
- **Policy Network**: A neural network with shared layers for policy and value functions.
- **Training**: The model is trained in chunks to manage memory usage and save checkpoints periodically.
- **Evaluation**: The model is evaluated on a set of Sudoku puzzles to measure success rate and average reward.

### DQN Model

The **Deep Q-Network (DQN)** model is also implemented using the `stable-baselines3` library. Key features include:
- **Q-Network**: A neural network that approximates the Q-value function.
- **Training**: The model is trained with a callback to log progress and save checkpoints.
- **Evaluation**: The model is evaluated on a set of Sudoku puzzles to measure success rate and average reward.

---

## Training and Evaluation

### Training
- The models are trained on the Sudoku dataset using the defined environment.
- Training is done in smaller chunks to manage memory and computational resources.
- Checkpoints are saved periodically to allow for resuming training.

### Evaluation
- The trained models are evaluated on a set of Sudoku puzzles to measure their performance.
- Metrics include **success rate** (percentage of puzzles solved correctly) and **average reward**.

---

## Results

### PPO Model
- Achieves a **high success rate** and **average reward** after training.
- Solves Sudoku puzzles efficiently by learning valid moves and strategies.

### DQN Model
- Demonstrates competitive performance in solving Sudoku puzzles.
- Provides a balance between exploration and exploitation during training.

---

## Usage

1. **Install Dependencies**:
   ```bash
   pip install stable-baselines3[extra] gymnasium pandas numpy
   ```

2. **Download Dataset**:
   ```bash
   kaggle datasets download bryanpark/sudoku
   unzip sudoku.zip
   ```

3. **Train the Model**:
   - Run the script to train the PPO or DQN model:
     ```bash
     python train_sudoku_ppo.py
     python train_sudoku_dqn.py
     ```

4. **Evaluate the Model**:
   - Use the evaluation script to test the trained model on Sudoku puzzles:
     ```bash
     python evaluate_sudoku.py
     ```

5. **Save and Load Models**:
   - Models are saved periodically during training and can be loaded for further evaluation or inference.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

This README provides an overview of the Sudoku Solver project, including setup instructions, usage, and results. For more details, refer to the code and comments in the scripts.
