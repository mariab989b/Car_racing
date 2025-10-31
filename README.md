# Car Racing with Deep Q-Network (DQN)

Reinforcement learning project implementing a DQN agent to solve the Car Racing environment from Gymnasium.

## Description

This project uses the Deep Q-Network (DQN) algorithm to train an autonomous agent capable of driving a racing car on a track. The agent learns to navigate the track by maximizing its reward while avoiding going off-road.

**Environment**: Car Racing-v2 (Gymnasium/Box2D)

**Algorithm**: Deep Q-Network (DQN)

## Features

- **Experience Replay Buffer**: Storage and resampling of transitions for more stable learning
- **Target Network**: Target network updated periodically to stabilize training
- **Frame Stacking**: Stacking of 4 consecutive frames to capture temporal dynamics
- **Epsilon-Greedy Exploration**: Balance between exploration and exploitation
- **Video Recording**: Generation of MP4 videos and GIFs of the agent's performance

## Prerequisites

- Python 3.12.12
- NVIDIA GPU (recommended for training, Tesla T4 used in this project)

## Installation

Install the required dependencies:

```bash
pip install gymnasium[box2d]
pip install torch torchvision
pip install matplotlib opencv-python tqdm tensorboard
pip install imageio imageio-ffmpeg
```

## Usage

### Training the Agent

Open the `car_racing_V2.ipynb` notebook and execute the cells sequentially:

1. Install dependencies
2. Environment setup
3. Define the DQN network architecture
4. Train the agent
5. Evaluate performance

### Loading a Pre-trained Model

```python
agent.load('models/best_model.pth')
```
Please unzip before usage

### Recording a Video

```python
record_episode(
    agent, 
    mp4_path="run_car_racing.mp4", 
    num_steps=2000, 
    fps=30, 
    to_gif=True, 
    gif_path="run_car_racing.gif"
)
```

## Architecture

### Neural Network

The DQN network uses a convolutional architecture to process game frames:

- Convolutional layers for visual feature extraction
- Fully-connected layers for decision making
- Output: Q-values for each possible action

### Frame Preprocessing

- Conversion to grayscale
- Resizing
- Normalization
- Stacking of 4 consecutive frames

## Hyperparameters

Main model hyperparameters:

- **Learning rate**: Optimizer learning rate
- **Gamma**: Discount factor for future rewards
- **Epsilon**: Exploration rate (decreases over time)
- **Batch size**: Size of mini-batches for training
- **Replay buffer size**: Memory buffer capacity
- **Target network update frequency**: Frequency of target network updates

## Results

The agent achieves satisfactory performance after training, with a total score of **933.10** during the final evaluation.

Training results can be visualized with TensorBoard:

```bash
tensorboard --logdir=runs
```

## Project Structure

```
.
├── car_racing_V2.ipynb      # Main notebook
├── models/
│   └── best_model.pth       # Best saved model
├── run_car_racing.mp4       # Demo video
└── run_car_racing.gif       # Demo GIF
```

## Available Actions

The agent can choose from a set of discrete actions:

- Accelerate
- Brake
- Turn left
- Turn right
- Combinations of these actions

## Possible Improvements

- Implement Double DQN to reduce estimation bias
- Add Dueling DQN to separate state value and action advantages
- Use Prioritized Experience Replay for more efficient sampling
- Increase diversity of training tracks
- Optimize hyperparameters with automated search techniques

## References

- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Car Racing Environment](https://gymnasium.farama.org/environments/box2d/car_racing/)

## License

This project is for educational and research purposes.


**Note**: This project requires a GPU for efficient training. Training time may vary depending on available resources.
