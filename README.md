# Pacman Imitation Learning

Supervised learning project to train a Pacman agent to imitate an expert using an MLP neural network.

## Overview

This project implements an imitation learning system where a neural network learns to play Pacman by imitating expert actions. The network takes the game state (32 features) as input and predicts one of 5 possible actions: NORTH, SOUTH, EAST, WEST, STOP.

**Results:**
- Test set accuracy: ~87-88%
- Architecture: MLP (32 → 128 → 64 → 32 → 5)
- Dataset: 15,018 state-action examples
- Training: ~150 epochs

## Installation

**Requirements:** Python 3.8+, PyTorch 2.0+

```bash
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python train.py
```

**Watch the agent play:**
```bash
python run.py
```

**Generate submission file:**
```bash
python write_submission.py
```

## Project Structure

```
pacman_imitation_learning/
├── datasets/
│   ├── pacman_dataset.pkl      # Training dataset
│   └── pacman_test.pkl         # Test dataset
├── pacman_module/              # Game engine (do not modify)
├── data.py                     # Feature engineering + Dataset
├── architecture.py             # MLP neural network
├── train.py                    # Training script
├── pacmanagent.py              # Agent using the model
├── run.py                      # Game visualization
├── write_submission.py         # CSV generation
├── pacman_model.pth            # Trained model
└── submission.csv              # Predictions for Gradescope
```

## Network Architecture

```
Input (32)  →  Linear+BN+GELU+Dropout (128)
            →  Linear+BN+GELU+Dropout (64)
            →  Linear+BN+GELU+Dropout (32)
            →  Linear (5)
```

Each hidden layer: Linear → BatchNorm → GELU → Dropout (0.3)

## Features (32)

| Category | Count | Description |
|----------|:-----:|-------------|
| Position | 2 | Pacman's normalized X and Y |
| Ghost | 7 | Direction X/Y to ghost, Manhattan distance, directional flags (N/S/E/W) |
| Food | 9 | Food count, average distance, direction and distance to closest, directional flags |
| Maze | 9 | Distance to 4 walls, distance to center X/Y, corner/corridor/open space flags |
| Legal | 5 | Flags for each legal action |

All features are normalized (positions by maze dimensions, distances by maximum Manhattan distance).

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 1e-3 |
| Epochs | 150 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Train/test split | 80/20 |

## Customization

**Modify the architecture** (in `architecture.py`):
```python
PacmanNetwork(
    input_features=32,
    num_actions=5,
    hidden_dims=[128, 64, 32],
    activation=nn.GELU(),
    dropout=0.3
)
```

**Add features** (in `data.py`): modify `state_to_tensor()` and update `input_features` in `architecture.py`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No convergence | Check normalization, reduce learning rate |
| Accuracy ~50-60% | Check dataset and action mapping |
| Overfitting | Increase dropout, reduce network size |
| No display | Install tkinter (`sudo apt-get install python3-tk`) |

## FAQ

**Why an MLP instead of a CNN?**
The input is a 1D vector of manually extracted features with no 2D spatial structure. An MLP is well-suited for this type of tabular data.

**Why GELU?**
GELU is a smoother activation than ReLU, standard in modern architectures (BERT, GPT). It weights inputs by their value rather than gating them by their sign.

**Why normalize?**
Neural networks converge better when all features are on the same scale. Without normalization, large values dominate the gradient.

**Why does accuracy plateau at ~88%?**
In some situations, multiple actions are equally valid. The network may predict differently from the expert while still being correct.

**Does the model generalize?**
Yes. Features are normalized by maze size, so the model learns general patterns (flee ghosts, go toward food) rather than memorizing a specific layout.

## Documentation

For a detailed explanation, see `explication.txt`.

## References

- [1] Ioffe & Szegedy (2015) - Batch Normalization
- [2] Hendrycks & Gimpel (2016) - GELU
- [3] Srivastava et al. (2014) - Dropout
- [4] Kingma & Ba (2015) - Adam Optimizer
- [5] LeCun et al. (1998) - MNIST (classification principles)

See `Bibliography.txt` for full citations.