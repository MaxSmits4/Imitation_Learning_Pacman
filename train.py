"""
3. train.py

    References:
    [4] Kingma & Ba (2015) - Adam Optimizer
    [5] LeCun et al. (1998) - MNIST: Train/test split, loss function principles
"""

import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset

# for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluation_mode(model, loader):
    """
    Evaluate model accuracy on dataset
    Inspired MNIST[5] §Evaluate the model
    """
    correct_pred = 0
    nb_action_batch = 0

    model.eval()

    with torch.no_grad():
        for features, actions in loader:  # action vector 256 expert actions
            outputs = model.forward(features)  # Forward - Output=logits
            _, predicted = torch.max(outputs, dim=1)  # max in logits vector
            nb_action_batch += actions.size(0)  # size(0) = nbr d'elements
            correct_pred += (predicted == actions).sum().item()

    model.train()  # training mode
    return correct_pred / nb_action_batch  # = accuracy


if __name__ == "__main__":
    """
    Pipeline from [5]§Training loop:

    Hyperparameters -> 80/20 -> 2 DataLoader test & eval
        Test: forward -> loss -> backward -> update
        Eval: forward in
    """

    # Hyperparameters
    batch_size = 128
    epochs = 150
    learning_rate = 1e-3  # Standard Adam LR (était 1e-4 = trop lent)

    # Load expert dataset
    dataset = PacmanDataset(path="datasets/pacman_dataset.pkl")
    print(f"Dataset: {len(dataset)} samples")

    # 80/20 train/test split
    dataset_size = len(dataset)
    test_size = int(0.20 * dataset_size)
    train_size = dataset_size - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(f"Train: {len(train_set)} | Test: {len(test_set)}\n")

    # dataloader: decompose ds in random batch
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size)

    # Initialize model
    model = PacmanNetwork()

    # Adam adapts the LR for each θ
    # training is faster and more stable
    optim = Adam(model.parameters(), lr=learning_rate)

    # Best model tracking
    best_accuracy = 0.0
    best_epoch = 0
    save_file = "pacman_model.pth"

    # Training loop
    print("Starting training...\n")

    model.train()  # Training mode

    for epoch in range(epochs):
        for features, actions in loader_train:
            loss = model.cross_entropy_loss(features, actions)  # forward & loss
            optim.zero_grad()  # Gradients accumulate in PyTorch
            loss.backward()  # gradients = (∂loss/∂θ) for each θ
            optim.step()  # θnew ← θold - LR ∇_θ L̂(θ) = uptade = adam

        test_accuracy = evaluation_mode(model, loader_test)

        # Vizualisation in terminal:
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {test_accuracy:.2%}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), save_file)  # dict containing every θ

    # print results in terminal
    print(f"\nBest model: epoch {best_epoch + 1} "
          f"with {best_accuracy:.2%} accuracy")
    print(f"Saved to: {save_file}")
