"""
3. train.py

    References:
    [4] Kingma & Ba (2015) - Adam Optimizer
    [5] LeCun et al. (1998) - MNIST: Train/test split, loss function principles
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Pipeline(nn.Module):
    """
    Training pipeline for Pacman imitation learning

    Pipeline from [5]§Training loop:
        Hyperparameters -> 80/20 -> 2 DataLoader test & eval
        Test: forward -> loss -> backward -> update
        Eval: forward only
    """

    def __init__(self, path):
        """
        Initialize the training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.path = path

        # Load expert dataset
        self.dataset = PacmanDataset(self.path)
        print(f"Dataset: {len(self.dataset)} samples")

        # Initialize model
        self.model = PacmanNetwork()

        # CrossEntropyLoss: standard loss for multi-class classification
        # Combines Softmax + Log + NLL in one function
        self.criterion = self.model.criterion

        # Adam adapts the LR for each θ
        # Training is faster and more stable [4]
        self.learning_rate = 1e-3  # Standard Adam LR  
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Hyperparameters
        self.batch_size = 128
        self.epochs = 150
        self.test_ratio = 0.20

        # Best model tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.save_file = "pacman_model.pth"

    def _setup_dataloaders(self):
        """
        Split dataset 80/20 and create DataLoaders
        Inspired by MNIST[5] train/test split methodology
        """
        dataset_size = len(self.dataset)
        test_size = int(self.test_ratio * dataset_size)
        train_size = dataset_size - test_size

        # Use dedicated generator to ensure same split regardless of model init order
        train_set, test_set = random_split(
            self.dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train: {len(train_set)} | Test: {len(test_set)}\n")

        # DataLoader: decompose dataset in random batches
        self.loader_train = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        self.loader_test = DataLoader(
            test_set, batch_size=self.batch_size
        )

    def _evaluate(self):
        """
        Evaluate model accuracy on test set
        Inspired by MNIST[5] §Evaluate the model
        """
        correct_pred = 0
        nb_action_batch = 0

        self.model.eval()

        with torch.no_grad():
            for features, actions in self.loader_test:
                outputs = self.model.forward(features)  # Forward - Output=logits
                _, predicted = torch.max(outputs, dim=1)  # max in logits vector
                nb_action_batch += actions.size(0)  # size(0) = nbr d'elements
                correct_pred += (predicted == actions).sum().item()

        self.model.train()  # Back to training mode
        return correct_pred / nb_action_batch  # = accuracy

    def train(self):
        """
        Main training loop

        For each epoch:
            1. Forward pass: compute predictions
            2. Loss: compute cross-entropy
            3. Backward pass: compute gradients (∂loss/∂θ)
            4. Update: θnew ← θold - LR * ∇_θ L̂(θ)
        """
        print("Starting training...\n")

        # Setup data
        self._setup_dataloaders()

        # Training mode (enables dropout, batchnorm updates)
        self.model.train()

        for epoch in range(self.epochs):
            for features, actions in self.loader_train:
                # Forward & loss
                loss = self.model.cross_entropy_loss(features, actions)

                # Gradients accumulate in PyTorch, so reset them
                self.optimizer.zero_grad()

                # Backward: gradients = (∂loss/∂θ) for each θ
                loss.backward()

                # Update weights with Adam
                self.optimizer.step()

            # Evaluate on test set
            test_accuracy = self._evaluate()

            # Visualization in terminal
            print(f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {test_accuracy:.2%}")

            # Save best model (early stopping)
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_file)

        # Print final results
        print(f"\nBest model: epoch {self.best_epoch + 1} "
              f"with {self.best_accuracy:.2%} accuracy")
        print(f"Saved to: {self.save_file}")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()