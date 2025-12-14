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
    def __init__(self, path):
        """
        Initialize your training pipeline.

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

        #We preferred to put the loss calculation in the architecture
        #file to logically group together the functions inherent to the model
        #self.criterion = self.model.criterion

        # Adam adapts the LR for each θ
        # training is faster and more stable
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def evaluation_mode(self, loader):
        """
            Evaluate model accuracy on dataset
            Inspired MNIST[5] §Evaluate the model
        """
        correct_pred = 0
        nb_action_batch = 0

        self.model.eval()

        with torch.no_grad():
            for features, actions in loader:  # action vector 256 expert actions
                outputs = self.model.forward(features)  # Forward - Output=logits
                _, predicted = torch.max(outputs, dim=1)  # max in logits vector
                nb_action_batch += actions.size(0)  # size(0) = nbr d'elements
                correct_pred += (predicted == actions).sum().item()

        self.model.train()  # training mode
        return correct_pred / nb_action_batch  # = accuracy

    def train(self):
        print("Beginning of the training of your network...")
        """
        Pipeline from [5]§Training loop:

        Hyperparameters -> 80/20 -> 2 DataLoader test & eval
            Test: forward -> loss -> backward -> update
            Eval: forward in
        """

        batch_size = 128
        epochs = 150
        learning_rate = 1e-3  # Standard Adam LR (était 1e-4 = trop lent)

        # 80/20 train/test split
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        train_set, test_set = random_split(self.dataset, [train_size, test_size])
        print(f"Train: {len(train_set)} | Test: {len(test_set)}\n")


        # dataloader: decompose ds in random batch
        loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(test_set, batch_size=batch_size)

        # Training
        best_accuracy = 0.0
        best_epoch = 0
        save_file = "pacman_model.pth"


        for epoch in range(epochs):
            # Train
            self.model.train()
            for features, actions in loader_train:
                loss = self.model.cross_entropy_loss(features, actions)  # forward & loss
                self.optimizer.zero_grad()  # Gradients accumulate in PyTorch
                loss.backward()  # gradients = (∂loss/∂θ) for each θ
                self.optimizer.step()  # θnew ← θold - LR ∇_θ L̂(θ) = uptade = adam

            # Evaluate
            test_accuracy = self.evaluation_mode(loader_test)

            # Vizualisation in terminal:
            print(f"Epoch {epoch + 1}/150 - Accuracy: {test_accuracy:.2%}")

            if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), save_file)  # dict containing every θ

        # print results in terminal
        print(f"\nBest model: epoch {best_epoch + 1} "
            f"with {best_accuracy:.2%} accuracy")
        print(f"Saved to: {save_file}")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
