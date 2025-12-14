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
        super().__init__()

        self.path = path
        self.dataset = PacmanDataset(self.path)
        self.model = PacmanNetwork()

        # We preferred to put the loss calculation in the architecture 
        # file to logically group together the functions inherent to the model
        #self.criterion = self.model.criterion 
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        print("Beginning of the training of your network...")

        # Split 80/20
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_set, test_set = random_split(
            self.dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # DataLoaders
        loader_train = DataLoader(train_set, batch_size=128, shuffle=True)
        loader_test = DataLoader(test_set, batch_size=128)

        # Training
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(150):
            # Train
            self.model.train()
            for features, actions in loader_train:
                loss = self.model.cross_entropy_loss(features, actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluate
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for features, actions in loader_test:
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs, 1)
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/150 - Accuracy: {accuracy:.2%}")

            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch
                torch.save(self.model.state_dict(), "pacman_model.pth")

        print(f"\nBest model: epoch {best_epoch + 1} with {best_acc:.2%} accuracy")
        print("Model saved !")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
