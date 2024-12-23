import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, input_size=178, num_classes=2):
        super(CNN1D, self).__init__()

        # First convolutional block with 32 filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),  # Dropout for regularization
        )

        # Second convolutional block with 64 filters
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Third convolutional block with 128 filters
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Calculate the size of flattened features
        self.flat_features = 128 * (input_size // (2 * 2 * 2))  # = 128 * 22

        # Fully connected layers
        self.fc = nn.Linear(self.flat_features, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, 178)
        x = self.conv1(x)  # Shape: (batch_size, 32, 89)
        x = self.conv2(x)  # Shape: (batch_size, 64, 44)
        x = self.conv3(x)  # Shape: (batch_size, 128, 22)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc(x))
        x = self.output(x)

        return x
