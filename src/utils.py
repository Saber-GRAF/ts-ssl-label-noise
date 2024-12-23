import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from augmentation import scaling, jitter, permutation


def prepare_data(data, batch_size=32, train_split=0.8):
    """
    Prepare data for training
    """
    # Convert data to tensors
    X = torch.FloatTensor(data.iloc[:, :-1].values).unsqueeze(1)  # All columns except last
    y = torch.LongTensor(data.iloc[:, -1].values)

    # Create dataset
    dataset = torch.utils.data.TensorDataset(X, y)

    # Split into train and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, labels, config):
        super().__init__()
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, -1)

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.config = config
        self.length = len(self.labels)  # Added this line

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        weak_aug = torch.from_numpy(scaling(sample.numpy(), self.config.jitter_scale_ratio)).float()

        strong_aug = torch.from_numpy(
            jitter(permutation(sample.numpy(), self.config.max_seg), self.config.jitter_ratio)
        ).float()

        return sample, label, weak_aug, strong_aug

    def __len__(self):
        return self.length


def add_label_noise(data, noise_ratio=0.2, random_state=42):
    """
    Add noise to labels by randomly flipping them based on the noise ratio

    Args:
        data (pd.DataFrame): Original dataframe with last column as label
        noise_ratio (float): Percentage of labels to flip (between 0 and 1)
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with noisy labels
    """
    # Make a copy of the original data
    noisy_data = data.copy()

    # Set random seed
    np.random.seed(random_state)

    # Get total number of samples
    n_samples = len(data)

    # Calculate number of samples to flip
    n_noise = int(n_samples * noise_ratio)

    # Randomly select indices to flip
    noise_idx = np.random.choice(n_samples, size=n_noise, replace=False)

    # Flip the selected labels (0 to 1 and 1 to 0)
    noisy_data.iloc[noise_idx, -1] = 1 - noisy_data.iloc[noise_idx, -1]

    # Calculate and print noise statistics
    original_labels = data.iloc[:, -1].values
    noisy_labels = noisy_data.iloc[:, -1].values
    actual_noise_ratio = np.mean(original_labels != noisy_labels)

    print(f"Requested noise ratio: {noise_ratio:.2%}")
    print(f"Actual noise ratio: {actual_noise_ratio:.2%}")
    print(f"Number of flipped labels: {n_noise}")
    print("\nClass distribution:")
    print("Original:")
    print(data.iloc[:, -1].value_counts(normalize=True))
    print("\nNoisy:")
    print(noisy_data.iloc[:, -1].value_counts(normalize=True))

    return noisy_data


def prepare_ssl_data(df, config):
    y = df.iloc[:, -1].values
    x = df.iloc[:, 1:-1].values

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Convert to correct shape [batch, channel, length]
    x_train = x_train.reshape(x_train.shape[0], 1, -1)
    x_val = x_val.reshape(x_val.shape[0], 1, -1)
    x_test = x_test.reshape(x_test.shape[0], 1, -1)

    # Create datasets
    train_dataset = CustomDataset(x_train, y_train, config)
    val_dataset = CustomDataset(x_val, y_val, config)
    test_dataset = CustomDataset(x_test, y_test, config)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

