import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device="cuda"):
    """
    Train the model and return training history
    """
    model = model.to(device)
    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_loss += loss.item()

        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # Store metrics in history
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_accuracy)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        # Update progress bar description with current metrics
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss/len(train_loader):.4f}",
                "train_acc": f"{train_accuracy:.4f}",
                "train_f1": f"{train_f1:.4f}",
                "val_loss": f"{val_loss/len(val_loader):.4f}",
                "val_acc": f"{val_accuracy:.4f}",
                "val_f1": f"{val_f1:.4f}",
            }
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_accuracy": val_accuracy,
                },
                "best_model.pth",
            )

    print(f"\nTraining completed! Best model saved at epoch {best_epoch}")
    print(f"Best validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    return history


def plot_training_history(history):
    """
    Plot training history metrics (accuracy and F1 score)

    Args:
        history: dictionary containing training metrics
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history["train_acc"], label="Training Accuracy")
    ax1.plot(history["val_acc"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plot F1 score
    ax2.plot(history["train_f1"], label="Training F1")
    ax2.plot(history["val_f1"], label="Validation F1")
    ax2.set_title("Model F1 Score over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # Filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def train_self_supervised(model, temporal_contr_model, train_loader, config, device):
    """
    Self-supervised learning with temporal contrastive loss.

    Args:
        model: Main model that generates features and predictions
        temporal_contr_model: Model for temporal contrastive learning
        train_loader: DataLoader containing augmented pairs
        config: Configuration object with batch_size and temperature settings
        device: torch device (cuda/cpu)
        num_epochs: Number of training epochs

    Returns:
        trained model and temporal contrastive model
    """

    # Initialize optimizers
    model_optimizer = Adam(model.parameters(), lr=0.001)
    temp_cont_optimizer = Adam(temporal_contr_model.parameters(), lr=0.001)

    # Initialize NT-Xent loss
    nt_xent_criterion = NTXentLoss(
        device=device,
        batch_size=config.batch_size,
        temperature=config.temperature,
        use_cosine_similarity=config.use_cosine_similarity,
    )

    num_epochs = config.num_epochs

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        temporal_contr_model.train()
        epoch_loss = 0

        for _, (_, _, aug1, aug2) in enumerate(train_loader):
            # Move data to device
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

            # Clear gradients
            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()

            # Get predictions and features
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # Normalize feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            # Compute temporal contrastive loss
            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # Compute NT-Xent loss
            nt_xent_loss = nt_xent_criterion(temp_cont_lstm_feat1, temp_cont_lstm_feat2)

            # Combine losses
            lambda1 = 1.0  # Weight for temporal contrastive loss
            lambda2 = 0.7  # Weight for NT-Xent loss
            total_loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_loss * lambda2

            # Backpropagation and optimization
            total_loss.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()

            epoch_loss += total_loss.item()

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    return model


def relabel_dataset(encoder, df, config, device):
    encoder = encoder.to(device)
    encoder.eval()

    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    x = x.reshape(x.shape[0], 1, -1)
    x = torch.from_numpy(x).float().to(device)

    embeddings = []
    batch_size = config.batch_size

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            _, features = encoder(batch)
            features = features.view(features.size(0), -1)
            embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    kmeans = KMeans(n_clusters=2, random_state=42)
    new_labels = kmeans.fit_predict(embeddings)

    # Create new dataframe with relabeled data
    relabeled_data = df.copy()
    relabeled_data["label"] = new_labels

    return relabeled_data
