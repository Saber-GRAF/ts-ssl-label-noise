import torch
import numpy as np
from tqdm import tqdm
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


def train_ssl_model(encoder, train_loader, config, device):
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epoch):
        encoder.train()
        train_loss = 0
        for _, (data, _, weak_aug, strong_aug) in enumerate(train_loader):
            data = data.to(device)
            weak_aug = weak_aug.to(device)
            strong_aug = strong_aug.to(device)

            optimizer.zero_grad()

            # Get representations
            _, features_orig = encoder(data)
            _, features_weak = encoder(weak_aug)
            _, features_strong = encoder(strong_aug)

            # Reshape features for matrix multiplication
            features_orig = features_orig.view(features_orig.size(0), -1)
            features_weak = features_weak.view(features_weak.size(0), -1)
            features_strong = features_strong.view(features_strong.size(0), -1)

            # Normalize features
            features_orig = F.normalize(features_orig, dim=1)
            features_weak = F.normalize(features_weak, dim=1)
            features_strong = F.normalize(features_strong, dim=1)

            # Compute similarity
            sim_weak = torch.mm(features_orig, features_weak.t())
            sim_strong = torch.mm(features_orig, features_strong.t())

            # Temperature parameter
            temp = 0.1

            # Compute loss
            loss = -torch.mean(
                torch.log(torch.exp(sim_weak.diag() / temp) / torch.exp(sim_weak / temp).sum(dim=1))
            ) - torch.mean(torch.log(torch.exp(sim_strong.diag() / temp) / torch.exp(sim_strong / temp).sum(dim=1)))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.num_epoch}, Loss: {train_loss/len(train_loader):.4f}")

    return encoder


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
