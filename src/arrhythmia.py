import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def get_paired_files(directory):
    """
    Get pairs of CSV and TXT files that share the same base number.

    Args:
        directory (str): Path to the directory containing the files

    Returns:
        list: List of tuples containing (csv_path, txt_path) pairs
    """
    # Get all files in directory
    files = os.listdir(directory)

    # Separate CSV and TXT files
    csv_files = sorted([f for f in files if f.endswith(".csv")])
    txt_files = sorted([f for f in files if f.endswith("annotations.txt")])

    # Create pairs based on the number in the filename
    pairs = []
    for csv_file in csv_files:
        # Extract the number from csv filename (e.g., '213' from '213.csv')
        number = csv_file.split(".")[0]

        # Find corresponding txt file
        txt_file = f"{number}annotations.txt"
        if txt_file in txt_files:
            # Create full paths
            csv_path = os.path.join(directory, csv_file)
            txt_path = os.path.join(directory, txt_file)
            pairs.append((csv_path, txt_path))

    return pairs


def plot_annotation_class_distribution(annotation):
    """
    Plot the distribution of heart beat classes in the MIT-BIH Arrhythmia Dataset

    Args:
        file_path (str): Path to the annotations file
    """
    # Read the annotation file
    df = pd.read_csv(
        annotation, sep=r"\s+", names=["Time", "Sample #", "Type", "Sub", "Chan", "Num", "Aux"], skiprows=1
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Original class distribution
    sns.countplot(data=df, x="Type", ax=ax1)
    ax1.set_title("Distribution of Original Beat Types")
    ax1.set_xlabel("Beat Type")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    # Add count labels on top of bars
    for p in ax1.patches:
        ax1.annotate(
            f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height()), ha="center", va="bottom"
        )

    # Create binary labels
    df["Binary_Label"] = (df["Type"] != "N").astype(int)
    binary_counts = df["Binary_Label"].value_counts()

    # Plot 2: Binary class distribution
    colors = ["lightgreen", "salmon"]
    binary_counts.plot(kind="bar", ax=ax2, color=colors)
    ax2.set_title("Binary Classification Distribution")
    ax2.set_xlabel("Class (0=Normal, 1=Abnormal)")
    ax2.set_ylabel("Count")

    # Add count labels on top of bars
    for i, v in enumerate(binary_counts):
        ax2.text(i, v, str(v), ha="center", va="bottom")

    # Calculate percentages
    total = len(df)
    normal_pct = (binary_counts[0] / total) * 100 if 0 in binary_counts else 0
    abnormal_pct = (binary_counts[1] / total) * 100 if 1 in binary_counts else 0

    # Add percentage labels
    ax2.text(-0.1, binary_counts[0] / 2, f"{normal_pct:.1f}%", ha="center")
    ax2.text(0.9, binary_counts[1] / 2, f"{abnormal_pct:.1f}%", ha="center")

    # Adjust layout
    plt.tight_layout()

    # Print summary statistics
    print("\nClass Distribution Summary:")
    print(f"Total beats: {total}")
    print(f"Normal beats (Class 0): {binary_counts[0]} ({normal_pct:.1f}%)")
    print(f"Abnormal beats (Class 1): {binary_counts[1]} ({abnormal_pct:.1f}%)")

    # Show plot
    plt.show()


def read_annotation_file(file_path):
    """Read and preprocess annotation file"""
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        names=["Time", "Sample #", "Type", "Sub", "Chan", "Num", "Aux"],
        skiprows=1,
        engine="python",
    )
    # Create binary labels
    df["label"] = (df["Type"] != "N").astype(int)
    # Rename 'Sample #' to 'pos'
    df = df.rename(columns={"Sample #": "pos"})
    return df[["Time", "pos", "label"]]


def read_signal_file(file_path):
    """Read and preprocess signal file"""
    data = pd.read_csv(file_path)
    # Get only the first signal (second column)
    signal_data = pd.DataFrame(
        {
            "pos": data.iloc[:, 0],  # First column (sample numbers)
            "signal": data.iloc[:, 1],  # Second column (first signal)
        }
    )
    return signal_data


def extract_beats(signal_data, annotations_df, window_size=89):
    """
    Extract beats from signal data using annotation positions

    Args:
        signal_data (pd.DataFrame): DataFrame containing ECG signal
        annotations_df (pd.DataFrame): DataFrame containing annotations
        window_size (int): Number of samples before and after the R-peak

    Returns:
        list: List of dictionaries containing beat information
    """
    beats = []

    for _, row in annotations_df.iterrows():
        pos = row["pos"]
        label = row["label"]

        # Check if we can extract a complete window
        if pos - window_size < 0 or pos + window_size >= len(signal_data):
            continue

        try:
            beat_data = {
                "pos": pos,
                "label": label,
                "label_text": "Normal beat" if label == 0 else "Arrhythmic beat",
                "signal": signal_data["signal"][pos - window_size : pos + window_size].values,
                "samples": signal_data["pos"][pos - window_size : pos + window_size].values,
            }
            beats.append(beat_data)
        except Exception as e:
            print(f"Error processing beat at position {pos}: {str(e)}")
            continue

    return beats


def process_file_pair(signal_path, annotation_path, window_size=178):
    """Process a pair of signal and annotation files"""
    try:
        # Read files
        annotations_df = read_annotation_file(annotation_path)
        signal_data = read_signal_file(signal_path)

        # Extract beats
        beats = extract_beats(signal_data, annotations_df, window_size)

        return beats

    except Exception as e:
        print(f"Error processing files {signal_path} and {annotation_path}: {str(e)}")
        return None


def process_directory(directory, window_size=89):
    all_beats = []
    file_pairs = get_paired_files(directory)

    for signal_path, annotation_path in file_pairs:
        print(f"\nProcessing: {Path(signal_path).stem}")
        beats = process_file_pair(signal_path, annotation_path, window_size)

        if beats:
            all_beats.extend(beats)
            print(f"Extracted {len(beats)} beats from {Path(signal_path).stem}")

    print(f"\nTotal beats extracted: {len(all_beats)}")
    print(f"Normal beats: {sum(1 for beat in all_beats if beat['label'] == 0)}")
    print(f"Arrhythmic beats: {sum(1 for beat in all_beats if beat['label'] == 1)}")

    return all_beats


def save_beats_to_csv(beats, output_file):
    """
    Save beats to CSV file with signal points as columns and label as final column

    Args:
        beats (list): List of beat dictionaries
        output_file (str): Path to save the CSV file
    """
    # Create list to store rows
    rows = []

    # Process each beat
    for beat in beats:
        # Get signal values and label
        signal_values = beat["signal"]
        label = beat["label"]

        # Create row with signal values and label
        row = list(signal_values)
        row.append(label)
        rows.append(row)

    # Create column names
    columns = [i for i in range(len(beats[0]["signal"]))]
    columns.append("label")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Saved {len(beats)} beats to {output_file}")
    print(f"CSV shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())


def create_balanced_sample(df, n_samples=10000, random_state=42):
    """
    Create balanced dataset by randomly sampling n_samples from each class

    Args:
        df: DataFrame with signal columns and label
        n_samples: Number of samples to take from each class
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame with n_samples * 2 total samples
    """
    # Separate data by class
    normal_beats = df[df["label"] == 0]
    arrhythmic_beats = df[df["label"] == 1]

    # Print original distribution
    print("Original distribution:")
    print(f"Normal beats: {len(normal_beats)}")
    print(f"Arrhythmic beats: {len(arrhythmic_beats)}")

    # Random sampling from each class
    sampled_normal = normal_beats.sample(n=n_samples, random_state=random_state)
    sampled_arrhythmic = arrhythmic_beats.sample(n=n_samples, random_state=random_state)

    # Combine samples
    balanced_df = pd.concat([sampled_normal, sampled_arrhythmic])

    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Print new distribution
    print("\nNew distribution:")
    print(balanced_df["label"].value_counts())

    return balanced_df


def plot_dataset_distribution(df):
    """
    Plot and print the distribution of labels in the dataset.

    Args:
        df: DataFrame containing a 'label' column
    """
    # Calculate counts and percentages
    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    percentages = (label_counts / total * 100).round(2)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create bar plot
    ax = sns.barplot(x=label_counts.index, y=label_counts.values)

    # Customize plot
    plt.title("Distribution of ECG Beat Labels", pad=20)
    plt.xlabel("Label (0: Normal, 1: Arrhythmic)")
    plt.ylabel("Count")

    # Add count labels on top of bars
    for i, v in enumerate(label_counts.values):
        plt.text(i, v, f"{v}\n({percentages[i]}%)", ha="center", va="bottom")

    # Print summary
    print("\nLabel Distribution Summary:")
    print(f"Total beats: {total}")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} beats ({percentages[label]}%)")

    plt.tight_layout()
    plt.show()


def plot_beat(df, row_index):
    """
    Plot a single ECG beat from the dataset

    Args:
        df: DataFrame containing the beats
        row_index: Index of the beat to plot
    """
    # Get the beat data
    beat = df.iloc[row_index]

    # Extract signal values (all columns except label)
    signal = beat[:-1].values  # Exclude the label column

    # Create time points (x-axis)
    time_points = np.arange(len(signal))

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot the beat
    plt.plot(time_points, signal, "b-", linewidth=2)

    # Add labels and title
    plt.xlabel("Sample Point")
    plt.ylabel("Amplitude")
    plt.title(f'ECG Beat (Row {row_index}, Label {int(beat["label"])})')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add label annotation
    label_text = "Normal Beat" if beat["label"] == 0 else "Arrhythmic Beat"
    plt.text(
        0.02,
        0.98,
        label_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.show()

def standardize_beats(df):
    """
    Standardize ECG beats data (zero mean and unit variance)
    
    Args:
        df: DataFrame with beat signals and label column
        
    Returns:
        DataFrame with standardized beats
    """
    # Separate features (signals) and label
    X = df.iloc[:, :-1]  # All columns except label
    y = df['label']
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create new DataFrame with standardized data
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['label'] = y
    
    return df_scaled