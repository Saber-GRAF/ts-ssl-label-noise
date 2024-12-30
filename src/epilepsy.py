import seaborn as sns
import matplotlib.pyplot as plt


def plot_sample_signals(dataset):
    sns.set_theme(style="whitegrid", context="talk")

    unique_classes = sorted(dataset["y"].unique())

    fig, axes = plt.subplots(nrows=len(unique_classes), ncols=1, figsize=(9, 4 * len(unique_classes)))
    fig.suptitle("EEG Signals by Class (One Sample per Class)", fontsize=20, fontweight="bold")

    for i, class_label in enumerate(unique_classes):
        class_data = dataset[dataset["y"] == class_label]

        random_sample = class_data.sample(n=1).drop(columns="y")  # Exclude the 'y' column

        signal_data = random_sample.melt(var_name="Time", value_name="Signal Value")
        sns.lineplot(
            data=signal_data,
            x="Time",
            y="Signal Value",
            ax=axes[i],
            color=sns.color_palette("husl", len(unique_classes))[i],  # Distinct color for each class
        )

        axes[i].set_title(f"Class {class_label}", fontsize=16, fontweight="bold")
        axes[i].set_ylabel("Signal Value", fontsize=12)
        axes[i].set_xlabel("Time", fontsize=12)
        axes[i].tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    sns.despine()

    plt.show()


def plot_class_distribution(data):
    sns.set_theme(style="whitegrid", context="talk")

    # Create the count plot
    plt.figure(figsize=(9, 5))
    ax = sns.countplot(x=data["y"], palette="viridis", hue=data["y"], legend=False)

    ax.set_title("Class Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # Add count annotations on the bars
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    sns.despine()

    plt.show()
