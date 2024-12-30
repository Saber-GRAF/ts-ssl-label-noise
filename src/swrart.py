import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to generate a simulated SWR
def generate_simulated_swr(t, A=1.0, f=10.0, phi=0.0, t0=0.0, sigma=0.1):
    """Generate a simulated SWR using a Gaussian-modulated sinusoidal wave.

    Parameters:
        t (numpy array): Time points
        A (float): Amplitude of the sine wave
        f (float): Frequency of the sine wave (Hz)
        phi (float): Phase offset (radians)
        t0 (float): Time point of the peak of the Gaussian
        sigma (float): Standard deviation of the Gaussian envelope

    Returns:
        numpy array: The simulated SWR signal
    """
    sine_wave = A * np.sin(2 * np.pi * f * t + phi)
    gaussian_envelope = np.exp(-((t - t0) ** 2) / (2 * sigma**2))
    swr = sine_wave * gaussian_envelope
    return swr


# Function to generate a more biologically realistic (chaotic) SWR
def generate_chaotic_swr(t, A=1.0, f=10.0, phi=0.0, t0=0.0, sigma=0.1, noise_level=0.2, chaos_factor=0.5):
    """Generate a more chaotic (biologically realistic) SWR using a Gaussian-modulated sinusoidal wave with added chaotic components.

    Parameters:
        t (numpy array): Time points
        A (float): Amplitude of the sine wave
        f (float): Frequency of the sine wave (Hz)
        phi (float): Phase offset (radians)
        t0 (float): Time point of the peak of the Gaussian
        sigma (float): Standard deviation of the Gaussian envelope
        noise_level (float): Standard deviation of Gaussian noise to be added
        chaos_factor (float): Factor to control the level of chaos (higher value means more chaos)

    Returns:
        numpy array: The chaotic SWR signal
    """
    # Generate the clean SWR signal
    swr = generate_simulated_swr(t, A, f, phi, t0, sigma)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, t.shape)  # generate_pink_noise(len(t), noise_level)

    # Add chaotic components (using sine and cosine functions with random frequencies and phases)

    # delta band
    chaotic_component = np.sin(2 * np.pi * np.random.uniform(1, 3) * t + np.random.uniform(0, 2 * np.pi))
    chaotic_component += np.cos(2 * np.pi * np.random.uniform(1, 3) * t + np.random.uniform(0, 2 * np.pi))

    # gamma band
    chaotic_component += np.sin(2 * np.pi * np.random.uniform(10, 20) * t + np.random.uniform(0, 2 * np.pi))
    chaotic_component += np.cos(2 * np.pi * np.random.uniform(10, 20) * t + np.random.uniform(0, 2 * np.pi))

    # Combine the SWR, noise, and chaotic components
    chaotic_swr = swr + noise + chaos_factor * chaotic_component

    return chaotic_swr


# Function to generate an SWR with improved three distinct regions: before, during, and after ripple occurrence
def generate_three_phase_swr(
    t, A=1.0, f=10.0, phi=0.0, t0=0.0, sigma=0.1, noise_level=0.1, chaos_factor=0.5, slow_wave_phi=0.0
):
    """Generate an improved SWR with three distinct regions.

    Parameters:

        t (numpy array): Time points

        A (float): Amplitude of the sine wave

        f (float): Frequency of the sine wave (Hz)

        phi (float): Phase offset (radians)

        t0 (float): Time point of the peak of the Gaussian

        sigma (float): Standard deviation of the Gaussian envelope

        noise_level (float): Standard deviation of Gaussian noise to be added

        chaos_factor (float): Factor to control the level of chaos (higher value means more chaos)

    Returns:

        numpy array: The improved SWR signal with three distinct regions

    """

    # Generate the SWR for the "during ripple" phase
    during_ripple = generate_chaotic_swr(t, A, f, phi, t0, sigma, noise_level, chaos_factor)

    # Generate the "before ripple" and "after ripple" phases (using a similar but less intense pattern)
    less_chaos = chaos_factor * 0.2
    less_noise = noise_level * 1

    before_ripple = generate_chaotic_swr(t, A * 0.5, f, phi, t0, sigma, less_noise, less_chaos)
    after_ripple = generate_chaotic_swr(t, A * 0.5, f, phi, t0, sigma, less_noise, less_chaos)

    # Combine the three phases to create the full SWR signal
    # Assuming the "during ripple" phase occurs in the middle third of the time array
    n = len(t)
    n_third = n // 3

    full_swr = np.concatenate(
        [before_ripple[:n_third], during_ripple[n_third : 2 * n_third], after_ripple[2 * n_third :]]
    )

    # Generate the slow wave component
    slow_wave_A = 0.5
    slow_wave_f = 2.0

    # Generate the slow wave component with increased amplitude and adjusted frequency
    slow_wave = slow_wave_A * np.sin(2 * np.pi * slow_wave_f * t + slow_wave_phi)

    # Combine the SWR with the more apparent slow wave
    swr_with_more_apparent_slow_wave = full_swr + slow_wave

    full_swr = swr_with_more_apparent_slow_wave / np.max(np.abs(swr_with_more_apparent_slow_wave))

    return full_swr


def plot_swr(swr, title):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(swr, color="k")
    plt.title(title)
    plt.xlabel("Time (pnts)")
    plt.xticks([0, 100, 200, 300, 400, 512])
    plt.yticks([-1, -0.50, 0, 0.50, 1])
    plt.ylabel("Amplitude")

    plt.axis("off")

    ax = plt.gca()  # Get current axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


# Function to generate an SWR for "before" and "after" learning with three distinct regions
def generate_learning_three_phase_swr(
    t, learning_state="before", A=1.0, f=10.0, phi=0.0, t0=0.0, sigma=0.1, noise_level=0.1, chaos_factor=0.5
):
    """Generate an SWR with three distinct regions based on learning state.

    Parameters:
        t (numpy array): Time points
        learning_state (str): 'before' or 'after' learning
        A (float): Amplitude of the sine wave
        f (float): Frequency of the sine wave (Hz)
        phi (float): Phase offset (radians)
        t0 (float): Time point of the peak of the Gaussian
        sigma (float): Standard deviation of the Gaussian envelope
        noise_level (float): Standard deviation of Gaussian noise to be added
        chaos_factor (float): Factor to control the level of chaos (higher value means more chaos)

    Returns:
        numpy array: The SWR signal with three distinct regions based on learning state
    """
    if learning_state == "before":
        # Higher variability and noise before learning
        A *= np.random.uniform(0.8, 1.2)
        f *= np.random.uniform(0.8, 1.2)
        noise_level *= np.random.uniform(0.06, 0.12)

    elif learning_state == "after":
        # Higher frequency, less variability, and less noise after learning
        A *= np.random.uniform(0.95, 1.05)
        f *= np.random.uniform(1.1, 1.5)  # increase in intrinsic frequency
        noise_level *= np.random.uniform(0.05, 0.1)

    slow_wave_phi = np.random.uniform(-np.pi, np.pi)

    # Generate the three-phase SWR based on the learning state
    three_phase_swr = generate_three_phase_swr(
        t, A, f, phi, t0, sigma, noise_level, chaos_factor, slow_wave_phi=slow_wave_phi
    )

    return three_phase_swr


# Function to generate a dataset of SWRs based on learning state
def generate_swr_dataset(n_samples, t, learning_state="before"):
    """Generate a dataset of SWRs based on learning state.

    Parameters:
        n_samples (int): Number of samples to generate
        t (numpy array): Time points
        learning_state (str): 'before' or 'after' learning

    Returns:
        pandas DataFrame: A DataFrame containing the generated SWRs
    """
    dataset = []
    for i in range(n_samples):
        # Generate random parameters for amplitude, frequency, noise, and ripple region length
        A = np.random.uniform(0.8, 1.2)
        f = 10
        noise_level = 0.5
        sigma = np.random.uniform(0.05, 0.2)

        # Generate the SWR based on the learning state and random parameters
        swr = generate_learning_three_phase_swr(
            t, learning_state, A, f, phi=0.0, t0=0.0, sigma=sigma, noise_level=noise_level
        )
        dataset.append(swr)

    return pd.DataFrame(dataset)

# Function to plot random SWRs from a given dataset
def plot_random_swrs(dataset, n_samples=3, title=""):
    random_samples = dataset.sample(n_samples)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5 * n_samples))
    
    for i, (_, swr) in enumerate(random_samples.iterrows()):
        axes[i].plot(np.arange(512), swr[:-1], color="k")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        # axes[i].grid(True)
        axes[i].axis('off')
    
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (pnts)")
    axes[int(n_samples / 2)].set_ylabel("Amplitude")
    plt.tight_layout()
    
    plt.xticks([0, 100, 200, 300, 400, 512])
    plt.yticks([-1, -0.50, 0, 0.50, 1])
    plt.ylim(-1, 1)

    plt.savefig('plot.svg', format='svg', bbox_inches='tight')
    plt.show()