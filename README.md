# Self-Supervised Learning for Label Noise Correction in Time Series Classification

This project implements the methodology described in our paper "Self-Supervised Learning as a Novel Solution to Label Noise: Insights from Sharp Wave Ripple Classification". The implementation demonstrates how self-supervised learning (SSL) can effectively address label noise in time-series classification tasks.

## Project Overview

Our approach uses self-supervised learning to improve the classification accuracy of time-series data in the presence of label noise. This implementation demonstrates the method's effectiveness using both synthetic and public datasets:

Synthetic Dataset:

SWR<sub>art</sub> Dataset (included in this repository)

Public Datasets:

2. MIT-BIH Arrhythmia Database [link from Kaggle](https://www.kaggle.com/datasets/mondejar/mitbih-database)

3. Epileptic Seizure Recognition Dataset [link from Kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)

## Key Features

- Self-supervised learning for label noise correction
- 1D CNN model for time-series classification
- Data augmentation techniques specific to time-series
- Label noise simulation and correction
- Comprehensive evaluation metrics

## Methodology

The project implements a three-stage approach:

1. **Self-Supervised Learning Stage**
   - Uses contrastive learning with temporal and contextual contrasting (E. Eldele et al., "Time-Series Representation Learning via Temporal and Contextual Contrasting," IJCAI 2021)
   - Generates weak and strong augmentations of time-series data
   - Learns robust features without relying on potentially noisy labels

2. **Label Correction Stage**
   - Utilizes learned features to identify and correct noisy labels
   - Implements clustering-based label refinement

3. **Supervised Classification Stage**
   - Trains a 1D CNN model on corrected labels
   - Employs cross-validation for robust evaluation

## Datasets

### Synthetic SWR<sub>art</sub> Dataset
- Custom synthetic dataset simulating Sharp Wave Ripples (SWRs)
- Generated using parameters derived from real SWR recordings
- Especially useful for validating SSL's performance under known noise conditions

### MIT-BIH Arrhythmia Database
- Contains ECG recordings for arrhythmia analysis
- Binary classification task: Normal vs. Arrhythmic beats
- Demonstrates method's effectiveness on real-world medical data

### Epileptic Seizure Recognition Dataset
- EEG time series data for epileptic seizure detection
- Binary classification task: Seizure vs. No-seizure
- Original dataset from UCI Machine Learning Repository


## Project Structure

```
ts-ssl-label-noise/
├── data/              
│   ├── swrart/         # Synthetic SWR dataset
│   ├── arrhythmia/     # MIT-BIH dataset
│   └── epilepsy/       # Epileptic Seizure dataset
├── src/                # Source code
│   ├── augmentation.py # data augmentations
│   ├── arrhythmia.py   # data augmentations
│   ├── epilepsy.py     # data augmentations
│   ├── swrart.py       # data augmentations
│   ├── models/         # Model architectures
│   ├── training.py     # Training functions
│   └── utils.py        # Utility functions
├── notebooks/          # Jupyter notebooks
├── requirements.txt    # Dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Saber-GRAF/ts-ssl-label-noise.git
cd ts-ssl-label-noise
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
The complete workflow for each dataset is available in separate Jupyter notebooks:

```bash
notebooks/swrart.ipynb
```

```bash
notebooks/arrhythmia.ipynb:
```

```bash
notebooks/epilepsy.ipynb:
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{graf2024ssl,
    title={Self-Supervised Learning as a Novel Solution to Label Noise: Insights from Sharp Wave Ripple Classification},
    author={Graf Saber, Meyrand Pierre, Herry Cyril, Bem Tiaza, Tsai Feng-Sheng},
    journal={Scientific Reports},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
