# Spiking Neural Networks for Ordinal Regression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Official Implementation** of the paper "*Spiking Neural Networks for Ordinal Regression*"

This repository contains the **official implementation** of our latest research on **Spiking Neural Networks for Ordinal Regression Tasks**. The code provides a comprehensive framework for training SNNs on various ordinal regression benchmarks using multiple loss functions, demonstrating the effectiveness of neuromorphic computing approaches for ordinal classification problems.

## Highlights

- **First work** to systematically explore spiking neural networks for ordinal regression
- **Comprehensive evaluation** across 8 benchmark datasets (image + tabular)
- **14+ loss functions** compared in the SNN context
- **State-of-the-art** neuromorphic approach for ordinal tasks
- **Reproducible** experiments with complete codebase and training scripts

## Features

- **Spiking Neural Network Implementation**: ResNet18-based SNN architecture using leaky integrate-and-fire (LIF) neurons
- **Multiple Loss Functions**: Support for 14+ ordinal regression loss functions
- **Multi-Dataset Support**: Compatible with both image and tabular ordinal datasets
- **Flexible Training**: Configurable hyperparameters and training strategies
- **Comprehensive Metrics**: Evaluation using Accuracy, MAE, F1, QWK, Kendall's Tau, and more

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Axel-wenwu-Ma/SPIKING-NEURAL-NETWORKS-FOR-ORDINAL-REGRESSION.git
cd SPIKING-NEURAL-NETWORKS-FOR-ORDINAL-REGRESSION
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your datasets in the `data/` directory (see Dataset section below)

## Usage

### Quick Start

Train a model with default settings:
```bash
python main.py --dataset FGNET --loss UnimodalNet --REP 0 --lr 1e-3 --epochs 100
```

### Command Line Arguments

- `--dataset`: Dataset name (required)
- `--loss`: Loss function name (required)
- `--REP`: Repetition/fold number (required)
- `--lr`: Learning rate (default: 1e-3)
- `--epochs`: Number of training epochs (default: 100)
- `--batchsize`: Batch size (default: 32)
- `--model`: Model architecture (default: "snn_resnet18")
- `--lamda`: Lambda parameter for certain loss functions
- `--output`: Path to save the trained model
- `--learnable_params`: Enable learnable parameters in loss functions
- `--use_original_loss`: Use original loss formulation

### Running Experiments with Batch Script

For running multiple experiments in parallel:
```bash
bash run.sh
```

Edit the `run.sh` file to configure:
- Datasets to use
- Loss functions to evaluate
- Learning rates and lambda values
- Number of parallel jobs

## Supported Datasets

### Image Datasets
- **FGNET**: Face aging dataset
- **SMEAR2005**: Cervical cancer screening
- **HCI**: Historical Color Images

### Tabular Datasets
- **CAR**: Car evaluation
- **NEW_THYROID**: Thyroid disease
- **ABALONE5/ABALONE10**: Abalone age prediction
- **BALANCE_SCALE**: Balance scale weight

## Supported Loss Functions

### Standard Losses (no lambda parameter)
- `CrossEntropy`: Standard cross-entropy
- `POM`: Proportional Odds Model
- `OrdinalEncoding`: Ordinal encoding
- `CrossEntropy_UR`: Cross-entropy with uniform regularization
- `CDW_CE`: Class Distance Weighted CE
- `BinomialUnimodal_CE`: Binomial unimodal
- `PoissonUnimodal`: Poisson unimodal
- `UnimodalNet`: Unimodal network
- `ORD_ACL`: Ordinal adaptive consistency loss
- `VS_SL`: Variational softmax likelihood

### Lambda-based Losses (require --lamda parameter)
- `WassersteinUnimodal_KLDIV`: Wasserstein with KL divergence
- `WassersteinUnimodal_Wass`: Wasserstein distance
- `CO2`: Consistency regularization
- `HO2`: Higher-order regularization

## Model Architecture

The default model is an SNN-based ResNet18 with:
- Leaky Integrate-and-Fire (LIF) neurons
- 4 time steps
- Beta parameter: 0.9
- Dropout rate: 0.5

For tabular datasets, a multi-layer SNN MLP is used automatically.

## Evaluation Metrics

The framework computes the following metrics:
- **Accuracy (Acc)**: Classification accuracy
- **Mean Absolute Error (MAE)**: Average ordinal distance
- **F1 Score**: Macro F1 score
- **Quadratic Weighted Kappa (QWK)**: Agreement metric
- **Kendall's Tau**: Rank correlation
- **Zero Mean Error (ZME)**: Mean error
- **Negative Log Likelihood (NLL)**: Probabilistic loss
- **Unimodal Wasserstein**: Distribution consistency

## Results

Training results are saved to the `output/` directory with timestamped filenames. Logs are saved to the `log/` directory.

Best metrics are automatically computed and displayed at the end of training:
```
best_metrics in X epoch: {'Acc': 0.XX, 'MAE': X.XX, 'F1': 0.XX, ...}
```

## Project Structure

```
.
├── main.py              # Main training script
├── snn_models.py        # SNN model architectures
├── models.py            # Additional model definitions
├── model.py             # Model utilities
├── losses.py            # Loss function implementations
├── metrics.py           # Evaluation metrics
├── dataset_loader.py    # Data loading utilities
├── run.sh               # Batch experiment script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex

```

**Note**: Please update the citation with the final publication details once the paper is published.

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration opportunities regarding this official implementation:
- **Issues**: Please open an issue on [GitHub Issues](https://github.com/Axel-wenwu-Ma/SPIKING-NEURAL-NETWORKS-FOR-ORDINAL-REGRESSION/issues)
- **Author**: Wenwu Ma
- **Repository**: [https://github.com/Axel-wenwu-Ma/SPIKING-NEURAL-NETWORKS-FOR-ORDINAL-REGRESSION](https://github.com/Axel-wenwu-Ma/SPIKING-NEURAL-NETWORKS-FOR-ORDINAL-REGRESSION)

## Acknowledgments

Our codebase incorporates multiple open-source contributions. We thank the authors of Unimodal Distributions for Ordinal Regression for their excellent work on ordinal regression with CNNs.
