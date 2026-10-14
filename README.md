# Coffee Bean Defect Classification — Thesis POC

A Python-based proof-of-concept application for evaluating trained lightweight Convolutional Neural Networks (CNNs) in the classification of dried raw coffee bean defects.

## Overview

This project provides a graphical user interface (GUI) for testing saved deep learning model checkpoints against coffee bean images.

The application allows users to select images, load trained model checkpoints, and view classification results, including predicted labels and confidence scores.

## Supported Model Architectures

- MobileNetV1
- MobileNetV2
- MobileNetV3
- EfficientNet-B0
- SqueezeNet
- ShuffleNet

## Optimizers Evaluated

- Adam
- AdamW
- Adan
- RMSprop
- SGD

## Features

- PyQt-based graphical user interface
- Saved model checkpoint loading
- Coffee bean image classification
- Prediction confidence visualization
- Comparison of model predictions and actual labels
- Classification result summaries

## Project Structure

```text
Thesis POC/
├── model_factory.py
├── test_saved_checkpoint_qt.py
├── GUI_instruction.txt
├── README.md
├── requirements.txt
├── .gitignore
├── coffee_dataset/          # Excluded from Git
├── linuxcheckpoints/        # Excluded from Git
└── *.html                   # Evaluation reports
```

## Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

## Running the Application

The project provides two testing interfaces: a command-line inference script and a graphical user interface.

### 1. Command-Line Testing

The `test_saved_checkpoint.py` script loads previously trained CNN checkpoints to classify coffee bean images.

**Using a training run directory:**

```bash
python test_saved_checkpoint.py \
    --run_dir linuxcheckpoints/example_run \
    --image coffee_dataset/example.jpg
```

The run directory must contain `best_model.pth` and `results.json`. The script reads model configuration information from the metadata file.

**Using a checkpoint directly:**

```bash
python test_saved_checkpoint.py \
    --checkpoint linuxcheckpoints/example_run/best_model.pth \
    --model squeezenet \
    --dropout 0.2 \
    --image coffee_dataset/example.jpg
```

When using `--checkpoint`, the model architecture and dropout rate must be specified manually.

**Classifying multiple images and exporting results:**

```bash
python test_saved_checkpoint.py \
    --checkpoint linuxcheckpoints/example_run/best_model.pth \
    --model squeezenet \
    --dropout 0.2 \
    --image coffee_dataset/ \
    --recursive \
    --output predictions.csv
```

### Available Command-Line Arguments

| Argument | Description |
|---|---|
| `--run_dir` | Run folder containing checkpoint and metadata |
| `--checkpoint` | Direct path to a saved checkpoint |
| `--image` | Input image or image directory |
| `--model` | CNN architecture (required with `--checkpoint`) |
| `--dropout` | Dropout value (required with `--checkpoint`) |
| `--recursive` | Search image directories recursively |
| `--output` | Save prediction results as a CSV file |
| `--device` | Select CPU or CUDA |

### Classification Outputs

The application classifies dried raw coffee bean images into three defect categories:

- **Black**
- **Broken**
- **Moldy**

For each image, the command-line tool displays its predicted class, confidence score, and the class probabilities.

Results can optionally be exported to a CSV file for further analysis.

### 2. Graphical User Interface

Run the Qt-based graphical application using:

```bash
python test_saved_checkpoint_qt.py
```

Refer to `GUI_instruction.txt` for additional usage instructions.

### Model Checkpoints

Trained model files are not included in the Git repository. Users must provide compatible checkpoint files before running inference.

The command-line tool expects checkpoints containing a `model_state_dict` compatible with the architecture defined in `model_factory.py`.

## Dataset and Model Checkpoints

The `coffee_dataset` and `linuxcheckpoints` directories are excluded from Git version control because they contain datasets and potentially large model files.

These resources must be obtained and configured separately.

## Project Status

Proof of Concept — Academic Research

## Notes

This project was developed for academic research and experimentation on lightweight CNN architectures for coffee bean defect classification.
