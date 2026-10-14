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

```bash
python test_saved_checkpoint_qt.py
```

The application requires compatible trained checkpoints and coffee bean image data, which are not included in this repository.

## Dataset and Model Checkpoints

The `coffee_dataset` and `linuxcheckpoints` directories are excluded from Git version control because they contain datasets and potentially large model files.

These resources must be obtained and configured separately.

## Project Status

Proof of Concept — Academic Research

## Notes

This project was developed for academic research and experimentation on lightweight CNN architectures for coffee bean defect classification.
