# XAI-SVHN: Interpreting Convolutional Neural Networks on Street View House Numbers

**Exploring neural network interpretability on the SVHN dataset using CNN and XAI tools.**

---

## 📌 Overview
This project aims to:
- Train a CNN on the **SVHN dataset** (Street View House Numbers).
- Apply **XAI techniques** (SHAP, Grad-CAM, Feature Visualization) to interpret model decisions.
- Compare interpretability and performance with a small Vision Transformer (ViT).

---

## 📂 Project Structure

```bash
xai-svhn/
├── data/            # SVHN dataset (train/test)
├── models/          # CNN and ViT architectures
├── notebooks/       # Training, evaluation, and XAI analysis
├── results/         # Visualizations, metrics, and explanations
├── environment.yml # Conda environment dependencies
└── README.md        # This file
```

## 🛠 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/edgar-demeude/xai-svhn.git
   cd xai-svhn
   ```

2. Create a conda environment and install dependencies:
    ```bash
    conda env create -f environment.yml
    conda activate xai-svhn
    ```

3. Download the SVHN dataset (automatically handled by PyTorch in the notebooks).

## 🚀 Usage
- **Train the CNN:** Open `notebooks/training.ipynb` and run all cells.
- **Generate explanations:** Use `notebooks/xai_analysis.ipynb` for SHAP, Grad-CAM, and feature visualization.
- **Compare models:** See `results/` for performance metrics and interpretability visualizations.

## 📊 Results

- **Model performance** (accuracy, loss curves) in `results/metrics/`.
- **Interpretability visualizations** (neuron activations, attention maps) in `results/visualizations/`.

## 🔍 XAI Techniques Used

- **SHAP:** Feature importance analysis.
- **Grad-CAM:** Highlighting important regions in input images.
- **Feature Visualization:** Understanding what individual neurons detect.