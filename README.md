# CNN Image Classification Project

This repository contains a CNN-based image classification workflow for a small image dataset with labeled classes. The notebook explores the dataset, preprocesses images, builds and trains a CNN model, evaluates the results, and explains the business value of the solution.

## Project Structure

- `notebook.ipynb` - main analysis and implementation notebook
- `part_2_cnn_computer_vision/` - dataset folder
  - `labels.csv` - image filename labels
  - `images/` - image folders organized by class
    - `dent/`
    - `normal/`
    - `scratch/`
    - `stain/`
- `results/requirements.txt` - required Python packages

## Problem Statement

This is an image classification task. Each image belongs to exactly one category, and the goal is to predict the image class from one of the available labels.

The problem is NOT:
- object detection
- segmentation

## Notebook Workflow

1. **Dataset exploration**
   - Load `labels.csv` using `pandas`
   - Display class counts and sample images
   - Confirm data organization and class distribution

2. **Image preprocessing**
   - Resize images to `128x128`
   - Normalize pixel values to the range `[0, 1]`
   - Apply data augmentation for training

3. **CNN model creation**
   - Build a sequential CNN using `tensorflow.keras`
   - Use multiple convolution + pooling layers
   - Flatten features and add dense layers with dropout
   - Compile with `adam` optimizer and `categorical_crossentropy`

4. **Model training and evaluation**
   - Split the dataset into training, validation, and test sets
   - Train for 10 epochs
   - Plot accuracy and loss curves
   - Evaluate on the held-out test set
   - Generate a confusion matrix and sample predictions

## Dependencies

Install the required dependencies from `results/requirements.txt`:

```bash
pip install -r results/requirements.txt
```

The notebook imports packages including:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `Pillow`

## How to Run

1. Install dependencies with pip.
2. Open `notebook.ipynb` in Jupyter Notebook or JupyterLab.
3. Execute the cells in order.

## Business Use Case

A CNN image classification system like this can be applied to healthcare image analysis, such as classifying X-ray or MRI scans into categories like:
- Normal
- Diseased
- Abnormal

Benefits include faster diagnosis, reduced clinician workload, early detection, and improved accuracy.

## Notes

- The notebook uses a CSV file plus folder structure for image labels.
- The model is designed for multi-class classification with a softmax output.
- Image augmentation helps the model generalize better.
