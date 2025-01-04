

# MRI Brain Tumor Detection Project

## Overview

This project implements a brain tumor detection system using **Gabor filters** and a **3D U-Net architecture**. The model is trained on the **BraTS 2020 dataset**, which contains multimodal MRI images (FLAIR, T1, T1CE, T2) and corresponding segmentation masks. The goal is to accurately segment brain tumors from MRI scans.

## Table of Contents

1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Gabor Filter Visualization](#gabor-filter-visualization)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
5. [License](#license)

## Requirements

To run this project, you will need the following Python packages:

- `numpy`
- `opencv-python`
- `matplotlib`
- `nibabel`
- `tensorflow`
- `keras`
- `sklearn`
- `segmentation-models-3D`

You can install the required packages using pip:

```bash
pip install numpy opencv-python matplotlib nibabel tensorflow keras scikit-learn segmentation-models-3D
```

## Dataset

This project uses the **BraTS 2020 dataset**, which can be downloaded from the BraTS Challenge website. The dataset contains the following modalities:

- FLAIR
- T1
- T1CE
- T2
- Segmentation masks

Make sure to place the dataset in the appropriate directory as specified in the code.

## Installation

1. Clone this repository or download the notebook file.
2. Ensure that the BraTS dataset is downloaded and placed in the specified directory.
3. Open the notebook in **Google Colab** or **Jupyter Notebook**.

## Usage

### Load and Preprocess Data

The code loads the MRI images and corresponding masks, normalizes the pixel values, and prepares the data for training.

### Gabor Filter Application

The code applies **Gabor filters** to the MRI images to enhance feature extraction.

### Model Definition

A **3D U-Net** model is defined for semantic segmentation of brain tumors.

### Training

The model is trained using the training dataset, and the training history is recorded.

### Evaluation

The model's performance is evaluated using metrics such as **Intersection over Union (IoU)** and accuracy.

## Gabor Filter Visualization

The project includes a section that visualizes Gabor filters with varying parameters (theta and gamma). This helps in understanding how different filter configurations affect the feature extraction process.

Example code to visualize Gabor filters:

```python
# Example of creating Gabor filters
ksize = 50
sigma = 3
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
gammas = [1, 0.5, 0.01]

# Code to visualize Gabor filters
```

## Model Training

The model is trained using the following parameters:

- **Batch size**: 2
- **Learning rate**: 0.0001
- **Number of epochs**: 100

The training process includes data augmentation and uses a custom loss function that combines **Dice loss** and **focal loss** to handle class imbalance.

Example code for model training:

```python
# Example of model training
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)
```

## Model Evaluation

After training, the model's performance is evaluated on a validation dataset. The code calculates the **Mean IoU** and visualizes the predictions against the ground truth masks.

Example code for model evaluation:

```python
# Example of model evaluation
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
```

## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.
