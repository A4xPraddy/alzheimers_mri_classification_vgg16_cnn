# Alzheimer’s Disease Classification using VGG16

A deep learning–based system for classifying Alzheimer’s Disease (AD) stages using structural MRI scans.
This project uses a fine-tuned VGG16 CNN, ResNet50 architecture to classify MRI images into four stages of Alzheimer’s Disease.

# Overview

This project implements a transfer-learning pipeline for multi-class Alzheimer’s Disease classification using MRI images.
The model classifies images into:

#### Very Mild Demented

#### Mild Demented

#### Moderate Demented

The model uses preprocessing, augmentation, dropout, and early stopping to improve robustness and reduce overfitting.

# Datasets Used

#### AMD Dataset – Retinal imaging dataset used for Age-related Macular Degeneration studies.

#### UK Biobank (UKB) – A large-scale biomedical dataset with MRI, genetics, and clinical records.

#### These datasets together provide strong imaging variability for effective training.

# Model Architecture

The model is built on VGG16 (ImageNet pre-trained) with the following modifications:

Removed fully connected top layers

Added Global Average Pooling

Added Dense layers with ReLU

Added Dropout for regularization

Final Softmax layer for 4-class classification

This architecture provides a strong balance between accuracy and interpretability.

# Training Setup

#### Framework: TensorFlow / Keras

#### Optimizer: Adam with LR scheduling

#### Loss: Categorical Cross-Entropy

#### Regularization Techniques: Dropout -> Data Augmentation -> Early Stopping

#### Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

# Results

Strong performance in detecting Non-Demented and Very Mild Demented categories

Some confusion between Mild and Moderate Demented (expected due to overlapping MRI features)

Training/validation curves indicate reduced overfitting thanks to regularization

Overall, the model delivers promising accuracy on MRI-based Alzheimer’s classification.

# Future Enhancements

Test advanced deep learning architectures (DenseNet, ViT)

Add Grad-CAM heatmaps for model explainability

Explore multi-modal learning by integrating MRI + clinical data

Deploy as a web-based diagnostic tool for clinicians

# References

Suk et al., 2015 – Deep learning–based features for AD/MCI classification

Korolev et al., 2017 – CNNs for 3D brain MRI classification

Wang et al., 2018 – Transfer learning for Alzheimer’s diagnosis

Marcus et al., 2007 – OASIS MRI dataset

Jack et al., 2008 – ADNI MRI methods
