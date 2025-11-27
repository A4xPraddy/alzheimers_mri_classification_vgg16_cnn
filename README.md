# Alzheimer-Disease-Multiclass-Classification-Dementia-Progression-

# Alzheimer's Disease Detection Using Deep Learning

## Description
This project implements deep learning techniques for detecting Alzheimer's Disease (AD) from MRI brain scans. It explores two primary approaches:
1. **Transfer Learning with VGG16:** A pre-trained VGG16 model is fine-tuned to classify different stages of Alzheimer's Disease.
2. **Custom CNN Model:** A deep learning architecture specifically designed for AD classification.

The models are evaluated on two datasets:
- **ADNI Dataset:** Classifies subjects into Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), Late Mild Cognitive Impairment (LMCI), and Alzheimer's Disease (AD).
- **Alzheimer's Disease Multiclass Dataset:** Categorizes MRI scans into NonDemented, VeryMildDemented, MildDemented, and ModerateDemented.

## Features
- Uses **CNNs and Transfer Learning (VGG16)** for classification.
- **Data preprocessing techniques**: Segmentation, histogram equalization, CLAHE, and K-means clustering.
- **Data augmentation** to mitigate class imbalance.
- **Evaluation metrics**: Accuracy, Precision, Recall, and F1-score.

## Results
| Model | Dataset | Classification Task | Accuracy |
|--------|---------|-------------------|----------|
| VGG16 | ADNI | AD/EMCI/LMCI/NC | 98% |
| VGG16 | Alzheimer's Multiclass | Multi-class | 98% |
| Custom CNN | ADNI | AD/EMCI/LMCI/NC | 99% |
| Custom CNN | Alzheimer's Multiclass | Multi-class | 98% |

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/A4xMimic/Alzheimer-Disease-Multiclass-Classification-Dementia-Progression-.git
   cd Alzheimer-Disease-Multiclass-Classification-Dementia-Progression-
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset Preparation
Download the datasets:
- **ADNI Dataset**: [ADNI Website](https://adni.loni.usc.edu/)
- **Alzheimer's Disease Multiclass Dataset**: Available on Kaggle.

Place the dataset in the `data/` directory and structure it as follows:
```
data/
  ├── ADNI/
  ├── Alzheimer's Multiclass/
```

## Model Training & Evaluation
Run the training script:
```sh
python train.py --model vgg16 --dataset adni
```
To test a trained model:
```sh
python evaluate.py --model vgg16 --dataset adni
```

## Project Structure
```
├── data/                     # Datasets
├── models/                   # Trained models
├── src/
│   ├── preprocessing.py      # Data preprocessing
│   ├── model_vgg16.py       # VGG16 model implementation
│   ├── model_cnn.py         # Custom CNN model implementation
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

## Acknowledgments
- ADNI for providing the dataset.
- The deep learning community for open-source tools like TensorFlow and PyTorch.

