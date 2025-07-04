# REDDEL-Net

## Description:
A Renyi Entropy-Driven Deep Ensemble Learning Approach for Classifying Alzheimer's Disease from Structural Brain MR Images

> Methodology
- A lightweight, interpretable, hybrid learning framework for Alzheimer's Disease classification using brain MRI scans
- It employs fine-tuned MobileNetV2 with embedded Convolutional Block Attention Module (CBAM) blocks after every convolutional 2d layer
- The trained MobileNetV2+CBAM model is used as a feature extractor; Grad-CAM and LIME are applied to visually highlight the most informative regions in MRI slices
- Renyi-entropy based feature selection optimizes the process by using 7-19% of the extracted feature space by preserving key discriminative information
- Soft-voting ensemble of Logistic Regression, Support Vector Machine with Radial Basis Function, and Random Forest is used for classification
- t-Stochastic Neighborhood Embedding is used to map the distinct class-wise separation; Confidence Intervals (>95%) are illustrated using violin plots
- Achieves 99.38% and 99.89% F1-Score on Kaggle AD and OASIS-1 datasets respectively
- REDDEL-Net requires minimal computational resources and inference times, making it suitable for real-time deployment in clinical environments

> Datasets Used
- **[Kaggle AD](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset?select=OriginalDataset)**
- **[OASIS-1](https://www.kaggle.com/datasets/pulavendranselvaraj/oasis-dataset)**

> End-to-End Workflow Used for Proposed Framework:
![End-to-End Workflow of Proposed REDDEL-Net](https://github.com/user-attachments/assets/93353864-34a9-4da8-bebb-34333fb85c06)

> Results for Kaggle AD:

- Training & Validation Loss Curve for MobileNetV2+CBAM over Epochs for Kaggle AD

![Training & Validation Loss Curve for MobileNetV2+CBAM over Epochs for Kaggle AD](https://github.com/user-attachments/assets/6c91d993-0bec-480d-a252-1912c1afbf38)


- Feature Selection versus Accuracy for Kaggle AD

![Feature Selection versus Accuracy](https://github.com/user-attachments/assets/f05f081a-4642-4cac-89ed-a22f79b9b9e2)


- Accuracy curve for 5-fold Stratified Cross Validation for Kaggle AD

![Accuracy curve for 5-fold Stratified Cross Validation for Kaggle AD](https://github.com/user-attachments/assets/eff3b4b9-0b5c-40dc-805e-8b3637ef39fb)


- Violin Plot for Distribution of Performance Metrics Across Cross Validation Folds for Kaggle AD

![Violin Plot for Distribution of Performance Metrics Across Cross Validation Folds for Kaggle AD](https://github.com/user-attachments/assets/3d38d61b-952f-4f49-b4d3-f2d1593fafd6)


- System Resource Usage Graph During Inference of Test Set on Kaggle AD

![System Resource Usage Graph During Inference of Test Set on Kaggle AD](https://github.com/user-attachments/assets/0c333e39-0fd8-4af3-ac44-dd43cf4dfbe1)


- Confusion Matrix for Kaggle AD

![Confusion Matrix for Kaggle AD](https://github.com/user-attachments/assets/f452f39f-4c4b-48a4-9c8a-2ef0918be52c)


- t-SNE of Kaggle AD Test-Set Features by Predicted Labels

![t-SNE of Kaggle AD Test-Set Features by Predicted Labels](https://github.com/user-attachments/assets/c2c3a767-f026-4924-a945-f67868a20b93)


> XAI Visualizations for OASIS-1 and Kaggle AD

- Grad-CAM for OASIS-1 and Kaggle AD

![Grad-CAM for OASIS-1 and Kaggle AD](https://github.com/user-attachments/assets/ff2ea134-e5e6-47af-a759-4038c26c0eeb)

- LIME for OASIS-1 and Kaggle AD

![LIME for OASIS-1 and Kaggle AD](https://github.com/user-attachments/assets/be67845e-f3bb-4bde-9f95-1096352c1afb)

- SHAP for OASIS-1 and Kaggle AD

![SHAP for Top 25 Features of OASIS-1 and Kaggle AD](https://github.com/user-attachments/assets/7f3f0ab5-45b2-482f-9987-a5e78687afe9)


## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

> Program Files:
- [Dependencies](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/requirements.txt)
- [Data Loader](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/data_loader.py)
- [MobileNetV2+CBAM Architecture](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/models.py)
- [Training](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/train.py)
- [Evaluation](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/evaluate.py)
- [Feature Extraction](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/feature_extractor.py)
- [Renyi-Entropy + ML-Ensemble Classifer](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/ml_pipeline.py)
- [Explainable AI (XAI)](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/src/visualize_xai.py)

> Scripts:
- [Training Pipeline](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/scripts/run_training.py)
- [ML Pipeline](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/scripts/run_ml_pipeline.py)
- [Inference Pipeline](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/scripts/run_inference.py)

> Implementation Files:
- [Kaggle AD Implementation Jupyter Notebook](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/alzheimers-classification-mri-kaggle-ad-dataset.ipynb)
- [OASIS-1 Implementation Jupyter Notebook](https://github.com/ksatrajit0/MRI-Alz-DeepEnsemble-XAI/blob/main/alzheimers-classification-mri-oasis-dataset.ipynb)

### Importance of Project:
- Exemplary results with minimal computational resources and inference times
- End-to-end interpretability and outperforms SOTA approaches on benchmarked, standard datasets
- Real-time clinical applicability for better therapeutic outcomes

### Paper:
> It'd be great if you could cite our paper (under review) if this code has been helpful to you.
 
> Thank you very much!
