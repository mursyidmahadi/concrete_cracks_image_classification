# Concrete Crack Classification (Transfer Learning)
This section contains the code and resources for building and training a machine learning model to detect concrete cracks in buildings. Image classification model was used to produce an accurate classification of concrete images into positive (cracks) or negative (without cracks) with the help of transfer learning techniques. The dataset is open-source and available to public for research purposes.

This case study is important to help and improvise the building structure and its safety, thus giving awareness to authority or construction builders in detecting any uprising problems ahead.

# Task Implementation

## 1. EDA (Exploratory Data Analysis)
A sample images of concrete were classified into two classes positive (with cracks) and negative (without cracks).

## 2. Data Preprocessing
3 steps were included in this stage:
- The dataset were splitted into train, validation, test sets with the ratio of 3:1:1.
- The dataset were loaded into PrefetchDataset for faster processing time
- Data augmentation model were added as a layer fro image classification model. (include Random flip and random rotation of images)

## 3. Model Development
The model architecture were used as a guideline to build this machine learning models
MobileNetV2 model is used for image processing and features extraction
- Data Augmentation are set as optional, so there will be two models as comparison for performance
- Softmax Activation Function was used/act as Activation Function
- Categorical Cross-Entropy Function (loss function)
- Only 5 epochs of training (no early stopping)
