# Credit Card Fraud Detection: Dealing with Imbalanced Datasets

This project focuses on detecting fraudulent credit card transactions using various machine learning models, with an emphasis on handling the challenges of imbalanced datasets. The data consists of transactions labeled as either fraud (1) or non-fraud (0), with a significant imbalance towards non-fraudulent transactions. This imbalance presents challenges in model performance, particularly in accurately predicting fraud.

## Goals and Objectives:

### Goals:
- **Accurate Fraud Detection**: Build a model that can detect fraudulent transactions while minimizing false positives and false negatives.
- **Handle Imbalanced Data**: Apply techniques such as undersampling and oversampling to address the imbalance in the dataset and ensure better model performance.
- **Compare Multiple Models**: Evaluate various machine learning models and a neural network to determine the most effective approach for this specific task.

### Objectives:
1. **Understand the Data**:
   - Analyze the dataset to gain insights into the features and distributions.
   - Identify and address any class imbalance issues in the dataset.

2. **Data Preprocessing**:
   - Scale and distribute the data appropriately to prepare it for model training.
   - Split the dataset into training and testing sets to evaluate model performance effectively.

3. **Handle Class Imbalance**:
   - Implement **Random UnderSampling** and **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced dataset for training.
   - Ensure that these techniques are applied correctly to improve fraud detection accuracy.

4. **Dimensionality Reduction**:
   - Use **t-SNE** for reducing the dimensionality of the dataset and visualizing the clustering of fraudulent and non-fraudulent transactions.

5. **Train and Compare Models**:
   - Evaluate the performance of various classification algorithms, including:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - K-Nearest Neighbors
   - Build a basic neural network and compare its accuracy with traditional classifiers.

6. **Evaluate Performance**:
   - Use precision, recall, F1-score, and AUC-ROC metrics to evaluate the models' performance, especially with imbalanced data.
   - Focus on metrics that handle imbalanced datasets effectively, ensuring that fraud detection is accurate without sacrificing performance on non-fraudulent transactions.

---

## Project Overview:

1. **Data Understanding and Exploration**:
   - Load and inspect the dataset.
   - Analyze the distribution of features and identify the proportion of fraud vs. non-fraud transactions.

2. **Data Preprocessing**:
   - Scale and transform features.
   - Split the data into training and testing sets for model validation.

3. **Handling Class Imbalance**:
   - Implement **NearMiss Algorithm** for undersampling the majority class.
   - Apply **SMOTE** for oversampling the minority class.

4. **Dimensionality Reduction and Visualization**:
   - Use **t-SNE** to reduce the dimensionality of the data for easier visualization of fraud patterns.

5. **Classification Models**:
   - Train and evaluate various classifiers such as:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors

6. **Neural Networks**:
   - Build a simple neural network to classify transactions and compare its performance to the traditional classifiers.

7. **Performance Evaluation**:
   - Assess model performance using metrics such as precision, recall, F1-score, and AUC-ROC, with particular attention to handling imbalanced datasets.

---

## Dependencies:

- **Pandas**, **Numpy**: Data manipulation and analysis.
- **Matplotlib**, **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models and evaluation metrics.
- **Imbalanced-learn**: Tools for handling imbalanced datasets (e.g., SMOTE, NearMiss).
- **TensorFlow/Keras** (optional): For building neural networks.

---

## Conclusion:

This project demonstrates effective ways to deal with imbalanced datasets, especially in fraud detection scenarios. By comparing different models and techniques, you can gain insights into the strengths and weaknesses of various approaches for handling real-world data imbalance challenges.

---

## How to Use:

1. Clone the repository:
   ```bash
   git clone https://github.com/shaheerm10/credit-fraud-detection
   cd credit-fraud-detection
