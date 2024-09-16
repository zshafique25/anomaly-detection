# Anomaly Detection using Autoencoder Neural Networks

This repository contains a project that implements anomaly detection using deep learning methods, specifically Autoencoder Neural Networks. The project demonstrates how to detect outliers (anomalies) in datasets using a neural network model trained to minimize reconstruction error for normal data points.

## Problem Description

Anomaly detection refers to the task of identifying unusual patterns or data points that do not conform to the expected behavior of a dataset. This is a critical task in various fields, including fraud detection, network security, and industrial system monitoring. Traditional methods often fall short for high-dimensional data, so this project utilizes an autoencoder, a type of neural network designed to learn efficient representations of data.

### Objective
The primary goal of this project is to build a model that:
- Learns to reconstruct normal data patterns.
- Identifies anomalies by measuring reconstruction error for unseen test data.

## Methodology

### 1. **Data Preprocessing**
   - The dataset is loaded and split into training and testing sets.
   - Feature scaling and normalization are applied to ensure faster convergence and better model performance.

### 2. **Model Architecture**
   - A deep autoencoder network is used. It consists of two main parts:
     - **Encoder**: Compresses the input data to a lower-dimensional representation.
     - **Decoder**: Attempts to reconstruct the original input from the compressed data.
   - Mean Squared Error (MSE) is used as the loss function for reconstruction.
   - The model is trained to minimize the reconstruction loss for normal data samples.

### 3. **Training**
   - The model is trained using normal data.
   - Epochs, batch size, and learning rate are carefully tuned to avoid overfitting.

### 4. **Anomaly Detection**
   - Once trained, the model is used to predict and reconstruct test data.
   - Reconstruction error is computed for each test data point. Points with high reconstruction error are classified as anomalies.

### 5. **Visualization**
   - The latent feature embeddings of the data are visualized using t-SNE (t-distributed stochastic neighbor embedding) to project high-dimensional data into two or three dimensions for better understanding of normal vs anomalous data distribution.

## Results

- The trained autoencoder model successfully identifies anomalous data points.
- Visualization using t-SNE shows a clear separation between normal and anomalous data in the latent space.
- Anomalies are detected based on a threshold reconstruction error, which can be adjusted based on the specific use case or application.

## Files

- `Anomaly_detection.ipynb`: The main Jupyter notebook containing all code, including data preprocessing, model training, and anomaly detection.
- `tsne_<epochs>.png`: The t-SNE plots generated after dimensionality reduction, showing the distribution of normal and anomalous data.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/zshafique25/anomaly-detection.git
    ```
2. Run the Jupyter notebook:
    ```bash
    jupyter notebook Anomaly_detection.ipynb
    ```

## Conclusion

This project demonstrates the effectiveness of deep autoencoder models for anomaly detection tasks. The model provides a high level of accuracy in detecting anomalous data points by leveraging reconstruction error as a metric. This approach can be applied to various domains where detecting abnormal behavior is crucial.
