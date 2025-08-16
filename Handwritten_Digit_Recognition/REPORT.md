# Handwritten Digit Recognition Project Report

## Project Overview
This project uses the MNIST dataset to train a neural network for recognizing handwritten digits (0-9). The goal is to accurately classify images of digits and demonstrate practical skills in computer vision and deep learning.

## Dataset
- Source: MNIST (Keras/TensorFlow)
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28x28 pixels
- Classes: 10 (digits 0-9)

## Methodology
1. **Data Acquisition:**
   - Loaded MNIST dataset directly from Keras.
2. **Preprocessing:**
   - Normalized pixel values to [0, 1].
   - Converted labels to categorical format.
3. **Modeling:**
   - Built a simple neural network using Keras Sequential API.
   - Layers: Flatten, Dense (ReLU), Dropout, Dense (Softmax).
   - Trained for 10 epochs with validation split.
4. **Evaluation:**
   - Accuracy, confusion matrix, classification report.
   - Visualizations for sample digits, confusion matrix, and training history.

## Results
- **Test Accuracy:** ~98.1%
- **Confusion Matrix:**
  - Shows excellent classification performance across all digit classes.
- **Classification Report:**
  - Precision, recall, and F1-score are all very high for every digit class.

## Visualizations
- Sample digit images (`mnist_samples.png`)
- Confusion matrix heatmap (`confusion_matrix.png`)
- Training accuracy history (`training_history.png`)

## Conclusion
The neural network model achieves outstanding accuracy (98.1%) on the MNIST test set. The model is robust and generalizes well to unseen data. Further improvements can be made by using convolutional neural networks (CNNs) or advanced regularization techniques.

---
Prepared for Coding Samurai Internship
