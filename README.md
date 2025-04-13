# MNIST Digit Classification using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The project includes both the training implementation and a user-friendly graphical interface for testing the model.

## Project Structure

- `training.py` - Contains the CNN model implementation and training code
- `mnist-recognition-interface.py` - GUI application for testing the trained model
- `mnist_cnn_model.h5` - Pre-trained model weights
- `png/` - Directory containing training results and visualizations

## Visualizations

The `png` directory contains the following visualizations:

### Training History Plots
- `training_history_adam_categorical_crossentropy.png` - Training metrics using Adam optimizer with categorical crossentropy
- `training_history_sgd_categorical_crossentropy.png` - Training metrics using SGD optimizer with categorical crossentropy
- `training_history_rmsprop_categorical_crossentropy.png` - Training metrics using RMSprop optimizer with categorical crossentropy
- `training_history_adam_mean_squared_error.png` - Training metrics using Adam optimizer with mean squared error

### Confusion Matrices
- `confusion_matrix_adam_categorical_crossentropy.png` - Confusion matrix for Adam + categorical crossentropy
- `confusion_matrix_sgd_categorical_crossentropy.png` - Confusion matrix for SGD + categorical crossentropy
- `confusion_matrix_rmsprop_categorical_crossentropy.png` - Confusion matrix for RMSprop + categorical crossentropy
- `confusion_matrix_adam_mean_squared_error.png` - Confusion matrix for Adam + mean squared error

## Model Architecture

The CNN architecture consists of:
1. Conv2D layer: 32 filters (3x3) with ReLU activation
2. MaxPooling2D layer: 2x2 pool size
3. Conv2D layer: 64 filters (3x3) with ReLU activation
4. MaxPooling2D layer: 2x2 pool size
5. Conv2D layer: 64 filters (3x3) with ReLU activation
6. Flatten layer
7. Dense layer: 128 neurons with ReLU activation
8. Dropout layer: 0.5 dropout rate
9. Output layer: 10 neurons with softmax activation

## Training Results

| Configuration | Accuracy | Training Time (sec) |
|--------------|----------|-------------------|
| Adam + categorical_crossentropy | 0.9908 | 416.30 |
| SGD + categorical_crossentropy | 0.9757 | 377.44 |
| RMSprop + categorical_crossentropy | 0.9912 | 342.27 |
| Adam + mean_squared_error | 0.9899 | 348.69 |

## Key Findings

1. RMSprop + categorical_crossentropy achieved the highest accuracy (0.9912)
2. Adam + categorical_crossentropy showed the best balance between speed and accuracy
3. SGD performed significantly worse than adaptive optimizers
4. categorical_crossentropy outperformed mean_squared_error for classification

## GUI Features

The graphical interface includes:
- Drawing canvas for handwritten digit input
- Line thickness adjustment
- Real-time digit recognition
- Confidence level display
- Probability distribution visualization
- Image saving and loading capabilities
- Canvas clearing function

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Tkinter

## Usage

1. Training the model:
```bash
python training.py
```

2. Running the GUI:
```bash
python mnist-recognition-interface.py
```

## Common Misclassifications

The model occasionally confuses:
- 4 ↔ 9
- 7 ↔ 1
- 5 ↔ 3
- 9 ↔ 7

These errors are primarily due to visual similarities between the digits.

## Conclusion

The implemented CNN architecture successfully achieves high accuracy (>99%) in digit classification. The combination of RMSprop optimizer with categorical crossentropy loss function yields the best results, though Adam optimizer provides a good balance between training speed and accuracy. 