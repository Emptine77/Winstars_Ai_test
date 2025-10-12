# MNIST Digit Classification Pipeline

A comprehensive machine learning pipeline that implements and compares multiple classification approaches for the MNIST handwritten digit dataset.

## Project Overview

This project implements three different classification models for MNIST digit recognition:

1. **Random Forest Classifier**: Traditional ensemble learning approach
2. **Feedforward Neural Network**: Deep learning with fully connected layers
3. **Convolutional Neural Network**: State-of-the-art CNN architecture for image classification

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Emptine77/Winstars_Ai_test/edit/main/task1/
cd mnist-classifier

# Install dependencies
pip install -r requirements.txt
```

## Running the Classifier

Run the classifier in interactive mode to test different models:

```bash
python mnist_classifier.py
```

**Interactive Commands:**
- `rf` - Random Forest Classifier
- `nn` - Feedforward Neural Network
- `cnn` - Convolutional Neural Network
- `Exit` - Quit the program

### Example Session

```
Enter which model to use: 'rf' (random forest), 'nn' (feedforward neural network), 'cnn' (convolutional neural network).
Type 'Exit' to quit.
Model> rf
You selected: rf model.
Training model...
Accuracy on subset: 0.9691

Model> nn
You selected: nn model.
Training model...
Epoch [5/100] Train Loss: 0.0658, Acc: 97.95% | Val Loss: 0.0856, Val Acc: 97.34%
...
Early stopping at epoch 54
Loaded best model from training
Accuracy on subset: 0.9840

Model> cnn
You selected: cnn model.
Training model...
Epoch [5/50] Train Loss: 0.0325, Acc: 99.01% | Val Loss: 0.0360, Val Acc: 98.98%
...
Early stopping at epoch 29
Loaded best model from training
Accuracy on subset: 0.9922
```

## Model Architectures

### 1. Random Forest Classifier

**Architecture:**
- Ensemble learning method
- Default scikit-learn configuration
- No hyperparameter tuning

**Performance:**
- Test Accuracy: 96.91%
- Training Time: Fast (~seconds)
- Best for: Quick baseline, interpretability

### 2. Feedforward Neural Network

**Architecture:**
- Input Layer: 784 neurons (28×28 flattened images)
- Hidden Layer 1: 700 neurons + BatchNorm + GELU + Dropout(0.2)
- Hidden Layer 2: 700 neurons + BatchNorm + GELU + Dropout(0.2)
- Hidden Layer 3: 350 neurons + BatchNorm + GELU + Dropout(0.2)
- Hidden Layer 4: 256 neurons + BatchNorm + GELU + Dropout(0.1)
- Output Layer: 10 neurons (digit classes)

**Training Configuration:**
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: OneCycleLR (max_lr=0.01)
- Loss: CrossEntropyLoss
- Batch Size: 128
- Max Epochs: 100
- Early Stopping: Patience=15

**Performance:**
- Test Accuracy: 98.40%
- Best Validation Accuracy: 98.54% (Epoch 40)
- Training Time: ~5-10 minutes (stopped at epoch 54)

### 3. Convolutional Neural Network

**Architecture:**
- Conv Block 1: Conv2d(1→32) + BatchNorm + GELU + MaxPool2d + Dropout2d(0.2)
- Conv Block 2: Conv2d(32→64) + BatchNorm + GELU + MaxPool2d + Dropout2d(0.2)
- Flatten Layer
- FC Layer 1: Linear(3136→128) + GELU + Dropout(0.1)
- Output Layer: Linear(128→10)

**Training Configuration:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Loss: CrossEntropyLoss
- Batch Size: 256
- Max Epochs: 50
- Early Stopping: Patience=15

**Performance:**
- Test Accuracy: 99.22%
- Best Validation Accuracy: 99.41% (Epoch 29)
- Training Time: ~10-15 minutes (stopped at epoch 29)

## Training Process

### Data Preprocessing

1. **Dataset Loading**: MNIST dataset fetched from OpenML
2. **Normalization**: Pixel values normalized to [0, 1]
3. **Train-Test Split**: 80% training, 20% testing
4. **Validation Split**: 10% of training data for validation

### Training Features

**All Neural Network Models Include:**
- Automatic train/validation splitting
- Batch processing with DataLoader
- Gradient clipping for stability
- Early stopping to prevent overfitting
- Learning rate scheduling
- Best model checkpointing
- GPU acceleration (if available)

**Model-Specific Features:**

*Feedforward Network:*
- OneCycleLR scheduler for faster convergence
- Multiple hidden layers with BatchNorm
- GELU activation for better gradients

*Convolutional Network:*
- 2D convolutions for spatial feature extraction
- MaxPooling for dimensionality reduction
- ReduceLROnPlateau for adaptive learning rate

## Performance Comparison

| Model | Test Accuracy | Training Time | Parameters | Best Use Case |
|-------|---------------|---------------|------------|---------------|
| Random Forest | 96.91% | ~seconds | N/A | Quick baseline |
| Feedforward NN | 98.40% | ~5-10 min | ~1.5M | Balanced performance |
| CNN | **99.22%** | ~10-15 min | ~150K | Best accuracy |

### Training Metrics

**Feedforward Neural Network:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|------------|-----------|----------|---------|-------|
| 5/100 | 0.0658 | 97.95% | 0.0856 | 97.34% | - |
| 10/100 | 0.0545 | 98.22% | 0.0766 | 97.75% | - |
| 15/100 | 0.0434 | 98.52% | 0.0731 | 98.07% | - |
| 20/100 | 0.0385 | 98.74% | 0.0767 | 97.93% | - |
| 30/100 | 0.0258 | 99.19% | 0.0658 | 98.27% | - |
| 40/100 | 0.0173 | 99.43% | 0.0680 | 98.54% | ✓ Best model |
| 54/100 | - | - | - | - | Early stopping |

**Convolutional Neural Network:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|------------|-----------|----------|---------|-------|
| 5/50 | 0.0325 | 99.01% | 0.0360 | 98.98% | - |
| 10/50 | 0.0197 | 99.37% | 0.0281 | 99.29% | - |
| 15/50 | 0.0078 | 99.75% | 0.0258 | 99.38% | - |
| 20/50 | 0.0043 | 99.90% | 0.0284 | 99.29% | - |
| 25/50 | 0.0036 | 99.92% | 0.0267 | 99.34% | - |
| 29/50 | 0.0027 | 99.94% | 0.0263 | 99.41% | ✓ Best model |

### Key Observations

1. **CNN Superiority**: CNN achieves the highest accuracy (99.22%) with fewer parameters than the feedforward network
2. **Fast Convergence**: All models converge within 30-60 epochs thanks to advanced optimization techniques
3. **No Overfitting**: Small gap between train and test accuracy across all models
4. **Early Stopping**: Effective early stopping prevents unnecessary computation
5. **Gradient Stability**: Batch normalization and gradient clipping ensure stable training

## Troubleshooting

### CUDA Out of Memory

If you encounter memory issues with neural networks:

**Feedforward Network:**
```python
# Reduce batch size
FeedForwardMnistClassifier(input_size=784, hidden_size=700, output_size=10, batch_size=64)
```

**CNN:**
```python
# Reduce batch size
ConvolutionalMnistClassifier(batch_size=128)
```

### Slow Training

If training is too slow:
- Ensure CUDA is available: The code automatically uses GPU if available
- Reduce number of epochs
- Increase batch size (if memory allows)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Contact

For questions, you can contact me [Empt1ne77@gmail.com]
