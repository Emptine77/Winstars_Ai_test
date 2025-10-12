# Animal Text-Image Verification Pipeline

A complete machine learning pipeline that combines Natural Language Processing (NER) and Computer Vision to verify if a text description matches the content of an image.

## Project Overview

This project implements a two-stage ML pipeline:

1. **NER Model**: Extracts animal entities from text using a BERT-based transformer model
2. **Image Classifier**: Classifies animals in images using ResNet50 with transfer learning
3. **Verification Pipeline**: Combines both models to verify text-image correspondence

### Supported Animals

The pipeline supports 10 animal classes:
- Dog
- Cat
- Horse
- Spider
- Butterfly
- Chicken
- Sheep
- Cow
- Squirrel
- Elephant

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Emptine77/Winstars_Ai_test

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the Animals-10 dataset from Kaggle:
```bash
# Using Kaggle API
kaggle datasets download -d alessiocorrado99/animals10
unzip animals10.zip -d raw-img/
```

**IMPORTANT**:
The original Animals-10 dataset is in **Italian**.
You should either download an English version (if available) or rename the folders manually to their English equivalents, as shown below.
Dataset structure:
```
raw-img/
├── cane/           # dog
├── cavallo/        # horse
├── elefante/       # elephant
├── farfalla/       # butterfly
├── gallina/        # chicken
├── gatto/          # cat
├── mucca/          # cow
├── pecora/         # sheep
├── ragno/          # spider
└── scoiattolo/     # squirrel
```

## Exploratory Data Analysis

Run the EDA notebook to analyze the dataset:

```bash
jupyter notebook eda_notebook.ipynb
```

This will generate:
- Class distribution visualizations
- Image dimension statistics
- Sample images from each class
- Training recommendations

## Training Models

### 1. Train NER Model

Train the animal entity recognition model:

```bash
python ner_train.py \
    --model_name bert-base-uncased \
    --output_dir models/ner_model \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 3e-5
```

**Parameters:**
- `--model_name`: Pretrained BERT model (default: bert-base-uncased)
- `--output_dir`: Directory to save trained model
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--max_len`: Maximum sequence length (default: 128)

### 2. Train Image Classifier

Train the animal image classification model:

```bash
python image_class_train.py \
    --data_dir raw-img \
    --output_dir models/image_classifier \
    --batch_size 32 \
    --num_epochs 5 \
    --learning_rate 0.001
```

**Parameters:**
- `--data_dir`: Root directory of image dataset
- `--output_dir`: Directory to save trained model
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--num_workers`: Number of data loading workers

## Inference

### NER Model Inference

Extract animals from text:

```bash
# Single text
python ner_inference.py \
    --model_path models/ner_model \
    --text "There is a cow in the picture"

# Interactive mode
python ner_inference.py --model_path models/ner_model
```

### Image Classifier Inference

Classify animal images:

```bash
# Single image
python image_class_inferencer.py \
    --model_path models/image_classifier \
    --image path/to/image.jpg \
    --top_k 3

# Interactive mode
python image_class_inference.py --model_path models/image_classifier
```

## Complete Pipeline

Verify if a text matches an image:

```bash
python pipeline.py \
    --ner_model models/ner_model \
    --classifier_model models/image_classifier \
    --text "There is a cow in the picture" \
    --image path/to/cow.jpg \
    --confidence_threshold 0.5
```

**Output:**
```
================================================================================
INTERACTIVE MODE (Ctrl+C to exit)
================================================================================
Text: Can you see the cat in this photo? 
Image path: D:\Downloads\rn_image_picker_lib_temp_1c159566-036d-4984-9de5-12061937613c.jpg  

================================================================================
VERIFICATION
================================================================================

Extracted animals: ['cat']

Image predictions:
  1. cat: 99.97%
  2. dog: 0.03%

True
 Text mentions ['cat'], image shows cat
================================================================================
```

### Interactive Mode

Run the pipeline interactively:

```bash
python pipeline.py \
    --ner_model models/ner_model \
    --classifier_model models/image_classifier
```

## Project Structure

```
animal-verification-pipeline/
├── README.md
├── requirements.txt
├── eda_notebook.ipynb          # Exploratory Data Analysis
├── train_ner.py                # NER model training
├── infer_ner.py                # NER model inference
├── train_image_classifier.py   # Image classifier training
├── infer_image_classifier.py   # Image classifier inference
├── pipeline.py                 # Complete verification pipeline
├── models/
│   ├── ner_model/              # Trained NER model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── label_map.json
│   └── image_classifier/       # Trained image classifier
│       ├── best_model.pth
│       ├── class_mapping.json
│       └── training_history.png
└── raw-img/                    # Dataset directory
    ├── dog/
    ├── spider/
    └── ...
```

## How It Works

### Pipeline Flow

1. **Text Processing**:
   - User provides text like "There is a cow in the picture"
   - NER model tokenizes and analyzes the text
   - Extracts animal entities using BIO tagging scheme
   - Normalizes entities (e.g., "cows" → "cow")

2. **Image Processing**:
   - User provides an image
   - Image is resized to 224×224 pixels
   - Normalized using ImageNet statistics
   - Fed through ResNet50 classifier
   - Returns predicted animal class and confidence

3. **Verification**:
   - Compares extracted text entities with image prediction
   - Checks confidence threshold
   - Returns a boolean: True if match, False otherwise

### Model Architectures

**NER Model:**
- Base: BERT (bert-base-uncased)
- Task: Token Classification
- Labels: O, B-ANIMAL, I-ANIMAL
- Training: Synthetic data generation with templates

**Image Classifier:**
- Base: ResNet50 
- Modified: Custom FC layers with dropout
- Training: Transfer learning with data augmentation
- Optimizer: Adam with learning rate scheduling

## Performance

### NER Model
- F1 Score: 1.00
- Precision: 1.00
- Recall: 1.00

### Image Classifier
- Validation Accuracy: 95.91%
- Training uses:
  - Data augmentation (flips, rotations, color jitter)
  - Class weighting for imbalanced classes
  - Early stopping on validation loss

## Training Process for Image Classifier Model

### Model Architecture
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Transfer Learning**: Early layers frozen, custom classification head
- **Classifier Head**: 
  - Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→10)

### Training Configuration
- **Dataset Split**: 70% train / 10% validation / 20% test (stratified)
- **Batch Size**: 32
- **Epochs**: 5
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss with class weights
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)

### Data Augmentation
- **Training**: Random crop, horizontal flip, rotation (±15°), color jitter
- **Validation/Test**: Center crop only
- **Normalization**: ImageNet statistics

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|------------|-----------|----------|---------|-------|
| 1/5   | 0.4206     | 87.30%    | 0.1472   | 95.15%  | ✓ Best model saved |
| 2/5   | 0.3045     | 90.65%    | 0.1427   | 95.42%  | ✓ Best model saved |
| 3/5   | 0.2909     | 91.12%    | 0.1571   | 94.84%  | - |
| 4/5   | 0.2748     | 91.67%    | 0.1227   | 95.91%  | ✓ Best model saved |
| 5/5   | 0.2684     | 91.84%    | 0.1396   | 95.07%  | - |

### Final Performance
- **Best Validation Accuracy**: 95.91% (Epoch 4)
- **Test Accuracy**: 95.74%
- **Training Time**: ~1h 30min (5 epochs)
- **Time per Epoch**: ~18 minutes

### Key Observations
1. **Rapid Convergence**: Model achieved 95%+ validation accuracy in just 1 epoch
2. **No Overfitting**: Small gap between train (91.84%) and validation (95.91%) accuracy
3. **Stable Performance**: Test accuracy (95.74%) closely matches best validation accuracy
4. **Effective Transfer Learning**: Pretrained ResNet50 weights provided excellent starting point
5. **Class Balancing**: Weighted loss function successfully handled class imbalance

### Training Metrics Graph
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/192f4a1a-58a1-4a71-a0f0-0708815e63a9" />

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python train_image_classifier.py --batch_size 16  # Instead of 32
```

### NER Not Extracting Animals
- Check if text uses supported animal names
- Ensure proper text formatting (no special characters)
- Retrain with additional templates
- Check the text for errors

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

For questions or issues, please open an issue on GitHub or contact [Empt1ne77@gmail.com]

