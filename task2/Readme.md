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
git clone https://github.com/Emptine77/Winstars_Ai_test/blob/main/task2/Readme.md

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

## 📈 Performance

### NER Model
- F1 Score: ~0.95
- Precision: ~0.96
- Recall: ~0.94

### Image Classifier
- Validation Accuracy: ~85-90%
- Training uses:
  - Data augmentation (flips, rotations, color jitter)
  - Class weighting for imbalanced classes
  - Early stopping on validation loss

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

