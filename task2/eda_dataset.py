# Animal Classification Dataset - Exploratory Data Analysis
# This notebook analyzes the Animals-10 dataset for image classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# 1. DATASET OVERVIEW
# =============================================================================

# Dataset: Animals-10 from Kaggle
# Contains 10 classes: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant
# Link: https://www.kaggle.com/datasets/alessiocorrado99/animals10
OUTPUT_DIR = "eda_outputs"
DATA_DIR = "raw-img"  # Update this path to your dataset location
CLASSES = ['dog', 'horse', 'elephant', 'butterfly', 'chicken',
           'cat', 'cow', 'sheep', 'spider', 'squirrel']


print("=" * 80)
print("ANIMALS-10 DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# =============================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# =============================================================================

print("\n2. CLASS DISTRIBUTION")
print("-" * 80)

# Count images per class
class_counts = {}
for class_name in CLASSES:
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = count
    else:
        class_counts[class_name] = 0

# Create DataFrame
df_distribution = pd.DataFrame(list(class_counts.items()), 
                               columns=['Class', 'Count'])
df_distribution = df_distribution.sort_values('Count', ascending=False)

print(df_distribution.to_string(index=False))
print(f"\nTotal images: {df_distribution['Count'].sum()}")
print(f"Average per class: {df_distribution['Count'].mean():.2f}")
print(f"Std deviation: {df_distribution['Count'].std():.2f}")

# Visualize distribution
plt.figure(figsize=(12, 6))
bars = plt.bar(df_distribution['Class'], df_distribution['Count'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(df_distribution))))
plt.xlabel('Animal Class', fontsize=12, fontweight='bold')
plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
plt.title('Distribution of Images Across Animal Classes', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. IMAGE DIMENSION ANALYSIS
# =============================================================================

print("\n3. IMAGE DIMENSION ANALYSIS")
print("-" * 80)

# Sample images to analyze dimensions
widths, heights, aspects = [], [], []
sample_size = 100  # Sample 100 images per class

for class_name in CLASSES:
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        continue
    
    images = [f for f in os.listdir(class_path) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in images[:sample_size]:
        try:
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspects.append(w / h)
        except Exception as e:
            continue

# Create dimension statistics
dim_stats = pd.DataFrame({
    'Metric': ['Mean Width', 'Mean Height', 'Min Width', 'Min Height', 
               'Max Width', 'Max Height', 'Mean Aspect Ratio'],
    'Value': [
        f"{np.mean(widths):.2f}",
        f"{np.mean(heights):.2f}",
        f"{np.min(widths)}",
        f"{np.min(heights)}",
        f"{np.max(widths)}",
        f"{np.max(heights)}",
        f"{np.mean(aspects):.2f}"
    ]
})

print(dim_stats.to_string(index=False))

# Visualize dimensions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Width distribution
axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Width (pixels)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Image Width Distribution', fontweight='bold')
axes[0].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(widths):.0f}')
axes[0].legend()

# Height distribution
axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Height (pixels)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Image Height Distribution', fontweight='bold')
axes[1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(heights):.0f}')
axes[1].legend()

# Aspect ratio distribution
axes[2].hist(aspects, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
axes[2].set_ylabel('Frequency', fontweight='bold')
axes[2].set_title('Aspect Ratio Distribution', fontweight='bold')
axes[2].axvline(np.mean(aspects), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aspects):.2f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'image_dimensions.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. SAMPLE IMAGES VISUALIZATION
# =============================================================================

print("\n4. SAMPLE IMAGES FROM EACH CLASS")
print("-" * 80)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for idx, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        continue
    
    images = [f for f in os.listdir(class_path) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        img_path = os.path.join(class_path, images[0])
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(class_name.upper(), 
                           fontweight='bold', fontsize=12)
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. RECOMMENDATIONS FOR TRAINING
# =============================================================================

print("\n5. RECOMMENDATIONS FOR MODEL TRAINING")
print("-" * 80)

recommendations = """
Based on the EDA, here are the recommendations:

1. IMAGE PREPROCESSING:
   - Resize all images to 224x224 (standard for transfer learning)
   - Apply data augmentation to handle class imbalance
   - Normalize pixel values using ImageNet statistics

2. DATA SPLITTING:
   - Use 80% for training, 10% for validation, 10% for testing
   - Apply stratified splitting to maintain class distribution

3. MODEL ARCHITECTURE:
   - Use transfer learning with ResNet50 or EfficientNet
   - Fine-tune the last few layers
   - Add dropout for regularization

4. TRAINING STRATEGY:
   - Use class weights to handle imbalance
   - Apply learning rate scheduling
   - Use early stopping based on validation loss

5. DATA AUGMENTATION:
   - Random horizontal flips
   - Random rotation (±15 degrees)
   - Random brightness/contrast adjustments
   - Random crops and zooms
"""

print(recommendations)

# =============================================================================
# 6. EXPORT METADATA
# =============================================================================

# Save class distribution
df_distribution.to_csv(os.path.join(OUTPUT_DIR, 'class_distribution.csv'), index=False)

# Save dimension statistics
pd.DataFrame({
    'widths': widths,
    'heights': heights,
    'aspects': aspects
}).to_csv(os.path.join(OUTPUT_DIR, 'image_dimensions.csv'), index=False)

print("\n" + "=" * 80)
print("EDA COMPLETE - Files saved:")
print("  - class_distribution.png")
print("  - image_dimensions.png")
print("  - sample_images.png")
print("  - class_distribution.csv")
print("  - image_dimensions.csv")
print("=" * 80)