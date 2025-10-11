"""
Complete ML Pipeline for Text-Image Verification
Combines NER model for text extraction and Image Classifier for verification
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import pipeline as hf_pipeline
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class TextImageVerificationPipeline:
    """
    Complete pipeline for verifying if text description matches image content
    """
    
    def __init__(self, ner_model_path, classifier_model_path):
        """
        Initialize the pipeline with both models
        
        Args:
            ner_model_path (str): Path to NER model directory
            classifier_model_path (str): Path to image classifier directory
        """
        print("=" * 80)
        print("INITIALIZING TEXT-IMAGE VERIFICATION PIPELINE")
        print("=" * 80)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {self.device}")
        
        # Initialize NER model
        print("\n1. Loading NER Model...")
        self._load_ner_model(ner_model_path)
        
        # Initialize Image Classifier
        print("\n2. Loading Image Classifier...")
        self._load_image_classifier(classifier_model_path)
        
        print("\n" + "=" * 80)
        print("PIPELINE READY")
        print("=" * 80)
    
    def _load_ner_model(self, model_path):
        """Load NER model for animal extraction"""   
        # Use Transformers pipeline - much simpler!
        self.ner_pipeline = hf_pipeline(
            "token-classification",
            model=model_path,
            aggregation_strategy="simple"
        )
        
        # Load animal variations
        variation_path = os.path.join(model_path, 'animal_variations.json')
        if os.path.exists(variation_path):
            with open(variation_path, 'r') as f:
                variations_data = json.load(f)
                self.variation_to_canonical = variations_data.get('variation_to_canonical', {})
        else:
            self.variation_to_canonical = {}
            for animal in ['dog', 'cat', 'horse', 'spider', 'butterfly', 
                          'chicken', 'sheep', 'cow', 'squirrel', 'elephant']:
                self.variation_to_canonical[animal] = animal
                self.variation_to_canonical[animal + 's'] = animal
        
        # Canonical animals
        self.canonical_animals = {
            'dog', 'cat', 'horse', 'spider', 'butterfly', 
            'chicken', 'sheep', 'cow', 'squirrel', 'elephant'
        }
        
        print(f"   ✓ NER model loaded from: {model_path}")
    
    def _load_image_classifier(self, model_path):
        """Load image classification model"""
        # Load class mapping
        with open(os.path.join(model_path, 'class_mapping.json'), 'r') as f:
            mapping = json.load(f)
        self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        self.num_classes = len(self.idx_to_class)
        
        # Create model architecture
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Load weights
        checkpoint = torch.load(
            os.path.join(model_path, 'best_model.pth'),
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.classifier_model = model.to(self.device)
        self.classifier_model.eval()
        
        # Define image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"   ✓ Image classifier loaded from: {model_path}")
        print(f"   ✓ Classes: {list(self.idx_to_class.values())}")
    
    def extract_animals_from_text(self, text):
        """
        Extract animal entities from text using NER model
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of extracted animal names (normalized to canonical form)
        """
        # Use pipeline to extract entities
        entities = self.ner_pipeline(text)
        
        # Extract and normalize animal names
        animals = set()
        
        for entity in entities:
            if entity['entity_group'] == 'ANIMAL':
                # Get the word and clean it
                word = entity['word'].lower().strip('.,!?#')
                
                # Map to canonical form
                canonical = self.variation_to_canonical.get(word)
                
                if canonical:
                    animals.add(canonical)
                elif word in self.canonical_animals:
                    animals.add(word)
        
        return list(animals)
    
    def classify_image(self, image_path):
        """
        Classify animal in image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top prediction
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = self.idx_to_class[predicted_idx.item()]
        
        return predicted_class, confidence.item()
    
    def verify(self, text, image_path, confidence_threshold=0.5, verbose=True):
        """
        Main verification function
        
        Args:
            text (str): Input text description
            image_path (str): Path to image file
            confidence_threshold (float): Minimum confidence for image classification
            verbose (bool): Print detailed information
            
        Returns:
            bool: True if text matches image, False otherwise
        """
        if verbose:
            print("\n" + "=" * 80)
            print("VERIFICATION PROCESS")
            print("=" * 80)
            print(f"\nText: '{text}'")
            print(f"Image: {image_path}")
        
        # Step 1: Extract animals from text
        extracted_animals = self.extract_animals_from_text(text)
        
        if verbose:
            print(f"\n1. Extracted animals from text: {extracted_animals}")
        
        if not extracted_animals:
            if verbose:
                print("   ✗ No animals found in text")
            return False
        
        # Step 2: Classify image
        predicted_animal, confidence = self.classify_image(image_path)
        
        if verbose:
            print(f"\n2. Image classification:")
            print(f"   Predicted: {predicted_animal}")
            print(f"   Confidence: {confidence*100:.2f}%")
        
        # Step 3: Check confidence threshold
        if confidence < confidence_threshold:
            if verbose:
                print(f"   ✗ Confidence below threshold ({confidence_threshold*100:.0f}%)")
            return False
        
        # Step 4: Verify match
        is_match = predicted_animal in extracted_animals
        
        if verbose:
            print(f"\n3. Verification:")
            print(f"   Text mentions: {extracted_animals}")
            print(f"   Image contains: {predicted_animal}")
            
            if is_match:
                print(f"   ✓ MATCH: The text correctly describes the image!")
            else:
                print(f"   ✗ NO MATCH: The text does not match the image.")
            
            print("=" * 80)
        
        return is_match
    
    def batch_verify(self, text_image_pairs, confidence_threshold=0.5):
        """
        Verify multiple text-image pairs
        
        Args:
            text_image_pairs (list): List of tuples (text, image_path)
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            list: List of verification results
        """
        results = []
        
        print("\n" + "=" * 80)
        print(f"BATCH VERIFICATION: {len(text_image_pairs)} pairs")
        print("=" * 80)
        
        for i, (text, image_path) in enumerate(text_image_pairs, 1):
            print(f"\n[{i}/{len(text_image_pairs)}] Processing...")
            
            is_match = self.verify(text, image_path, confidence_threshold, verbose=False)
            
            results.append({
                'text': text,
                'image': image_path,
                'match': is_match
            })
            
            status = "✓ MATCH" if is_match else "✗ NO MATCH"
            print(f"    {status}")
        
        # Summary
        matches = sum(1 for r in results if r['match'])
        print("\n" + "=" * 80)
        print(f"SUMMARY: {matches}/{len(results)} matches")
        print("=" * 80)
        
        return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Verify if text description matches image content'
    )
    
    parser.add_argument('--ner_model', type=str, default='models/ner_model',
                       help='Path to NER model directory')
    parser.add_argument('--classifier_model', type=str, default='models/image_classifier',
                       help='Path to image classifier directory')
    parser.add_argument('--text', type=str, default=None,
                       help='Input text description')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--input_file', type=str, default=None,
                       help='JSON file with text-image pairs')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TextImageVerificationPipeline(
        args.ner_model,
        args.classifier_model
    )
    
    # Single verification
    if args.text and args.image:
        result = pipeline.verify(
            args.text,
            args.image,
            args.confidence_threshold,
            verbose=True
        )
        
        print(f"\nFinal Result: {result}")
        sys.exit(0 if result else 1)
    
    # Batch verification from file
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Expected format: [{"text": "...", "image": "..."}]
        text_image_pairs = [(item['text'], item['image']) for item in data]
        
        results = pipeline.batch_verify(text_image_pairs, args.confidence_threshold)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
    
    # Interactive mode
    else:
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("\nEnter text and image path to verify (Ctrl+C to exit)")
        
        try:
            while True:
                print("\n" + "-" * 80)
                text = input("Text: ").strip()
                if not text:
                    continue
                
                image_path = input("Image path: ").strip()
                if not image_path or not os.path.exists(image_path):
                    print("Invalid image path!")
                    continue
                
                result = pipeline.verify(
                    text,
                    image_path,
                    args.confidence_threshold,
                    verbose=True
                )
                
                print(f"\nResult: {result}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")


if __name__ == '__main__':
    main()