"""
Optimized ML Pipeline for Text-Image Verification
Combines NER model for text extraction and Image Classifier for verification
"""

import argparse
import os
import sys
from ner_inference import AnimalNERExtractor
from image_class_inference import AnimalClassifier


class TextImageVerificationPipeline:
    """Pipeline for verifying if text description matches image content"""
    
    def __init__(self, ner_model_path, classifier_model_path):
 
        print("\n1. Loading NER Model...")
        self.ner_extractor = AnimalNERExtractor(ner_model_path)
        
        print("\n2. Loading Image Classifier...")
        self.classifier = AnimalClassifier(classifier_model_path)
    
    def verify(self, text, image_path, top_k=3, verbose=True):
        """
        Verify text-image match with top-k predictions
        
        Args:
            text (str): Input text description
            image_path (str): Path to image file
            top_k (int): Number of top predictions to show
            verbose (bool): Print detailed information
            
        Returns:
            bool: True if text matches image, False otherwise
        """
        if verbose:
            print("\n" + "=" * 80)
            print("VERIFICATION")
            print("=" * 80)
        
        # Extract animals from text
        extracted_animals = self.ner_extractor.extract_animals(text)
        
        if verbose:
            print(f"\nExtracted animals: {extracted_animals}")
        
        if not extracted_animals:
            if verbose:
                print("Animals not found in text")
            return False
        
        # Get top-k predictions
        predictions = self.classifier.predict(image_path, top_k=top_k)
        
        if verbose:
            print("\nImage predictions:")
            for i, (animal, conf) in enumerate(predictions, 1):
                print(f"  {i}. {animal}: {conf*100:.2f}%")
        
        # Verify match
        predicted_animal = predictions[0][0]
        is_match = predicted_animal in extracted_animals
        
        if verbose:
            print(f"\n{is_match} \n Text mentions {extracted_animals}, image shows {predicted_animal}")
            print("=" * 80)
        
        return is_match 

def main():
    parser = argparse.ArgumentParser(description='Verify text-image match')
    parser.add_argument('--ner_model', default='models/ner_model')
    parser.add_argument('--classifier_model', default='models/image_classifier')
    parser.add_argument('--text', default=None)
    parser.add_argument('--image', default=None)
    parser.add_argument('--top_k', type=int, default=2)
    
    args = parser.parse_args()
    
    pipeline = TextImageVerificationPipeline(args.ner_model, args.classifier_model)
    
    # Single verification
    if args.text and args.image:
        result = pipeline.verify(args.text, args.image, top_k=args.top_k)
        sys.exit(0 if result else 1)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE (Ctrl+C to exit)")
    print("=" * 80)
    
    try:
        while True:
            text = input("Text: ").strip()
            if not text:
                continue
            
            image_path = input("Image path: ").strip()
            if not os.path.exists(image_path):
                print("Invalid image path!")
                continue
            
            pipeline.verify(text, image_path, top_k=args.top_k)
    
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == '__main__':
    main()