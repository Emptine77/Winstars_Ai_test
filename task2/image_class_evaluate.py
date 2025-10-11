"""
Image Classification Inference Script for Animal Recognition
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class AnimalClassifier:
    """Animal image classifier"""
    
    def __init__(self, model_path):
        """
        Initialize the classifier
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class mapping
        class_mapping_path = os.path.join(model_path, 'class_mapping.json')
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)
        
        self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        self.num_classes = len(self.idx_to_class)
        
        print(f"Loading model from: {model_path}")
        print(f"Number of classes: {self.num_classes}")
        
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
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")
    
    def predict(self, image_path, top_k=1):
        """
        Predict the class of an image
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Return top k predictions
            
        Returns:
            list: List of tuples (class_name, probability)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_class[idx.item()]
            predictions.append((class_name, prob.item()))
        
        return predictions
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of predictions for each image
        """
        results = []
        for img_path in image_paths:
            predictions = self.predict(img_path, top_k=3)
            results.append({
                'image': img_path,
                'predictions': predictions
            })
        return results


def main():
    """Main function for command-line inference"""
    parser = argparse.ArgumentParser(description='Classify animal images')
    
    parser.add_argument('--model_path', type=str, default='models/image_classifier',
                       help='Path to trained model directory')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images to classify')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Return top k predictions')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = AnimalClassifier(args.model_path)
    
    print("\n" + "=" * 80)
    print("ANIMAL IMAGE CLASSIFICATION")
    print("=" * 80)
    
    # Process single image
    if args.image:
        print(f"\nClassifying: {args.image}")
        predictions = classifier.predict(args.image, args.top_k)
        
        print("\nPredictions:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
    
    # Process directory
    elif args.image_dir:
        print(f"\nProcessing directory: {args.image_dir}")
        
        image_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.image_dir, file))
        
        print(f"Found {len(image_paths)} images")
        
        results = classifier.predict_batch(image_paths)
        
        # Display results
        print("\nResults:")
        for result in results[:args.top_k]:  # Show top_k results
            print(f"\n{os.path.basename(result['image'])}:")
            for class_name, prob in result['predictions']:
                print(f"  - {class_name}: {prob*100:.2f}%")
        
        # Save to file
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
    
    # Interactive mode
    else:
        print("\nInteractive mode - Enter image path to classify (Ctrl+C to exit)")
        try:
            while True:
                image_path = input("\nImage path: ").strip()
                if not image_path or not os.path.exists(image_path):
                    print("Invalid image path!")
                    continue
                
                predictions = classifier.predict(image_path, args.top_k)
                print("\nPredictions:")
                for i, (class_name, prob) in enumerate(predictions, 1):
                    print(f"{i}. {class_name}: {prob*100:.2f}%")
        except KeyboardInterrupt:
            print("\n\nExiting...")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()