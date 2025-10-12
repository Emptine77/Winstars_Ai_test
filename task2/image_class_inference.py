"""
Optimized Image Classification Inference for Animal Recognition
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class mapping
        with open(os.path.join(model_path, 'class_mapping.json'), 'r') as f:
            mapping = json.load(f)
        
        self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        self.num_classes = len(self.idx_to_class)
        
        print(f"Loading model from: {model_path}")
        print(f"Number of classes: {self.num_classes}")
        
        # Create model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
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
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")
    
    def predict(self, image_path, top_k=1):
        """Predict the class of an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_class[idx.item()]
            predictions.append((class_name, prob.item()))
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Classify animal images')
    parser.add_argument('--model_path', default='models/image_classifier')
    parser.add_argument('--image', default=None)
    parser.add_argument('--top_k', type=int, default=3)
    args = parser.parse_args()
    
    classifier = AnimalClassifier(args.model_path)
    
    print("\n" + "=" * 80)
    print("ANIMAL IMAGE CLASSIFICATION")
    print("=" * 80)
    
    # Single image
    if args.image:
        print(f"\nClassifying: {args.image}")
        predictions = classifier.predict(args.image, args.top_k)
        
        print("\nPredictions:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
    
    # Interactive mode
    else:
        print("\nInteractive mode (Ctrl+C to exit)")
        try:
            while True:
                image_path = input("\nImage path: ").strip()
                if not os.path.exists(image_path):
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