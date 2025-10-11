"""
NER Model Inference Script for Animal Entity Extraction
"""

import argparse
import json
import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import warnings
warnings.filterwarnings('ignore')


class AnimalNERExtractor:
    """Animal entity extractor using trained NER model"""
    
    def __init__(self, model_path):
        """
        Initialize the NER extractor
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        print(f"Loading model from: {model_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        label_map_path = os.path.join(model_path, 'label_map.json')
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        
        # Define valid animals
        self.valid_animals = {
            'dog', 'cat', 'horse', 'spider', 'butterfly', 'chicken',
            'sheep', 'cow', 'squirrel', 'elephant',
            'dogs', 'cats', 'horses', 'spiders', 'butterflies', 
            'chickens', 'cows', 'squirrels', 'elephants'
        }
        
        print("Model loaded successfully!")
    
    def extract_animals(self, text):
        """
        Extract animal entities from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of extracted animal names
        """
        # Tokenize
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Extract entities
        word_ids = encoding.word_ids(batch_index=0)
        predicted_labels = predictions[0].cpu().numpy()
        
        entities = []
        current_entity = []
        
        for word_id, label_id in zip(word_ids, predicted_labels):
            if word_id is None:
                continue
            
            label = self.id2label[label_id]
            
            if label == 'B-ANIMAL':
                if current_entity:
                    entities.append(' '.join(current_entity))
                current_entity = [tokens[word_id]]
            elif label == 'I-ANIMAL' and current_entity:
                current_entity.append(tokens[word_id])
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Clean and normalize entities
        cleaned_entities = []
        for entity in entities:
            entity = entity.lower().strip('.,!?')
            if entity in self.valid_animals:
                # Normalize plural forms
                if entity.endswith('s') and entity[:-1] in self.valid_animals:
                    entity = entity[:-1]
                cleaned_entities.append(entity)
        
        return list(set(cleaned_entities))  # Remove duplicates
    
    def extract_from_file(self, input_file, output_file=None):
        """
        Extract animals from text file
        
        Args:
            input_file (str): Path to input text file
            output_file (str): Path to output JSON file (optional)
            
        Returns:
            dict: Dictionary with texts and extracted animals
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = []
        for text in texts:
            animals = self.extract_animals(text)
            results.append({
                'text': text,
                'animals': animals
            })
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        
        return results


def main():
    """Main function for command-line inference"""
    parser = argparse.ArgumentParser(description='Extract animal entities from text')
    
    parser.add_argument('--model_path', type=str, default='models/ner_model',
                       help='Path to trained model directory')
    parser.add_argument('--text', type=str, default=None,
                       help='Input text for extraction')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file containing texts (one per line)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AnimalNERExtractor(args.model_path)
    
    print("=" * 80)
    print("ANIMAL NER EXTRACTION")
    print("=" * 80)
    
    # Process single text
    if args.text:
        print(f"\nInput: {args.text}")
        animals = extractor.extract_animals(args.text)
        print(f"Extracted animals: {animals}")
    
    # Process file
    elif args.input_file:
        print(f"\nProcessing file: {args.input_file}")
        results = extractor.extract_from_file(args.input_file, args.output_file)
        
        print(f"\nProcessed {len(results)} texts")
        print("\nSample results:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. Text: {result['text']}")
            print(f"   Animals: {result['animals']}")
    
    # Interactive mode
    else:
        print("\nInteractive mode - Enter text to extract animals (Ctrl+C to exit)")
        try:
            while True:
                text = input("\nEnter text: ").strip()
                if not text:
                    continue
                
                animals = extractor.extract_animals(text)
                print(f"Extracted animals: {animals}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()