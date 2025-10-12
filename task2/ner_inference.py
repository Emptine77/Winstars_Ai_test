"""
Optimized NER Model Inference for Animal Entity Extraction
"""

import argparse
import json
import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification


class AnimalNERExtractor:
    """
    Animal entity extractor using trained NER model
    
    This class loads a pre-trained BERT-based NER model and extracts animal entities
    from input text. It uses BIO (Begin-Inside-Outside) tagging scheme to identify
    and extract animal mentions from natural language text.
    
    Attributes:
        device (torch.device): Computation device (CPU or CUDA)
        tokenizer (BertTokenizerFast): BERT tokenizer for text processing
        model (BertForTokenClassification): Trained NER model
        id2label (dict): Mapping from label IDs to label names
        valid_animals (set): Set of valid animal names for validation
    """
    def __init__(self, model_path):
        """
        Initialize the NER extractor with a trained model
        
        Args:
            model_path (str): Path to the directory containing trained model files
                             (model weights, tokenizer config, label mapping)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {model_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        
        # Valid animals
        self.valid_animals = {
            'dog', 'cat', 'horse', 'spider', 'butterfly', 'chicken',
            'sheep', 'cow', 'squirrel', 'elephant',
            'dogs', 'cats', 'horses', 'spiders', 'butterflies', 
            'chickens', 'cows', 'squirrels', 'elephants'
        }
        
        print("Model loaded successfully!")
    
    def extract_animals(self, text):
        """
        Extract animal entities from input text

        Args:
            text (str): Input text to extract animals from
            
        Returns:
            list: List of unique animal names found in text
        """
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
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
        
        # Clean and normalize
        cleaned = []
        for entity in entities:
            entity = entity.lower().strip('.,!?')
            if entity in self.valid_animals:
                # Normalize plural
                if entity.endswith('s') and entity[:-1] in self.valid_animals:
                    entity = entity[:-1]
                cleaned.append(entity)
        
        return list(set(cleaned))


def main():
    parser = argparse.ArgumentParser(description='Extract animal entities from text')
    parser.add_argument('--model_path', default='models/ner_model')
    parser.add_argument('--text', default=None)
    args = parser.parse_args()
    
    extractor = AnimalNERExtractor(args.model_path)
    
    print("=" * 80)
    print("ANIMAL NER EXTRACTION")
    print("=" * 80)
    
    # Single text
    if args.text:
        print(f"\nInput: {args.text}")
        animals = extractor.extract_animals(args.text)
        print(f"Extracted animals: {animals}")
    
    # Interactive mode
    else:
        print("\nInteractive mode")
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