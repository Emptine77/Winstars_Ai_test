"""
Enhanced NER Model Training Script for Animal Entity Recognition
Uses BERT-based transformer for sequence labeling with synthetic and diverse data.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# ================================
# Synthetic NER Data Generator
# ================================
def create_enhanced_ner_data(num_samples=5000):
    """ 
    Create diverse synthetic NER data for animals with realistic sentences
    
    This function generates a large dataset of labeled sentences for training
    a Named Entity Recognition model.
    The data is automatically labeled using BIO tagging scheme:
    - B-ANIMAL: Beginning of an animal mention
    - I-ANIMAL: Inside/continuation of an animal mention
    - O: Outside any animal mention
    
    Args:
        num_samples (int): Number of synthetic sentences to generate (default: 5000)
        
    Returns:
        list: List of dictionaries, each containing:
              - 'tokens': list of word tokens
              - 'labels': list of corresponding BIO labels
    """
    animals = [
        'dog', 'cat', 'horse', 'spider', 'butterfly', 'chicken',
        'sheep', 'cow', 'squirrel', 'elephant'
    ]
    descriptors = ["big", "small", "black", "white", "young", "old", "happy", "sleepy", "curious", "lazy"]
    actions = ["running", "jumping", "sleeping", "playing", "eating", "sitting", "roaming", "climbing"]
    locations = [
        "in the garden", "on the street", "near the house", "under the tree",
        "on the sofa", "by the river", "in the park", "on the hill", "at the zoo", "on the balcony"
    ]
    templates = [
        "There is a {} {} {}.",
        "I saw a {} {} {} today.",
        "Look at the {} {} {}!",
        "A {} {} {} is nearby.",
        "Maybe a {} {} {} is here.",
        "Do you see a {} {} {}?",
        "I think a {} {} {} passed by.",
        "Someone spotted a {} {} {}.",
        "A {} {} {} suddenly appears.",
        "Have you noticed a {} {} {}?",
        "Check out the {} {} {}.",
        "The {} {} {} seems interesting.",
        "It's amazing how a {} {} {} looks.",
        "There's a {} {} {} over there.",
        "Could it be a {} {} {}?",
        "I just saw a {} {} {}!",
        "What a {} {} {} this is!",
        "The picture shows a {} {} {}.",
        "You can see a {} {} {} clearly.",
        "A {} {} {} is in the frame."
    ]

    real_sentences = [
        "My dog loves to play in the park.",
        "A cat is sleeping on the sofa.",
        "The horse is running across the field.",
        "I watched a butterfly landing on the flower.",
        "She saw a squirrel climbing the tree.",
        "An elephant is walking slowly in the zoo.",
        "The chicken is pecking the ground.",
        "A sheep is grazing near the river."
    ]

    data = []

    for _ in range(num_samples):
        animal = random.choice(animals)
        descriptor = random.choice(descriptors)
        action = random.choice(actions)
        location = random.choice(locations)
        template = random.choice(templates)

        sentence = template.format(descriptor, animal, location)
        if random.random() < 0.5:
            sentence += f" It is {action}."

        tokens = sentence.split()
        labels = ['O'] * len(tokens)

        animal_tokens = animal.split()
        for i in range(len(tokens) - len(animal_tokens) + 1):
            if [t.lower() for t in tokens[i:i + len(animal_tokens)]] == [t.lower() for t in animal_tokens]:
                labels[i] = 'B-ANIMAL'
                for j in range(1, len(animal_tokens)):
                    labels[i + j] = 'I-ANIMAL'
                break

        data.append({'tokens': tokens, 'labels': labels})

    for sent in real_sentences:
        tokens = sent.split()
        labels = ['O'] * len(tokens)
        for i, token in enumerate(tokens):
            for animal in animals:
                animal_tokens = animal.split()
                if [t.lower() for t in tokens[i:i + len(animal_tokens)]] == [t.lower() for t in animal_tokens]:
                    labels[i] = 'B-ANIMAL'
                    for j in range(1, len(animal_tokens)):
                        labels[i + j] = 'I-ANIMAL'
                    break
        data.append({'tokens': tokens, 'labels': labels})

    for _ in range(int(num_samples * 0.1)):
        sentence = " ".join(random.choices(
            ["The", "weather", "is", "nice", "today", "I", "love", "reading",
             "books", "walking", "in", "park", "with", "friends", "playing", "games", "it", "looks", "beautiful"],
            k=6
        ))
        tokens = sentence.split()
        labels = ['O'] * len(tokens)
        data.append({'tokens': tokens, 'labels': labels})

    random.shuffle(data)
    return data


# ================================
# Dataset
# ================================
class NERDataset(Dataset):
    """
    PyTorch Dataset for NER
    
    Attributes:
        texts (list): List of tokenized sentences
        labels (list): List of label sequences
        tokenizer (BertTokenizerFast): BERT tokenizer
        max_len (int): Maximum sequence length
        label2id (dict): Mapping from label names to IDs
        id2label (dict): Mapping from IDs to label names
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        Initialize NER dataset
        
        Args:
            texts (list): List of tokenized texts (list of word lists)
            labels (list): List of label sequences
            tokenizer (BertTokenizerFast): Tokenizer for encoding
            max_len (int): Maximum sequence length (default: 128)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'O': 0, 'B-ANIMAL': 1, 'I-ANIMAL': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[labels[word_id]])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# ================================
# Training and Evaluation
# ================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train model for one epoch
    
    Args:
        model: NER model to train
        dataloader: Training data loader
        optimizer: Optimizer for weight updates
        scheduler: Learning rate scheduler
        device: Computation device (CPU/GPU)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    """
    Evaluate model on validation/test set
    
    Uses seqeval library for proper sequence labeling metrics.
    Metrics are computed at the entity level (not token level).
    
    Args:
        model: NER model to evaluate
        dataloader: Validation/test data loader
        device: Computation device
        id2label (dict): Mapping from label IDs to names
        
    Returns:
        tuple: (f1_score, precision, recall, predictions, true_labels)
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            for pred, label, mask in zip(preds, labels, attention_mask):
                pred_labels = []
                true_label_list = []
                for p, l, m in zip(pred, label, mask):
                    if m.item() == 1 and l.item() != -100:
                        pred_labels.append(id2label[p.item()])
                        true_label_list.append(id2label[l.item()])

                if pred_labels:
                    predictions.append(pred_labels)
                    true_labels.append(true_label_list)

    f1 = f1_score(true_labels, predictions, mode='strict')
    precision = precision_score(true_labels, predictions, mode='strict')
    recall = recall_score(true_labels, predictions, mode='strict')

    return f1, precision, recall, predictions, true_labels


def train_ner_model(args):
    """
    Main training function for NER model

    Args:
        args: Command-line arguments containing:
              - model_name: Pretrained BERT model to use
              - output_dir: Where to save trained model
              - max_len: Maximum sequence length
              - batch_size: Training batch size
              - num_epochs: Number of training epochs
              - learning_rate: Learning rate
              - seed: Random seed for reproducibility
              - num_samples: Number of synthetic samples to generate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    data = create_enhanced_ner_data(num_samples=args.num_samples)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_dataset = NERDataset(
        [d['tokens'] for d in train_data],
        [d['labels'] for d in train_data],
        tokenizer,
        args.max_len
    )
    val_dataset = NERDataset(
        [d['tokens'] for d in val_data],
        [d['labels'] for d in val_data],
        tokenizer,
        args.max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    label2id = train_dataset.label2id
    id2label = train_dataset.id2label

    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        f1, precision, recall, _, _ = evaluate(model, val_loader, device, id2label)
        print(f"Validation F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            print("F1 improved, saving model...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
                json.dump({'label2id': label2id, 'id2label': id2label}, f)

    print("\nTraining complete.")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train enhanced NER model for animal recognition')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='models/ner_model', help='Output directory')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of synthetic sentences')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_ner_model(args)
