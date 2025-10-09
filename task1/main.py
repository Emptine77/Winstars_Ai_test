from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, epochs=100, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, output_size)
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr,
            weight_decay=1e-4 # L2 regularization
        )

        # Scheduler for learning rate decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr * 10,
            epochs=epochs,
            steps_per_epoch=438,  # Assuming 438 batches per epoch for MNIST with batch_size=128
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(self.device)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(self.device)
        
        # DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  # Shuffle the data
            pin_memory=True if self.device.type == 'cuda' else False  # Pin memory for faster transfers to GPU
        )
        
        best_loss = float('inf') # best validation loss for early stopping
        patience = 15 # epochs to wait for improvement before stopping
        patience_counter = 0 # counter for early stopping
        best_model_state = None # to store the best model state
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing
                
                outputs = self.model(batch_X) 
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
             
            avg_train_loss = epoch_loss / len(dataloader)
            train_accuracy = 100 * correct / total
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = 100 * (val_preds == y_val_tensor).sum().item() / len(y_val_tensor)
           
            # Early stopping 
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 5 == 0 or patience_counter == patience:
                print(f"Epoch [{epoch+1}/{self.epochs}] "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model from training")

    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        predictions = []
        # Batch prediction for large datasets
        batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                outputs = self.model(batch)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)

class ConvolutionalMnistClassifier(MnistClassifierInterface):
    def __init__(self, lr=0.001, epochs=50, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            # First conv block: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Dropout2d(0.2),
            
            # Second conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Dropout2d(0.2),
            
            # Flatten
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(64 * 7 * 7, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 10)  # 10 classes for MNIST
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        # Reshape from (N, 784) to (N, 1, 28, 28) for CNN
        X_train_tensor = torch.tensor(
            X_train_np.reshape(-1, 1, 28, 28), 
            dtype=torch.float32
        )
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
        
        X_val_tensor = torch.tensor(
            X_val_np.reshape(-1, 1, 28, 28), 
            dtype=torch.float32
        ).to(self.device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.long).to(self.device)
        
        # DataLoader 
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            avg_train_loss = epoch_loss / len(dataloader)
            train_accuracy = 100 * correct / total
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = 100 * (val_preds == y_val_tensor).sum().item() / len(y_val_tensor)
                
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or patience_counter == patience:
                print(f"Epoch [{epoch+1}/{self.epochs}] "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            self.scheduler.step(val_loss)
            # Clear cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model from training")

    def predict(self, X_test):
        print("Starting prediction...")
        
        # Convert to numpy if DataFrame
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Reshape from (N, 784) to (N, 1, 28, 28)
        X_test_tensor = torch.tensor(
            X_test_np.reshape(-1, 1, 28, 28), 
            dtype=torch.float32
        )
        
        self.model.eval()
        predictions = []
        batch_size = 1000
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size].to(self.device)
                outputs = self.model(batch)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        print("Prediction complete")
        return np.array(predictions)

class MnistClassifier:
    def __init__(self, classifier: str):
        if classifier == 'rf':
            self.classifier = RandomForestMnistClassifier()
        elif classifier == 'nn':
            self.classifier = FeedForwardMnistClassifier(input_size=784, hidden_size=700, output_size=10)
        elif classifier == 'cnn':
            self.classifier = ConvolutionalMnistClassifier()
        else:
            raise ValueError("Unsupported classifier type")

    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target 
    X = X.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype(np.int64)  # Convert target to integer type
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Enter which model to use: 'rf' (random forest), 'nn' (feedforward neural network), 'cnn' (not implemented).")
    print("Type 'Exit' to quit.")
    while True:
        choice = input("Model> ").strip()
        if choice.lower() == 'exit':
            print("Exiting.")
            break
        if choice not in ('rf', 'nn', 'cnn'):
            print("Invalid choice. Please enter 'rf', 'nn', 'cnn' or 'Exit'.")
            continue
        else:
            print(f"You selected: {choice} model.")
            mnist_classifier = MnistClassifier(classifier=choice)
        try:
            print("Training model...")
            mnist_classifier.train(X_train, y_train)
            predictions = mnist_classifier.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"Accuracy on subset: {acc:.4f}")
        except Exception as e:
            print("An error occurred while training or evaluating the model:", e)
            print("Please try again or type 'Exit' to quit.")