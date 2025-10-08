from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

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
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def train(self, X_train, y_train):
        pass
    def predict(self, X_test):
        pass 

class ConvolutionalMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

class MnistClassifier:
    def __init__(self, classifier: str):
        if classifier == 'rf':
            self.classifier = RandomForestMnistClassifier()
        elif classifier == 'nn':
            self.classifier = FeedForwardMnistClassifier(input_size=784, hidden_size=128, output_size=10)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mnist_classifier = MnistClassifier(classifier='rf')
    mnist_classifier.train(X_train, y_train)
    predictions = mnist_classifier.predict(X_test)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y_test, predictions))