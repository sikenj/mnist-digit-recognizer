python
"""
MNIST Digit Recognition with Neural Networks
Basic computer vision project demonstrating fundamental ML skills
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Try TensorFlow, fall back to scikit-learn if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("Using TensorFlow/Keras")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    from sklearn.neural_network import MLPClassifier
    print("TensorFlow not available, using scikit-learn MLP")

def load_mnist_data():
    """Load MNIST data from TensorFlow/Keras or scikit-learn"""
    if TENSORFLOW_AVAILABLE:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X_train, X_test, y_train, y_test = train_test_split(
            mnist.data, mnist.target, test_size=10000, random_state=42
        )
        X_train = X_train.values.reshape(-1, 28, 28)
        X_test = X_test.values.reshape(-1, 28, 28)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return (X_train, y_train), (X_test, y_test)

def plot_sample_digits(X, y, n=10):
    """Plot sample digits from the dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(n):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis('off')
    
    plt.suptitle('MNIST Sample Digits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def build_cnn_model():
    """Build a Convolutional Neural Network for MNIST"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_mlp_model():
    """Build a Multi-Layer Perceptron for MNIST"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=20,
        random_state=42,
        verbose=True
    )
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train model and evaluate performance"""
    if TENSORFLOW_AVAILABLE:
        # Reshape for CNN (add channel dimension)
        X_train_cnn = X_train.reshape(-1, 28, 28, 1)
        X_test_cnn = X_test.reshape(-1, 28, 28, 1)
        
        # Build and train CNN
        print("\nBuilding Convolutional Neural Network...")
        model = build_cnn_model()
        model.summary()
        
        print("\nTraining CNN model...")
        history = model.fit(
            X_train_cnn, y_train,
            batch_size=128,
            epochs=10,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating CNN model...")
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
        
        # Plot training history
        plot_training_history(history)
        
    else:
        # Flatten images for MLP
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Build and train MLP
        print("\nBuilding Multi-Layer Perceptron...")
        model = build_mlp_model()
        
        print("\nTraining MLP model...")
        model.fit(X_train_flat, y_train)
        
        # Evaluate
        print("\nEvaluating MLP model...")
        test_acc = model.score(X_test_flat, y_test)
        y_pred = model.predict(X_test_flat)
    
    return y_pred, test_acc if TENSORFLOW_AVAILABLE else test_acc

def plot_training_history(history):
    """Plot training accuracy and loss"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_misclassified(X_test, y_true, y_pred, n=10):
    """Plot misclassified examples"""
    misclassified_idx = np.where(y_pred != y_true)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return
    
    n = min(n, len(misclassified_idx))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i, idx in enumerate(misclassified_idx[:n]):
        axes[i].imshow(X_test[idx], cmap='gray')
        axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Digits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('misclassified_digits.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("=" * 60)
    print("MNIST DIGIT RECOGNITION PROJECT")
    print("Classic computer vision problem - Handwritten digit classification")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]} pixels")
    
    # 2. Visualize samples
    print("\n2. Visualizing sample digits...")
    plot_sample_digits(X_train[:10], y_train[:10])
    
    # 3. Train and evaluate
    print("\n3. Training model...")
    y_pred, test_acc = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # 4. Results
    print("\n4. Model Performance:")
    print(f"   Test Accuracy: {test_acc:.2%}")
    print(f"   Misclassified: {np.sum(y_pred != y_test)} out of {len(y_test)}")
    
    # 5. Detailed analysis
    print("\n5. Detailed classification report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Visualizations
    print("\n6. Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_misclassified(X_test, y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print(f"Final Model Accuracy: {test_acc:.2%}")
    print("=" * 60)
    
    # Save predictions
    np.save('predictions.npy', y_pred)
    np.save('true_labels.npy', y_test)
    print("\nPredictions saved to 'predictions.npy'")

if __name__ == "__main__":
    main()
