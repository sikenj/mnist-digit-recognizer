"""
MNIST Digit Recognition (scikit-learn version)
Lightweight version for GitHub Codespaces
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

print("=" * 60)
print("MNIST DIGIT RECOGNITION - Lightweight Version")
print("Using scikit-learn MLP Classifier")
print("=" * 60)

# 1. Load MNIST data
print("\n1. Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Take subset for faster training (10% of data)
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)

# 2. Split data
print("2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")

# 3. Train MLP classifier
print("\n3. Training MLP Classifier...")
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42,
    verbose=True
)
mlp.fit(X_train, y_train)

# 4. Evaluate
print("\n4. Evaluating model...")
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
print(f"Misclassified: {np.sum(y_pred != y_test)} out of {len(y_test)}")

# 5. Visualize samples
print("\n5. Creating visualizations...")

# Reshape for plotting
X_train_images = X_train.values.reshape(-1, 28, 28)

# Plot first 10 digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_images[i], cmap='gray')
    ax.set_title(f"Label: {y_train.iloc[i]}")
    ax.axis('off')
plt.suptitle('MNIST Sample Digits', fontsize=16)
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=120)
print("   ✓ Saved: mnist_samples.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
print("   ✓ Saved: confusion_matrix.png")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print(f"Final Accuracy: {accuracy:.2%}")
print("=" * 60)
