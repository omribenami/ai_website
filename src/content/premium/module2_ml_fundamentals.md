# Module 2: Machine Learning Fundamentals

## Overview

Machine Learning (ML) forms the foundation of modern AI systems. Before diving into deep learning and neural networks, it's essential to understand the core concepts, algorithms, and evaluation methods of traditional machine learning. This module will provide you with a solid understanding of ML fundamentals that will serve as building blocks for more advanced AI techniques.

## Types of Machine Learning

Machine learning algorithms can be categorized into several types based on how they learn:

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data, trying to find a function that maps inputs to known outputs.

**Key Characteristics:**
- Requires labeled data (input-output pairs)
- Goal is to learn a mapping function from inputs to outputs
- Performance can be clearly measured against known correct answers
- Used for classification and regression tasks

**Common Applications:**
- Email spam detection (classification)
- House price prediction (regression)
- Medical diagnosis (classification)
- Stock price forecasting (regression)

### Unsupervised Learning

Unsupervised learning algorithms work with unlabeled data, trying to find patterns or structure within the data.

**Key Characteristics:**
- Works with unlabeled data
- Goal is to discover hidden patterns or intrinsic structures
- No explicit feedback on performance
- Used for clustering, dimensionality reduction, and association tasks

**Common Applications:**
- Customer segmentation (clustering)
- Anomaly detection (outlier analysis)
- Feature extraction (dimensionality reduction)
- Market basket analysis (association)

### Reinforcement Learning

Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to maximize some notion of cumulative reward.

**Key Characteristics:**
- Based on interaction with an environment
- Learns through trial and error
- Delayed reward system
- Balance between exploration (trying new actions) and exploitation (using known good actions)

**Common Applications:**
- Game playing (AlphaGo, chess engines)
- Robotics control
- Autonomous vehicles
- Resource management

### Semi-Supervised Learning

Semi-supervised learning falls between supervised and unsupervised learning, using a small amount of labeled data with a large amount of unlabeled data.

**Key Characteristics:**
- Uses both labeled and unlabeled data
- Particularly useful when labeling data is expensive or time-consuming
- Can achieve good performance with less labeled data

**Common Applications:**
- Web content classification
- Speech recognition
- Medical image analysis

## The Machine Learning Workflow

A typical machine learning project follows these steps:

1. **Problem Definition**
   - Define the problem and objectives
   - Determine the type of ML task (classification, regression, etc.)
   - Establish success metrics

2. **Data Collection**
   - Gather relevant data from various sources
   - Ensure data quality and quantity
   - Consider ethical and privacy implications

3. **Data Exploration and Analysis**
   - Understand data distributions and relationships
   - Identify patterns, outliers, and missing values
   - Generate insights to guide feature engineering

4. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale or normalize numerical features
   - Split data into training, validation, and test sets

5. **Feature Engineering**
   - Create new features from existing ones
   - Select relevant features
   - Reduce dimensionality if needed

6. **Model Selection and Training**
   - Choose appropriate algorithms
   - Train models on the training data
   - Tune hyperparameters

7. **Model Evaluation**
   - Assess performance on validation data
   - Use appropriate metrics for the task
   - Compare different models

8. **Model Deployment**
   - Implement the model in a production environment
   - Set up monitoring and maintenance
   - Plan for model updates

9. **Iteration**
   - Gather feedback from deployment
   - Refine the model as needed
   - Retrain with new data

## Training, Validation, and Test Sets

Properly splitting your data is crucial for developing models that generalize well:

### Training Set
- Used to train the model
- Typically 60-80% of the data
- The model learns patterns from this data

### Validation Set
- Used to tune hyperparameters and evaluate during development
- Typically 10-20% of the data
- Helps prevent overfitting to the training data

### Test Set
- Used only for final evaluation
- Typically 10-20% of the data
- Simulates how the model will perform on unseen data
- Should never be used for training or hyperparameter tuning

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

## Cross-Validation

Cross-validation is a technique to assess how well a model generalizes to independent data:

### K-Fold Cross-Validation
1. Split the data into K equal folds
2. Train the model K times, each time using a different fold as validation and the rest as training
3. Average the performance across all K iterations

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}, Standard deviation: {scores.std():.4f}")
```

### Stratified K-Fold
Ensures that each fold maintains the same class distribution as the original dataset, important for imbalanced datasets:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict(X_val_fold)
    fold_score = accuracy_score(y_val_fold, y_pred)
    scores.append(fold_score)

print(f"Stratified K-Fold scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.4f}, Standard deviation: {np.std(scores):.4f}")
```

## Overfitting and Underfitting

Understanding and addressing these common problems is essential for building effective models:

### Overfitting
- Model learns the training data too well, including noise and outliers
- Performs well on training data but poorly on new data
- Complex model with too many parameters relative to the amount of training data

**Signs of Overfitting:**
- Large gap between training and validation performance
- Model performs increasingly worse as it sees more data

**Solutions:**
- Collect more training data
- Use regularization techniques
- Simplify the model
- Apply early stopping
- Use dropout (for neural networks)
- Implement data augmentation

### Underfitting
- Model is too simple to capture the underlying pattern in the data
- Performs poorly on both training and validation data

**Signs of Underfitting:**
- Poor performance on training data
- Similar poor performance on validation data
- Training and validation errors are both high

**Solutions:**
- Use a more complex model
- Add more features
- Reduce regularization
- Train longer (for iterative algorithms)

### The Bias-Variance Tradeoff

The relationship between model complexity, bias, and variance:

- **Bias**: Error from incorrect assumptions in the learning algorithm
  - High bias = underfitting
  - Simplified models tend to have high bias

- **Variance**: Error from sensitivity to small fluctuations in the training set
  - High variance = overfitting
  - Complex models tend to have high variance

- **Tradeoff**: Reducing bias typically increases variance and vice versa
  - The goal is to find the sweet spot with the right model complexity

## Model Evaluation Metrics

Different metrics are appropriate for different types of machine learning tasks:

### Classification Metrics

**Accuracy**
- Proportion of correct predictions
- Simple but can be misleading for imbalanced classes
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

**Precision**
- Proportion of positive identifications that were actually correct
- Focus on minimizing false positives
```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
```

**Recall (Sensitivity)**
- Proportion of actual positives that were correctly identified
- Focus on minimizing false negatives
```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
```

**F1 Score**
- Harmonic mean of precision and recall
- Balances precision and recall
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
```

**Confusion Matrix**
- Table showing true positives, false positives, true negatives, and false negatives
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

**ROC Curve and AUC**
- Receiver Operating Characteristic curve plots true positive rate vs. false positive rate
- Area Under the Curve (AUC) quantifies overall performance
```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

### Regression Metrics

**Mean Absolute Error (MAE)**
- Average of absolute differences between predicted and actual values
- Less sensitive to outliers than MSE
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

**Mean Squared Error (MSE)**
- Average of squared differences between predicted and actual values
- Penalizes larger errors more heavily
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

**Root Mean Squared Error (RMSE)**
- Square root of MSE
- Same units as the target variable
```python
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**R-squared (Coefficient of Determination)**
- Proportion of variance in the dependent variable that is predictable
- Range: 0 to 1 (higher is better)
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

## Classical Machine Learning Algorithms

Let's explore some fundamental machine learning algorithms:

### Linear and Logistic Regression

**Linear Regression**
- Predicts a continuous target variable based on one or more features
- Assumes a linear relationship between features and target
- Simple, interpretable, but limited to linear relationships

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X_new, y_pred, 'r-', linewidth=2, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()

print(f"Model intercept: {model.intercept_[0]:.2f}")
print(f"Model coefficient: {model.coef_[0][0]:.2f}")
```

**Logistic Regression**
- Despite the name, used for binary classification
- Predicts the probability of an instance belonging to a particular class
- Uses the logistic function to constrain output between 0 and 1

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7)

# Create a mesh grid
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict for each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.colorbar()
plt.show()
```

### Decision Trees and Random Forests

**Decision Trees**
- Hierarchical model that makes decisions based on asking a sequence of questions
- Can handle both classification and regression tasks
- Intuitive and easy to interpret, but prone to overfitting

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree for Iris Dataset")
plt.show()
```

**Random Forests**
- Ensemble method that builds multiple decision trees and merges their predictions
- Reduces overfitting by averaging out the predictions
- Generally more accurate than individual decision trees

```python
from sklearn.ensemble import RandomForestClassifier

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [iris.feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

### Support Vector Machines

- Finds the hyperplane that best separates classes with the maximum margin
- Effective in high-dimensional spaces
- Can use different kernel functions to handle non-linear boundaries
- Good for classification and regression, but can be computationally intensive

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with different kernels
kernels = ['linear', 'poly', 'rbf']
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels):
    # Train SVM
    model = SVC(kernel=kernel, gamma='auto', random_state=42)
    model.fit(X_train_scaled[:, :2], y_train)  # Using only first two features for visualization
    
    # Evaluate
    y_pred = model.predict(X_test_scaled[:, :2])
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create a mesh grid for visualization
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', edgecolors='k')
    plt.title(f'{kernel.upper()} Kernel (Accuracy: {accuracy:.4f})')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')

plt.tight_layout()
plt.show()
```

### K-Means Clustering

- Unsupervised learning algorithm that partitions data into K clusters
- Each data point belongs to the cluster with the nearest mean
- Simple and efficient, but requires specifying the number of clusters

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Train the model
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Determining the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
```

## Feature Engineering and Data Preprocessing

Effective feature engineering and preprocessing are often the keys to successful machine learning models:

### Data Cleaning Techniques

**Handling Missing Values**
```python
import pandas as pd
import numpy as np

# Create a dataset with missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
})

print("Original data:")
print(data)

# 1. Remove rows with missing values
data_dropna = data.dropna()
print("\nAfter dropping rows with missing values:")
print(data_dropna)

# 2. Fill missing values with mean
data_fillmean = data.fillna(data.mean())
print("\nAfter filling missing values with mean:")
print(data_fillmean)

# 3. Fill missing values with median
data_fillmedian = data.fillna(data.median())
print("\nAfter filling missing values with median:")
print(data_fillmedian)

# 4. Fill missing values with a specific value
data_fillvalue = data.fillna(0)
print("\nAfter filling missing values with 0:")
print(data_fillvalue)

# 5. Forward fill (use previous value)
data_ffill = data.ffill()
print("\nAfter forward fill:")
print(data_ffill)

# 6. Backward fill (use next value)
data_bfill = data.bfill()
print("\nAfter backward fill:")
print(data_bfill)
```

**Handling Outliers**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataset with outliers
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data = np.append(data, [10, -10, 8, -8])  # Add outliers

plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
sns.boxplot(data)
plt.title('Original Data with Outliers')

# 1. Remove outliers using IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]

plt.subplot(1, 3, 2)
sns.boxplot(data_no_outliers)
plt.title('After Removing Outliers (IQR Method)')

# 2. Cap outliers (Winsorization)
def cap_outliers(x, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(x, lower_percentile)
    upper_bound = np.percentile(x, upper_percentile)
    return np.clip(x, lower_bound, upper_bound)

data_capped = cap_outliers(data)

plt.subplot(1, 3, 3)
sns.boxplot(data_capped)
plt.title('After Capping Outliers (Winsorization)')

plt.tight_layout()
plt.show()
```

### Feature Selection and Extraction

**Feature Selection Methods**
```python
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 1. Filter Method: SelectKBest
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)
selected_features = [data.feature_names[i] for i in selector.get_support(indices=True)]

print("Top 5 features selected by SelectKBest:")
for i, feature in enumerate(selected_features):
    print(f"{i+1}. {feature}")

# 2. Wrapper Method: Recursive Feature Elimination (RFE)
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)
selected_features_rfe = [data.feature_names[i] for i in range(len(data.feature_names)) if rfe.support_[i]]

print("\nTop 5 features selected by RFE:")
for i, feature in enumerate(selected_features_rfe):
    print(f"{i+1}. {feature}")

# 3. Embedded Method: Feature Importance from Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
top_features = [data.feature_names[i] for i in sorted_idx[:5]]

print("\nTop 5 features selected by Random Forest Importance:")
for i, feature in enumerate(top_features):
    print(f"{i+1}. {feature} (Importance: {feature_importance[sorted_idx[i]]:.4f})")
```

**Dimensionality Reduction**
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')

# 2. t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('t-SNE: 2D Embedding')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(label='Class')

plt.tight_layout()
plt.show()

# Explained variance ratio for PCA
pca = PCA().fit(X_scaled)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend()
plt.grid(True)
plt.show()
```

### Handling Imbalanced Data

Imbalanced datasets can lead to biased models that perform poorly on minority classes:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Generate imbalanced dataset
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.95, 0.05], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check class distribution
print("Original class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# Train a model on imbalanced data
model_imbalanced = RandomForestClassifier(random_state=42)
model_imbalanced.fit(X_train, y_train)
y_pred_imbalanced = model_imbalanced.predict(X_test)

print("\nClassification Report (Imbalanced Data):")
print(classification_report(y_test, y_pred_imbalanced))

# 1. Oversampling with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
unique, counts = np.unique(y_train_smote, return_counts=True)
print(dict(zip(unique, counts)))

model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

print("\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote))

# 2. Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

print("\nClass distribution after Undersampling:")
unique, counts = np.unique(y_train_under, return_counts=True)
print(dict(zip(unique, counts)))

model_under = RandomForestClassifier(random_state=42)
model_under.fit(X_train_under, y_train_under)
y_pred_under = model_under.predict(X_test)

print("\nClassification Report (Undersampling):")
print(classification_report(y_test, y_pred_under))

# 3. Combination: SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = smote_tomek.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE + Tomek Links:")
unique, counts = np.unique(y_train_combined, return_counts=True)
print(dict(zip(unique, counts)))

model_combined = RandomForestClassifier(random_state=42)
model_combined.fit(X_train_combined, y_train_combined)
y_pred_combined = model_combined.predict(X_test)

print("\nClassification Report (SMOTE + Tomek Links):")
print(classification_report(y_test, y_pred_combined))

# Visualize the results
plt.figure(figsize=(20, 5))

# Original data
plt.subplot(1, 4, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('Original Imbalanced Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# SMOTE
plt.subplot(1, 4, 2)
plt.scatter(X_train_smote[:, 0], X_train_smote[:, 1], c=y_train_smote, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('After SMOTE Oversampling')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Undersampling
plt.subplot(1, 4, 3)
plt.scatter(X_train_under[:, 0], X_train_under[:, 1], c=y_train_under, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('After Undersampling')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# SMOTE + Tomek Links
plt.subplot(1, 4, 4)
plt.scatter(X_train_combined[:, 0], X_train_combined[:, 1], c=y_train_combined, cmap='viridis', alpha=0.8, edgecolors='k')
plt.title('After SMOTE + Tomek Links')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## Practical Exercise: End-to-End Machine Learning Project

Let's apply what we've learned to a real-world dataset:

```python
# Save this as ml_project.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import fetch_openml

# Load the dataset (Titanic)
print("Loading dataset...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
X = titanic.data
y = titanic.target

print(f"Dataset shape: {X.shape}")
print("\nFeature overview:")
print(X.info())

print("\nTarget distribution:")
print(y.value_counts(normalize=True))

# Data exploration
print("\nExploring data...")
plt.figure(figsize=(12, 10))

# Age distribution
plt.subplot(2, 2, 1)
sns.histplot(X['age'].dropna(), kde=True)
plt.title('Age Distribution')

# Fare distribution
plt.subplot(2, 2, 2)
sns.histplot(X['fare'].dropna(), kde=True)
plt.title('Fare Distribution')

# Survival by sex
plt.subplot(2, 2, 3)
survival_by_sex = pd.crosstab(X['sex'], y)
survival_by_sex.div(survival_by_sex.sum(1), axis=0).plot(kind='bar', stacked=False)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')

# Survival by class
plt.subplot(2, 2, 4)
survival_by_class = pd.crosstab(X['pclass'], y)
survival_by_class.div(survival_by_class.sum(1), axis=0).plot(kind='bar', stacked=False)
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')

plt.tight_layout()
plt.savefig('titanic_exploration.png')
plt.close()

# Data preprocessing
print("\nPreprocessing data...")
# Identify numeric and categorical features
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass']

# Define preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compare different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(models.items()):
    print(f"\nTraining {name}...")
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(2, 2, i+1)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Feature importance
if hasattr(best_model['model'], 'feature_importances_'):
    # Get feature names after preprocessing
    ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    feature_names = numeric_features.copy()
    for i, category in enumerate(categorical_features):
        feature_names.extend([f"{category}_{val}" for val in ohe.categories_[i]])
    
    # Get feature importances
    importances = best_model['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

print("\nProject completed! Check the generated visualizations.")
```

## Resources for Further Learning

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) - Comprehensive documentation for machine learning in Python
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Excellent book by Aurélien Géron
- [Machine Learning Mastery](https://machinelearningmastery.com/) - Practical tutorials and guides
- [Kaggle Competitions](https://www.kaggle.com/competitions) - Practice with real-world datasets and problems
- [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer) - Clear explanations of machine learning concepts

In the next module, we'll build on these machine learning fundamentals to explore deep learning and neural networks, which will allow us to tackle more complex problems in computer vision, natural language processing, and more.
