# Module 1.3: Python for AI - Essential Skills

## Overview

While you already have some Python experience and can understand code that isn't highly complex, AI development requires specific Python skills and libraries. This section will focus on the Python knowledge most relevant to AI development, helping you build on your existing foundation.

## Python Libraries for AI and Machine Learning

The Python ecosystem for AI is rich with specialized libraries. Here are the essential ones you'll need to master:

### NumPy: Numerical Computing Foundation

NumPy is the fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.

```python
import numpy as np

# Creating arrays
x = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Array operations
print(x + 10)  # Element-wise addition
print(x * 2)   # Element-wise multiplication

# Matrix operations
print(matrix.T)  # Transpose
print(np.dot(matrix, matrix))  # Matrix multiplication

# Statistical operations
print(np.mean(x))
print(np.std(matrix))

# Random number generation
random_array = np.random.rand(3, 3)  # 3x3 array of random values
```

**Key NumPy Skills for AI:**
- Array creation and manipulation
- Broadcasting (applying operations to arrays of different shapes)
- Vectorized operations (avoiding loops for better performance)
- Linear algebra operations
- Random number generation for stochastic processes

### Pandas: Data Manipulation and Analysis

Pandas provides data structures and functions needed to manipulate structured data efficiently.

```python
import pandas as pd

# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [85, 92, 78, 95]
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())
print(df.describe())  # Statistical summary
print(df['Age'].mean())  # Column mean

# Filtering
young_people = df[df['Age'] < 30]
high_scorers = df[df['Score'] > 90]

# Grouping and aggregation
grouped = df.groupby('Age').mean()

# Handling missing values
df_with_na = df.copy()
df_with_na.loc[0, 'Score'] = None
df_with_na.fillna(df_with_na['Score'].mean(), inplace=True)
```

**Key Pandas Skills for AI:**
- Data loading and preprocessing
- Handling missing values
- Feature engineering
- Data aggregation and transformation
- Time series analysis

### Matplotlib and Seaborn: Data Visualization

Visualization is crucial for understanding data patterns and model performance.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plotting with Matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig('sine_wave.png')
plt.show()

# Statistical visualization with Seaborn
sns.set(style="whitegrid")

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(np.random.normal(size=1000), kde=True)
plt.title('Normal Distribution')
plt.show()

# Relationship plots
tips = sns.load_dataset("tips")
plt.figure(figsize=(10, 6))
sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)
plt.title('Tips vs Total Bill')
plt.show()
```

**Key Visualization Skills for AI:**
- Creating various plot types (scatter, line, bar, histogram)
- Visualizing high-dimensional data
- Creating subplots for comparison
- Customizing plots for presentations
- Interactive visualization

### Scikit-learn: Machine Learning Toolkit

Scikit-learn provides simple and efficient tools for data mining and data analysis.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample workflow
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Key Scikit-learn Skills for AI:**
- Data preprocessing and feature engineering
- Model selection and evaluation
- Cross-validation techniques
- Hyperparameter tuning
- Pipeline construction

## Advanced Python Concepts for AI

Beyond libraries, certain Python programming concepts are particularly important for AI development:

### List Comprehensions and Generator Expressions

These provide concise ways to create lists and iterables, which is useful for data processing:

```python
# List comprehension
squares = [x**2 for x in range(10)]

# List comprehension with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Generator expression (memory efficient)
sum_of_squares = sum(x**2 for x in range(1000000))
```

### Lambda Functions

Anonymous functions useful for data transformation:

```python
# Sorting a list of tuples by the second element
data = [(1, 5), (3, 2), (2, 8)]
sorted_data = sorted(data, key=lambda x: x[1])

# Applying a function to each element
from functools import reduce
product = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])
```

### Map, Filter, and Reduce

Functional programming tools for data processing:

```python
# Map: Apply function to each item
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# Filter: Keep items that satisfy condition
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

# Reduce: Cumulatively apply function
from functools import reduce
factorial = reduce(lambda x, y: x * y, range(1, 6))  # 5!
```

### Decorators

Useful for adding functionality to functions, commonly used in deep learning frameworks:

```python
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

@timing_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done"

slow_function()  # Will print execution time
```

### Context Managers

Important for resource management, especially with GPU memory:

```python
# Using with statement for file handling
with open('data.txt', 'w') as f:
    f.write('Hello, AI!')

# Custom context manager
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        import time
        self.end = time.time()
        print(f"Execution took {self.end - self.start:.4f} seconds")

# Usage
with Timer():
    # Code to time
    import time
    time.sleep(1)
```

### Error Handling

Robust error handling is crucial for long-running AI training jobs:

```python
try:
    # Potentially problematic code
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
    # Fallback behavior
    result = float('inf')
finally:
    # Cleanup code that always runs
    print("Operation attempted")
```

## Object-Oriented Programming for AI

OOP is essential for organizing complex AI projects:

```python
class Model:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.is_trained = False
        self.parameters = {}
    
    def train(self, data, labels, epochs=100):
        print(f"Training for {epochs} epochs with learning rate {self.learning_rate}")
        # Training logic here
        self.is_trained = True
        return self
    
    def predict(self, data):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        # Prediction logic here
        return "Predictions"
    
    def evaluate(self, data, labels):
        predictions = self.predict(data)
        # Evaluation logic here
        return {"accuracy": 0.95}

# Usage
model = Model(learning_rate=0.001)
model.train(data="training_data", labels="training_labels", epochs=200)
results = model.evaluate(data="test_data", labels="test_labels")
```

## Parallel Processing and Multiprocessing

Efficient data processing often requires parallel execution:

```python
import multiprocessing
import time

def process_chunk(chunk):
    # Simulate processing
    time.sleep(1)
    return sum(chunk)

if __name__ == "__main__":
    # Create data
    data = list(range(1000000))
    
    # Split into chunks
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(data) // num_processes
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    final_result = sum(results)
    print(f"Result: {final_result}")
```

## Working with Data

### Data Loading and Preprocessing

```python
# CSV data
import pandas as pd
df = pd.read_csv('data.csv')

# Image data
from PIL import Image
import numpy as np
img = Image.open('image.jpg')
img_array = np.array(img)

# Text data
with open('text.txt', 'r') as f:
    text = f.read()
    
# Web data
import requests
response = requests.get('https://api.example.com/data')
data = response.json()
```

### Data Augmentation (for Computer Vision)

```python
from PIL import Image, ImageEnhance
import numpy as np

def augment_image(image_path):
    img = Image.open(image_path)
    
    # Rotate
    rotated = img.rotate(45)
    
    # Flip
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    brightened = enhancer.enhance(1.5)
    
    # Convert to array
    img_array = np.array(img)
    rotated_array = np.array(rotated)
    flipped_array = np.array(flipped)
    brightened_array = np.array(brightened)
    
    return [img_array, rotated_array, flipped_array, brightened_array]
```

## Performance Optimization

### Vectorization

Vectorized operations are much faster than loops in Python:

```python
import numpy as np
import time

# Slow way: using loops
def slow_distance(x, y):
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

# Fast way: using vectorization
def fast_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Compare performance
x = np.random.rand(1000000)
y = np.random.rand(1000000)

start = time.time()
slow_result = slow_distance(x, y)
slow_time = time.time() - start

start = time.time()
fast_result = fast_distance(x, y)
fast_time = time.time() - start

print(f"Slow: {slow_time:.4f}s, Fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time / fast_time:.1f}x")
```

### Profiling and Optimization

```python
import cProfile
import pstats

def function_to_profile():
    result = 0
    for i in range(1000000):
        result += i
    return result

# Profile the function
cProfile.run('function_to_profile()', 'profile_stats')

# Analyze results
p = pstats.Stats('profile_stats')
p.strip_dirs().sort_stats('cumulative').print_stats(10)
```

## Practical Exercise: Data Processing Pipeline

Let's create a simple data processing pipeline that demonstrates many of the Python skills needed for AI:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)  # 5 features
noise = np.random.randn(n_samples) * 0.1
y = 2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + noise  # Linear relationship with noise

# 2. Create a DataFrame
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 3. Explore the data
print("Data overview:")
print(df.head())
print("\nStatistical summary:")
print(df.describe())

# 4. Visualize relationships
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    plt.scatter(df[feature], df['target'], alpha=0.5)
    plt.title(f'{feature} vs target')
    plt.xlabel(feature)
    plt.ylabel('target')
plt.tight_layout()
plt.savefig('feature_relationships.png')

# 5. Feature engineering
df['feature_1_squared'] = df['feature_1'] ** 2
df['feature_1_feature_2'] = df['feature_1'] * df['feature_2']

# 6. Data preprocessing
X = df.drop('target', axis=1)
y = df['target']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Convert back to DataFrame for better visualization
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# 10. Show the processed data
print("\nProcessed training data (scaled):")
print(X_train_scaled_df.head())

print("\nPipeline complete! Data is ready for modeling.")
```

Save this as `data_pipeline.py` and run it to see the output and generated visualization.

## Resources for Further Learning

- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) - Comprehensive guide to NumPy, Pandas, Matplotlib, and more
- [Real Python Tutorials](https://realpython.com/) - Excellent Python tutorials with AI/ML focus
- [NumPy Documentation](https://numpy.org/doc/stable/) - Official NumPy documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Official Pandas documentation
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - Examples of various plot types
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html) - Official tutorials for machine learning

In the next module, we'll dive into machine learning fundamentals, building on these Python skills to create and train models.
