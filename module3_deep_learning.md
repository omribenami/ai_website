# Module 3: Deep Learning Essentials

## Overview

Deep Learning has revolutionized artificial intelligence, enabling breakthroughs in computer vision, natural language processing, and many other domains. This module builds on the machine learning fundamentals covered earlier and introduces you to neural networks and deep learning frameworks. With your RTX 3080 GPU, you'll be able to train sophisticated models efficiently.

## Neural Networks Fundamentals

### The Building Blocks: Neurons and Perceptrons

The fundamental unit of a neural network is the artificial neuron, inspired by biological neurons in the brain.

**The Perceptron: The Simplest Neural Network**

A perceptron takes multiple inputs, applies weights, adds a bias, and produces an output through an activation function:

1. **Inputs**: Features or outputs from previous neurons (x₁, x₂, ..., xₙ)
2. **Weights**: Parameters that determine the importance of each input (w₁, w₂, ..., wₙ)
3. **Bias**: An additional parameter that allows the model to fit the data better (b)
4. **Weighted Sum**: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
5. **Activation Function**: Introduces non-linearity, f(z)
6. **Output**: y = f(z)

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def activation(self, z):
        # Step function
        return 1 if z >= 0 else 0
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                # Forward pass
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)
                
                # Update weights and bias
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Example: Training a perceptron for logical OR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
perceptron.train(X, y)

# Visualize decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolors='k')

# Plot decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.title('Perceptron Decision Boundary for OR Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.grid(True)
plt.savefig('perceptron_or_gate.png')
plt.show()

# Test the perceptron
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction}")
```

**Limitations of Single Perceptrons**

Single perceptrons can only learn linearly separable patterns. For example, they cannot learn the XOR function:

```python
# XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=1000)
perceptron.train(X, y)

# Visualize attempt to learn XOR
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolors='k')

# Plot decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.title('Perceptron Fails to Learn XOR Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.grid(True)
plt.savefig('perceptron_xor_failure.png')
plt.show()

# Test the perceptron
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction}")
```

### Multi-layer Neural Networks

To overcome the limitations of single perceptrons, we use multi-layer neural networks with hidden layers:

1. **Input Layer**: Receives the input features
2. **Hidden Layers**: Intermediate layers that transform the input
3. **Output Layer**: Produces the final prediction

```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # Backpropagation
        m = X.shape[0]
        
        # Output layer error
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=10000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Example: Training a neural network for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=10000)

# Plot loss over time
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('nn_xor_loss.png')
plt.show()

# Visualize decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', s=100, edgecolors='k')

# Plot decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.title('Neural Network Decision Boundary for XOR Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.grid(True)
plt.savefig('nn_xor_decision_boundary.png')
plt.show()

# Test the neural network
predictions = nn.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]}")
```

### Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

# Plot activation functions
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, elu(x))
plt.title('ELU')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, swish(x))
plt.title('Swish')
plt.grid(True)

plt.tight_layout()
plt.savefig('activation_functions.png')
plt.show()
```

**Choosing the Right Activation Function**

- **Sigmoid**: Output layer for binary classification (0-1 range)
- **Tanh**: Hidden layers when data is centered around 0 (-1 to 1 range)
- **ReLU**: Default choice for hidden layers in most networks
- **Leaky ReLU**: When dying ReLU is a concern
- **ELU**: Smoother alternative to ReLU
- **Softmax**: Output layer for multi-class classification

### Backpropagation Algorithm

Backpropagation is the key algorithm for training neural networks:

1. **Forward Pass**: Compute predictions
2. **Compute Loss**: Measure the error
3. **Backward Pass**: Compute gradients
4. **Update Parameters**: Adjust weights and biases

The chain rule of calculus is the mathematical foundation of backpropagation, allowing us to compute how each parameter affects the final loss.

### Gradient Descent Optimization

Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize the loss function:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a simple loss function: f(x, y) = x^2 + y^2
def loss_function(x, y):
    return x**2 + y**2

def gradient(x, y):
    return np.array([2*x, 2*y])

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# Plot the loss function surface
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Loss')
ax.set_title('Loss Function Surface')
plt.savefig('loss_surface.png')
plt.show()

# Implement gradient descent
def gradient_descent(start_x, start_y, learning_rate, num_iterations):
    path_x = [start_x]
    path_y = [start_y]
    path_z = [loss_function(start_x, start_y)]
    
    x, y = start_x, start_y
    
    for _ in range(num_iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        
        path_x.append(x)
        path_y.append(y)
        path_z.append(loss_function(x, y))
    
    return np.array(path_x), np.array(path_y), np.array(path_z)

# Run gradient descent with different learning rates
learning_rates = [0.01, 0.1, 0.5]
start_x, start_y = 4.0, 4.0
num_iterations = 20

plt.figure(figsize=(15, 10))

# Contour plot
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.title('Gradient Descent Paths')
plt.xlabel('x')
plt.ylabel('y')

for lr in learning_rates:
    path_x, path_y, _ = gradient_descent(start_x, start_y, lr, num_iterations)
    plt.plot(path_x, path_y, 'o-', label=f'LR = {lr}')

plt.legend()
plt.colorbar(contour)

# Loss vs. iterations
plt.subplot(1, 2, 2)
for lr in learning_rates:
    _, _, path_z = gradient_descent(start_x, start_y, lr, num_iterations)
    plt.plot(range(num_iterations + 1), path_z, 'o-', label=f'LR = {lr}')

plt.title('Loss vs. Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('gradient_descent.png')
plt.show()
```

**Variants of Gradient Descent**

1. **Batch Gradient Descent**: Uses the entire dataset to compute gradients
   - Stable but slow for large datasets

2. **Stochastic Gradient Descent (SGD)**: Uses a single random example
   - Faster but noisy updates

3. **Mini-batch Gradient Descent**: Uses a small random batch of examples
   - Balance between stability and speed

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Implement different gradient descent variants
def batch_gradient_descent(X, y, learning_rate=0.1, num_iterations=100):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]
    losses = []
    
    for iteration in range(num_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        losses.append(loss)
    
    return theta, losses

def stochastic_gradient_descent(X, y, learning_rate=0.1, num_iterations=100):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]
    losses = []
    
    for iteration in range(num_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
        
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        losses.append(loss)
    
    return theta, losses

def mini_batch_gradient_descent(X, y, learning_rate=0.1, num_iterations=100, batch_size=20):
    m = X.shape[0]
    theta = np.random.randn(2, 1)
    X_b = np.c_[np.ones((m, 1)), X]
    n_batches = int(np.ceil(m / batch_size))
    losses = []
    
    for iteration in range(num_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for batch_index in range(n_batches):
            start_idx = batch_index * batch_size
            end_idx = min((batch_index + 1) * batch_size, m)
            X_batch = X_b_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            gradients = 2/len(X_batch) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta = theta - learning_rate * gradients
        
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        losses.append(loss)
    
    return theta, losses

# Run the different variants
theta_batch, losses_batch = batch_gradient_descent(X, y, learning_rate=0.1, num_iterations=50)
theta_sgd, losses_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=50)
theta_mini, losses_mini = mini_batch_gradient_descent(X, y, learning_rate=0.05, num_iterations=50, batch_size=20)

# Plot the results
plt.figure(figsize=(15, 10))

# Plot the data and regression lines
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.7)

# Plot the regression lines
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
plt.plot(X_new, X_new_b.dot(theta_batch), 'r-', linewidth=2, label='Batch GD')
plt.plot(X_new, X_new_b.dot(theta_sgd), 'g-', linewidth=2, label='SGD')
plt.plot(X_new, X_new_b.dot(theta_mini), 'b-', linewidth=2, label='Mini-batch GD')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Different GD Variants')
plt.legend()

# Plot the learning curves
plt.subplot(1, 2, 2)
plt.plot(losses_batch, 'r-', linewidth=2, label='Batch GD')
plt.plot(losses_sgd, 'g-', linewidth=2, label='SGD')
plt.plot(losses_mini, 'b-', linewidth=2, label='Mini-batch GD')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('gd_variants.png')
plt.show()

# Print the final parameters
print("Batch GD parameters:", theta_batch.ravel())
print("SGD parameters:", theta_sgd.ravel())
print("Mini-batch GD parameters:", theta_mini.ravel())
```

### Advanced Optimizers

Modern deep learning uses more sophisticated optimization algorithms:

1. **Momentum**: Adds a fraction of the previous update to the current one
   - Helps overcome local minima and speeds up convergence

2. **RMSprop**: Adapts learning rates for each parameter
   - Divides the learning rate by a running average of recent gradients

3. **Adam**: Combines momentum and RMSprop
   - Currently the most popular optimizer for deep learning

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a challenging loss function with local minima
def loss_function(x, y):
    return x**2 + 5 * np.sin(y)**2

def gradient(x, y):
    return np.array([2*x, 10 * np.sin(y) * np.cos(y)])

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# Plot the loss function surface
plt.figure(figsize=(12, 10))
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour)
plt.title('Loss Function with Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('complex_loss_surface.png')
plt.show()

# Implement different optimizers
def vanilla_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    path_x, path_y = [start_x], [start_y]
    x, y = start_x, start_y
    
    for _ in range(num_iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path_x.append(x)
        path_y.append(y)
    
    return np.array(path_x), np.array(path_y)

def momentum_gradient_descent(start_x, start_y, learning_rate, num_iterations, beta=0.9):
    path_x, path_y = [start_x], [start_y]
    x, y = start_x, start_y
    v_x, v_y = 0, 0
    
    for _ in range(num_iterations):
        grad = gradient(x, y)
        v_x = beta * v_x - learning_rate * grad[0]
        v_y = beta * v_y - learning_rate * grad[1]
        x += v_x
        y += v_y
        path_x.append(x)
        path_y.append(y)
    
    return np.array(path_x), np.array(path_y)

def rmsprop(start_x, start_y, learning_rate, num_iterations, beta=0.9, epsilon=1e-8):
    path_x, path_y = [start_x], [start_y]
    x, y = start_x, start_y
    s_x, s_y = 0, 0
    
    for _ in range(num_iterations):
        grad = gradient(x, y)
        s_x = beta * s_x + (1 - beta) * grad[0]**2
        s_y = beta * s_y + (1 - beta) * grad[1]**2
        x -= learning_rate * grad[0] / (np.sqrt(s_x) + epsilon)
        y -= learning_rate * grad[1] / (np.sqrt(s_y) + epsilon)
        path_x.append(x)
        path_y.append(y)
    
    return np.array(path_x), np.array(path_y)

def adam(start_x, start_y, learning_rate, num_iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    path_x, path_y = [start_x], [start_y]
    x, y = start_x, start_y
    m_x, m_y = 0, 0  # First moment
    v_x, v_y = 0, 0  # Second moment
    
    for t in range(1, num_iterations + 1):
        grad = gradient(x, y)
        
        # Update biased first moment estimate
        m_x = beta1 * m_x + (1 - beta1) * grad[0]
        m_y = beta1 * m_y + (1 - beta1) * grad[1]
        
        # Update biased second raw moment estimate
        v_x = beta2 * v_x + (1 - beta2) * grad[0]**2
        v_y = beta2 * v_y + (1 - beta2) * grad[1]**2
        
        # Bias correction
        m_x_corrected = m_x / (1 - beta1**t)
        m_y_corrected = m_y / (1 - beta1**t)
        v_x_corrected = v_x / (1 - beta2**t)
        v_y_corrected = v_y / (1 - beta2**t)
        
        # Update parameters
        x -= learning_rate * m_x_corrected / (np.sqrt(v_x_corrected) + epsilon)
        y -= learning_rate * m_y_corrected / (np.sqrt(v_y_corrected) + epsilon)
        
        path_x.append(x)
        path_y.append(y)
    
    return np.array(path_x), np.array(path_y)

# Run the different optimizers
start_x, start_y = 4.0, 4.0
num_iterations = 50

path_x_gd, path_y_gd = vanilla_gradient_descent(start_x, start_y, 0.1, num_iterations)
path_x_momentum, path_y_momentum = momentum_gradient_descent(start_x, start_y, 0.01, num_iterations)
path_x_rmsprop, path_y_rmsprop = rmsprop(start_x, start_y, 0.01, num_iterations)
path_x_adam, path_y_adam = adam(start_x, start_y, 0.1, num_iterations)

# Plot the optimization paths
plt.figure(figsize=(12, 10))
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour)

plt.plot(path_x_gd, path_y_gd, 'r-', linewidth=2, label='Gradient Descent')
plt.plot(path_x_momentum, path_y_momentum, 'g-', linewidth=2, label='Momentum')
plt.plot(path_x_rmsprop, path_y_rmsprop, 'b-', linewidth=2, label='RMSprop')
plt.plot(path_x_adam, path_y_adam, 'y-', linewidth=2, label='Adam')

plt.scatter([path_x_gd[-1], path_x_momentum[-1], path_x_rmsprop[-1], path_x_adam[-1]],
            [path_y_gd[-1], path_y_momentum[-1], path_y_rmsprop[-1], path_y_adam[-1]],
            c=['r', 'g', 'b', 'y'], s=100, edgecolors='k')

plt.title('Optimization Paths with Different Optimizers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('optimizer_comparison.png')
plt.show()

# Print final loss values
print("Final loss (GD):", loss_function(path_x_gd[-1], path_y_gd[-1]))
print("Final loss (Momentum):", loss_function(path_x_momentum[-1], path_y_momentum[-1]))
print("Final loss (RMSprop):", loss_function(path_x_rmsprop[-1], path_y_rmsprop[-1]))
print("Final loss (Adam):", loss_function(path_x_adam[-1], path_y_adam[-1]))
```

## Introduction to Deep Learning Frameworks

Modern deep learning relies on powerful frameworks that handle the complex computations efficiently. The two most popular frameworks are PyTorch and TensorFlow.

### PyTorch vs. TensorFlow: Detailed Comparison

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Paradigm** | Dynamic computational graph | Static computational graph (TF 1.x), Dynamic with eager execution (TF 2.x) |
| **Ease of Use** | More Pythonic, intuitive | More structured, comprehensive |
| **Debugging** | Easier due to dynamic nature | Improved in TF 2.x with eager execution |
| **Performance** | Excellent, especially for research | Excellent, especially for production |
| **Deployment** | Improving with TorchScript | Mature with TensorFlow Serving |
| **Community** | Strong in research | Strong in industry |
| **Mobile/Edge** | TorchMobile, growing | TensorFlow Lite, well-established |
| **Visualization** | TensorBoard support via PyTorch-TensorBoard | Native TensorBoard |

### Setting Up PyTorch with GPU Support

PyTorch is particularly popular in research due to its flexibility and Pythonic nature:

```python
# Install PyTorch with CUDA support
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

# Initialize the model
model = SimpleNN(input_size=2, hidden_size=4, output_size=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    losses.append(loss.item())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('pytorch_xor_loss.png')
plt.show()

# Test the model
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')
    
    for i in range(len(X)):
        print(f"Input: {X[i].cpu().numpy()}, Target: {y[i].item()}, Predicted: {predicted[i].item()}")
```

### Setting Up TensorFlow with GPU Support

TensorFlow is widely used in industry due to its production-ready features:

```python
# Install TensorFlow with GPU support
# pip install tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('tensorflow_xor_loss.png')
plt.show()

# Test the model
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(np.int32)
accuracy = np.mean(predicted_classes == y)
print(f'Accuracy: {accuracy:.4f}')

for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Predicted: {predicted_classes[i][0]}")
```

### Basic Operations and Tensor Manipulation

Understanding tensor operations is fundamental to deep learning:

**PyTorch Tensor Operations**

```python
import torch
import numpy as np

# Creating tensors
# From Python list
tensor_a = torch.tensor([1, 2, 3, 4])
print("Tensor from list:", tensor_a)

# From NumPy array
numpy_array = np.array([5, 6, 7, 8])
tensor_b = torch.from_numpy(numpy_array)
print("Tensor from NumPy:", tensor_b)

# With specific data type
tensor_c = torch.tensor([1.2, 3.4, 5.6], dtype=torch.float64)
print("Tensor with float64:", tensor_c)

# Special tensors
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 2)
rand = torch.rand(2, 3)  # Uniform distribution [0, 1)
randn = torch.randn(2, 3)  # Normal distribution (mean=0, std=1)

print("Zeros tensor:\n", zeros)
print("Ones tensor:\n", ones)
print("Random uniform tensor:\n", rand)
print("Random normal tensor:\n", randn)

# Tensor properties
tensor_d = torch.randn(3, 4, 5)
print("Shape:", tensor_d.shape)
print("Data type:", tensor_d.dtype)
print("Device:", tensor_d.device)

# Moving tensors to GPU
if torch.cuda.is_available():
    tensor_gpu = tensor_d.to('cuda')
    print("GPU tensor device:", tensor_gpu.device)

# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise operations
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)

# Matrix operations
m1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
m2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("Matrix multiplication (matmul):\n", torch.matmul(m1, m2))
print("Matrix multiplication (@):\n", m1 @ m2)

# Reshaping tensors
tensor_e = torch.tensor([1, 2, 3, 4, 5, 6])
reshaped = tensor_e.reshape(2, 3)
print("Reshaped tensor:\n", reshaped)

# View (shares memory with original tensor)
viewed = tensor_e.view(3, 2)
print("Viewed tensor:\n", viewed)

# Changing viewed tensor affects original
viewed[0, 0] = 99
print("Original tensor after view modification:", tensor_e)

# Slicing tensors
tensor_f = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("First row:", tensor_f[0])
print("First column:", tensor_f[:, 0])
print("Submatrix:\n", tensor_f[0:2, 1:3])

# Concatenation
cat_dim0 = torch.cat([tensor_f, tensor_f], dim=0)
cat_dim1 = torch.cat([tensor_f, tensor_f], dim=1)
print("Concatenated along dim 0:\n", cat_dim0)
print("Concatenated along dim 1:\n", cat_dim1)

# Stacking
stacked = torch.stack([tensor_f, tensor_f])
print("Stacked tensors shape:", stacked.shape)

# Broadcasting
broadcast_example = tensor_f + torch.tensor([10, 20, 30])
print("Broadcasting example:\n", broadcast_example)
```

**TensorFlow Tensor Operations**

```python
import tensorflow as tf
import numpy as np

# Creating tensors
# From Python list
tensor_a = tf.constant([1, 2, 3, 4])
print("Tensor from list:", tensor_a)

# From NumPy array
numpy_array = np.array([5, 6, 7, 8])
tensor_b = tf.constant(numpy_array)
print("Tensor from NumPy:", tensor_b)

# With specific data type
tensor_c = tf.constant([1.2, 3.4, 5.6], dtype=tf.float64)
print("Tensor with float64:", tensor_c)

# Special tensors
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 2])
rand = tf.random.uniform([2, 3])  # Uniform distribution [0, 1)
randn = tf.random.normal([2, 3])  # Normal distribution (mean=0, std=1)

print("Zeros tensor:\n", zeros)
print("Ones tensor:\n", ones)
print("Random uniform tensor:\n", rand)
print("Random normal tensor:\n", randn)

# Tensor properties
tensor_d = tf.random.normal([3, 4, 5])
print("Shape:", tensor_d.shape)
print("Data type:", tensor_d.dtype)

# Basic operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Element-wise operations
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)

# Matrix operations
m1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
m2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print("Matrix multiplication (matmul):\n", tf.matmul(m1, m2))
print("Matrix multiplication (@):\n", m1 @ m2)

# Reshaping tensors
tensor_e = tf.constant([1, 2, 3, 4, 5, 6])
reshaped = tf.reshape(tensor_e, [2, 3])
print("Reshaped tensor:\n", reshaped)

# Slicing tensors
tensor_f = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("First row:", tensor_f[0])
print("First column:", tensor_f[:, 0])
print("Submatrix:\n", tensor_f[0:2, 1:3])

# Concatenation
cat_dim0 = tf.concat([tensor_f, tensor_f], axis=0)
cat_dim1 = tf.concat([tensor_f, tensor_f], axis=1)
print("Concatenated along axis 0:\n", cat_dim0)
print("Concatenated along axis 1:\n", cat_dim1)

# Stacking
stacked = tf.stack([tensor_f, tensor_f])
print("Stacked tensors shape:", stacked.shape)

# Broadcasting
broadcast_example = tensor_f + tf.constant([10, 20, 30])
print("Broadcasting example:\n", broadcast_example)
```

## Deep Learning Model Training

### Loss Functions

Loss functions measure how well a model performs:

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
true_labels = np.random.randint(0, 3, size=10)  # 3 classes
predicted_probs = np.random.rand(10, 3)
predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)  # Normalize to sum to 1

# Convert to PyTorch tensors
y_true = torch.tensor(true_labels, dtype=torch.long)
y_pred = torch.tensor(predicted_probs, dtype=torch.float32)

# Binary case
binary_true = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
binary_pred = torch.tensor([0.1, 0.7, 0.8, 0.2, 0.6], dtype=torch.float32)

# Regression case
reg_true = torch.tensor([1.2, 2.3, 3.1, 4.7, 5.6], dtype=torch.float32)
reg_pred = torch.tensor([1.0, 2.5, 2.8, 4.9, 5.1], dtype=torch.float32)

# 1. Cross-Entropy Loss (for classification)
ce_loss = F.cross_entropy(y_pred, y_true)
print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

# 2. Binary Cross-Entropy Loss (for binary classification)
bce_loss = F.binary_cross_entropy(binary_pred, binary_true)
print(f"Binary Cross-Entropy Loss: {bce_loss.item():.4f}")

# 3. Mean Squared Error (for regression)
mse_loss = F.mse_loss(reg_pred, reg_true)
print(f"Mean Squared Error: {mse_loss.item():.4f}")

# 4. Mean Absolute Error (for regression)
mae_loss = F.l1_loss(reg_pred, reg_true)
print(f"Mean Absolute Error: {mae_loss.item():.4f}")

# 5. Huber Loss (for regression, robust to outliers)
huber_loss = F.smooth_l1_loss(reg_pred, reg_true)
print(f"Huber Loss: {huber_loss.item():.4f}")

# Visualize different loss functions for regression
plt.figure(figsize=(15, 10))

# Range of predictions for a true value of 0
predictions = np.linspace(-2, 2, 1000)
true_value = np.zeros_like(predictions)

# Calculate losses
mse = (predictions - true_value) ** 2
mae = np.abs(predictions - true_value)
huber = np.where(np.abs(predictions) < 1, 
                 0.5 * (predictions) ** 2, 
                 np.abs(predictions) - 0.5)

# Plot
plt.plot(predictions, mse, 'r-', linewidth=2, label='MSE')
plt.plot(predictions, mae, 'g-', linewidth=2, label='MAE')
plt.plot(predictions, huber, 'b-', linewidth=2, label='Huber')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title('Comparison of Regression Loss Functions')
plt.xlabel('Prediction (True value = 0)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('regression_losses.png')
plt.show()

# Visualize classification loss functions
plt.figure(figsize=(15, 10))

# Range of predicted probabilities for a positive class
p = np.linspace(0.001, 0.999, 1000)

# Calculate losses
bce_0 = -np.log(1 - p)  # BCE for true label = 0
bce_1 = -np.log(p)      # BCE for true label = 1

# Plot
plt.plot(p, bce_0, 'r-', linewidth=2, label='BCE (y=0)')
plt.plot(p, bce_1, 'g-', linewidth=2, label='BCE (y=1)')
plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
plt.title('Binary Cross-Entropy Loss')
plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('classification_losses.png')
plt.show()
```

**Choosing the Right Loss Function**

- **Classification**:
  - Binary: Binary Cross-Entropy
  - Multi-class: Cross-Entropy
  - Imbalanced: Weighted Cross-Entropy, Focal Loss

- **Regression**:
  - General purpose: Mean Squared Error
  - Robust to outliers: Mean Absolute Error, Huber Loss
  - When accuracy is critical: Log-cosh Loss

### Optimizers

Optimizers determine how to update model parameters based on gradients:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate synthetic data with noise
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 1))
y = np.sin(X) + 0.1 * np.random.randn(200, 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Function to train model with different optimizers
def train_with_optimizer(optimizer_name, learning_rate=0.01, epochs=500):
    model = SimpleModel()
    criterion = nn.MSELoss()
    
    # Initialize optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD with Momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3, 3, 100).view(-1, 1)
        y_pred = model(X_test)
    
    return losses, X_test.numpy(), y_pred.numpy()

# Train with different optimizers
optimizers = ['SGD', 'SGD with Momentum', 'RMSprop', 'Adam', 'AdamW']
results = {}

for opt_name in optimizers:
    losses, X_test, y_pred = train_with_optimizer(opt_name)
    results[opt_name] = {
        'losses': losses,
        'X_test': X_test,
        'y_pred': y_pred
    }

# Plot training losses
plt.figure(figsize=(12, 6))
for opt_name in optimizers:
    plt.plot(results[opt_name]['losses'], label=opt_name)
plt.title('Training Loss by Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_losses.png')
plt.show()

# Plot predictions
plt.figure(figsize=(15, 10))
plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(np.linspace(-3, 3, 100), np.sin(np.linspace(-3, 3, 100)), 'k-', label='True function')

for i, opt_name in enumerate(optimizers):
    X_test = results[opt_name]['X_test']
    y_pred = results[opt_name]['y_pred']
    plt.plot(X_test, y_pred, linewidth=2, label=f'{opt_name} prediction')

plt.title('Predictions by Different Optimizers')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_predictions.png')
plt.show()
```

**Choosing the Right Optimizer**

- **SGD**: Simple, works well with large datasets, but slow convergence
- **SGD with Momentum**: Faster convergence than SGD, helps escape local minima
- **RMSprop**: Good for non-stationary objectives, adapts learning rate per parameter
- **Adam**: Generally the best default choice, combines momentum and adaptive learning rates
- **AdamW**: Adam with proper weight decay, often better for large models

### Learning Rate Scheduling

Adjusting the learning rate during training can improve performance:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate synthetic data with noise
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 1))
y = np.sin(X) + 0.1 * np.random.randn(200, 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Function to train model with different schedulers
def train_with_scheduler(scheduler_name, base_lr=0.1, epochs=500):
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    # Initialize scheduler
    if scheduler_name == 'None':
        scheduler = None
    elif scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.995)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    # Training loop
    losses = []
    learning_rates = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        losses.append(loss.item())
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        if scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(loss)
            else:
                scheduler.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3, 3, 100).view(-1, 1)
        y_pred = model(X_test)
    
    return losses, learning_rates, X_test.numpy(), y_pred.numpy()

# Train with different schedulers
schedulers = ['None', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']
results = {}

for sched_name in schedulers:
    losses, learning_rates, X_test, y_pred = train_with_scheduler(sched_name)
    results[sched_name] = {
        'losses': losses,
        'learning_rates': learning_rates,
        'X_test': X_test,
        'y_pred': y_pred
    }

# Plot training losses
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
for sched_name in schedulers:
    plt.plot(results[sched_name]['losses'], label=sched_name)
plt.title('Training Loss by Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for sched_name in schedulers:
    plt.plot(results[sched_name]['learning_rates'], label=sched_name)
plt.title('Learning Rate by Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('scheduler_comparison.png')
plt.show()

# Plot predictions
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(np.linspace(-3, 3, 100), np.sin(np.linspace(-3, 3, 100)), 'k-', label='True function')

for i, sched_name in enumerate(schedulers):
    X_test = results[sched_name]['X_test']
    y_pred = results[sched_name]['y_pred']
    plt.plot(X_test, y_pred, linewidth=2, label=f'{sched_name} prediction')

plt.title('Predictions by Different Schedulers')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('scheduler_predictions.png')
plt.show()
```

**Common Learning Rate Schedules**

1. **Step Decay**: Reduce learning rate by a factor after a fixed number of epochs
2. **Exponential Decay**: Multiply learning rate by a factor each epoch
3. **Cosine Annealing**: Learning rate follows a cosine curve from initial value to near zero
4. **Reduce on Plateau**: Reduce learning rate when a metric stops improving
5. **One Cycle Policy**: Learning rate first increases then decreases, with momentum doing the opposite

### Batch Normalization

Batch normalization helps stabilize and accelerate training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create models with and without batch normalization
class ModelWithoutBN(nn.Module):
    def __init__(self):
        super(ModelWithoutBN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ModelWithBN(nn.Module):
    def __init__(self):
        super(ModelWithBN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Generate synthetic data with noise
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 1))
y = np.sin(X) + 0.1 * np.random.randn(200, 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Function to train model
def train_model(model, learning_rate=0.01, epochs=500):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3, 3, 100).view(-1, 1)
        y_pred = model(X_test)
    
    return losses, X_test.numpy(), y_pred.numpy()

# Train models
model_without_bn = ModelWithoutBN()
model_with_bn = ModelWithBN()

print("Training model without batch normalization...")
losses_without_bn, X_test, y_pred_without_bn = train_model(model_without_bn)

print("\nTraining model with batch normalization...")
losses_with_bn, X_test, y_pred_with_bn = train_model(model_with_bn)

# Plot training losses
plt.figure(figsize=(12, 6))
plt.plot(losses_without_bn, label='Without BatchNorm')
plt.plot(losses_with_bn, label='With BatchNorm')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('batchnorm_comparison.png')
plt.show()

# Plot predictions
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(np.linspace(-3, 3, 100), np.sin(np.linspace(-3, 3, 100)), 'k-', label='True function')
plt.plot(X_test, y_pred_without_bn, 'r-', linewidth=2, label='Without BatchNorm')
plt.plot(X_test, y_pred_with_bn, 'g-', linewidth=2, label='With BatchNorm')
plt.title('Predictions Comparison')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('batchnorm_predictions.png')
plt.show()
```

**Benefits of Batch Normalization**

1. **Faster Convergence**: Allows higher learning rates
2. **Reduces Internal Covariate Shift**: Stabilizes the distribution of layer inputs
3. **Regularization Effect**: Adds noise during training, reducing overfitting
4. **Reduces Sensitivity to Initialization**: Makes networks more robust to poor initialization

### Regularization Techniques

Regularization helps prevent overfitting:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create models with different regularization techniques
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class L1RegModel(nn.Module):
    def __init__(self):
        super(L1RegModel, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss

class L2RegModel(nn.Module):
    def __init__(self):
        super(L2RegModel, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class DropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Generate synthetic data with noise
np.random.seed(42)
X_train = np.random.uniform(-3, 3, size=(100, 1))
y_train = np.sin(X_train) + 0.1 * np.random.randn(100, 1)

X_test = np.random.uniform(-3.5, 3.5, size=(50, 1))  # Slightly wider range to test generalization
y_test = np.sin(X_test) + 0.1 * np.random.randn(50, 1)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Function to train model
def train_model(model_name, epochs=1000):
    if model_name == 'Base':
        model = BaseModel()
        l1_lambda = 0
    elif model_name == 'L1':
        model = L1RegModel()
        l1_lambda = 0.001
    elif model_name == 'L2':
        model = L2RegModel()
        l1_lambda = 0
    elif model_name == 'Dropout':
        model = DropoutModel()
        l1_lambda = 0
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    criterion = nn.MSELoss()
    
    if model_name == 'L2':
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)  # L2 regularization
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Add L1 regularization if applicable
        if model_name == 'L1':
            l1_loss = model.l1_loss()
            loss += l1_lambda * l1_loss
        
        train_losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # Generate predictions for visualization
    model.eval()
    with torch.no_grad():
        X_viz = torch.linspace(-4, 4, 200).view(-1, 1)
        y_pred = model(X_viz)
    
    return train_losses, test_losses, X_viz.numpy(), y_pred.numpy()

# Train models with different regularization techniques
models = ['Base', 'L1', 'L2', 'Dropout']
results = {}

for model_name in models:
    print(f"\nTraining {model_name} model...")
    train_losses, test_losses, X_viz, y_pred = train_model(model_name)
    results[model_name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'X_viz': X_viz,
        'y_pred': y_pred
    }

# Plot training and test losses
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
for model_name in models:
    plt.plot(results[model_name]['train_losses'], label=f'{model_name} Train')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for model_name in models:
    plt.plot(results[model_name]['test_losses'], label=f'{model_name} Test')
plt.title('Test Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('regularization_losses.png')
plt.show()

# Plot predictions
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.3, label='Train Data')
plt.scatter(X_test, y_test, alpha=0.3, label='Test Data')
plt.plot(np.linspace(-4, 4, 200), np.sin(np.linspace(-4, 4, 200)), 'k-', label='True function')

for model_name in models:
    X_viz = results[model_name]['X_viz']
    y_pred = results[model_name]['y_pred']
    plt.plot(X_viz, y_pred, linewidth=2, label=f'{model_name} prediction')

plt.title('Predictions with Different Regularization Techniques')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('regularization_predictions.png')
plt.show()
```

**Common Regularization Techniques**

1. **L1 Regularization (Lasso)**: Adds the sum of absolute values of weights to the loss
   - Encourages sparse weights (many zeros)
   - Good for feature selection

2. **L2 Regularization (Ridge)**: Adds the sum of squared weights to the loss
   - Prevents any weight from becoming too large
   - Most common form of regularization

3. **Dropout**: Randomly sets a fraction of inputs to zero during training
   - Forces the network to learn redundant representations
   - Very effective for deep networks

4. **Early Stopping**: Stop training when validation performance starts to degrade
   - Simple and effective
   - Requires a validation set

5. **Data Augmentation**: Create new training examples by transforming existing ones
   - Very effective for image data
   - Domain-specific (different for images, text, etc.)

### Training Monitoring and Visualization

Monitoring training progress is essential for debugging and optimization:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model
class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize the model
model = ClassificationModel(input_size=20, hidden_size=50, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training monitoring class
class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.learning_rates = []
        self.gradients = []
        self.weight_norms = []
    
    def update(self, train_loss, train_acc, test_loss, test_acc, model, optimizer):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Calculate gradient norm
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.gradients.append(total_grad_norm)
        
        # Calculate weight norm
        total_weight_norm = 0
        for param in model.parameters():
            total_weight_norm += param.norm(2).item() ** 2
        total_weight_norm = total_weight_norm ** 0.5
        self.weight_norms.append(total_weight_norm)
    
    def plot(self, save_path=None):
        plt.figure(figsize=(15, 15))
        
        # Plot losses
        plt.subplot(3, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(3, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.test_accs, label='Test Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(3, 2, 3)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Plot gradient norm
        plt.subplot(3, 2, 4)
        plt.plot(self.gradients)
        plt.title('Gradient Norm vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        # Plot weight norm
        plt.subplot(3, 2, 5)
        plt.plot(self.weight_norms)
        plt.title('Weight Norm vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Norm')
        plt.grid(True)
        
        # Plot train vs test accuracy
        plt.subplot(3, 2, 6)
        plt.scatter(self.train_accs, self.test_accs, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Train vs. Test Accuracy')
        plt.xlabel('Train Accuracy')
        plt.ylabel('Test Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Initialize the training monitor
monitor = TrainingMonitor()

# Training and evaluation function
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    # Update monitor
    monitor.update(train_loss, train_acc, test_loss, test_acc, model, optimizer)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Plot training metrics
monitor.plot(save_path='training_monitoring.png')
```

**Key Metrics to Monitor**

1. **Training and Validation Loss**: To detect overfitting or underfitting
2. **Training and Validation Accuracy**: To measure model performance
3. **Learning Rate**: To verify scheduler behavior
4. **Gradient Norms**: To detect vanishing or exploding gradients
5. **Weight Norms**: To monitor model complexity
6. **Layer Activations**: To ensure proper information flow

## Practical Exercise: Building a Neural Network from Scratch

Let's apply what we've learned to build a neural network from scratch for a real-world dataset:

```python
# Save this as deep_learning_exercise.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the neural network
class BreastCancerNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3):
        super(BreastCancerNN, self).__init__()
        
        # Create layers dynamically based on hidden_sizes list
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Initialize the model
input_size = X_train.shape[1]  # Number of features
hidden_sizes = [64, 32, 16]
model = BreastCancerNN(input_size, hidden_sizes)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training monitoring
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# Training function
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch()
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    test_loss, test_acc = evaluate()
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Update learning rate
    scheduler.step(test_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(test_accs)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(test_losses)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('breast_cancer_training.png')
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test_tensor).float().mean()
    print(f"Final Test Accuracy: {accuracy.item():.4f}")

# Feature importance analysis
def compute_feature_importance():
    feature_importance = np.zeros(input_size)
    
    for i in range(input_size):
        # Create a modified test set with one feature shuffled
        X_test_modified = X_test.copy()
        np.random.shuffle(X_test_modified[:, i])
        X_test_modified_tensor = torch.tensor(X_test_modified, dtype=torch.float32)
        
        # Evaluate on modified data
        with torch.no_grad():
            y_pred_modified = model(X_test_modified_tensor)
            y_pred_modified_class = (y_pred_modified > 0.5).float()
            modified_accuracy = (y_pred_modified_class == y_test_tensor).float().mean()
        
        # Feature importance is the drop in accuracy
        feature_importance[i] = accuracy.item() - modified_accuracy.item()
    
    return feature_importance

# Compute and plot feature importance
feature_importance = compute_feature_importance()
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(12, 8))
plt.bar(range(input_size), feature_importance[sorted_idx])
plt.xticks(range(input_size), [data.feature_names[i] for i in sorted_idx], rotation=90)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance (Accuracy Drop)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("Top 5 most important features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"{i+1}. {data.feature_names[idx]}: {feature_importance[idx]:.4f}")
```

## Resources for Further Learning

- [Deep Learning Book](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Fast.ai Courses](https://www.fast.ai/)
- [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)

In the next module, we'll explore Computer Vision with deep learning, building on these neural network fundamentals to create models that can understand and interpret images.
