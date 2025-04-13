# Practical Experiments and Projects

This section contains hands-on projects and experiments designed to reinforce the concepts covered in the theoretical modules. Each project is structured to build practical skills while applying AI concepts in real-world scenarios.

## Project 1: Image Classification with Convolutional Neural Networks

### Overview
In this project, you'll build an image classification system using convolutional neural networks (CNNs). You'll train a model to recognize different objects in images, leveraging your RTX 3080 GPU for accelerated training.

### Learning Objectives
- Implement CNNs for image classification
- Apply data augmentation techniques
- Use transfer learning with pre-trained models
- Optimize model performance
- Visualize and interpret CNN features

### Requirements
- Python 3.8+
- PyTorch or TensorFlow
- CUDA drivers for GPU acceleration
- Jupyter Notebook

### Dataset
We'll use the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Implementation Steps

#### Step 1: Setup and Data Preparation

```python
# image_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('cifar10_samples.png')
    plt.show()

# Get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images[:8]))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
```

#### Step 2: Define the CNN Architecture

```python
# Define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create the model and move it to GPU
model = SimpleCNN().to(device)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
```

#### Step 3: Train the Model

```python
# Training function
def train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    train_losses = []
    train_accs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}, '
                      f'accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0
        
        # Calculate epoch accuracy
        epoch_acc = 100 * correct / total
        train_accs.append(epoch_acc)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, testloader, criterion)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_end = time.time()
        print(f'Epoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds')
        print(f'Train Accuracy: {epoch_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
    end_time = time.time()
    print(f'Finished Training in {end_time - start_time:.2f} seconds')
    return train_losses, train_accs

# Evaluation function
def evaluate_model(model, testloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(testloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# Train the model
train_losses, train_accs = train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=20)

# Save the model
torch.save(model.state_dict(), 'cifar10_cnn.pth')
```

#### Step 4: Evaluate and Visualize Results

```python
# Load the saved model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_cnn.pth'))
model.eval()

# Evaluate on test set
test_loss, test_acc = evaluate_model(model, testloader, criterion)
print(f'Test Accuracy: {test_acc:.2f}%')

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Visualize some predictions
def visualize_predictions(model, testloader, classes, num_images=8):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 6))
    for i in range(num_images):
        ax = fig.add_subplot(2, num_images//2, i+1, xticks=[], yticks=[])
        imshow_single(images[i].cpu())
        title_color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f'{classes[predicted[i]]}', color=title_color)
    plt.savefig('prediction_examples.png')
    plt.show()

def imshow_single(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

visualize_predictions(model, testloader, classes)
```

#### Step 5: Implement Transfer Learning

```python
# Transfer learning with ResNet
import torchvision.models as models

# Define a model using transfer learning
class TransferModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Create the transfer learning model
transfer_model = TransferModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transfer_model.resnet.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Train the transfer learning model
transfer_train_losses, transfer_train_accs = train_model(
    transfer_model, trainloader, criterion, optimizer, scheduler, num_epochs=10)

# Save the model
torch.save(transfer_model.state_dict(), 'cifar10_transfer.pth')

# Evaluate the transfer learning model
transfer_test_loss, transfer_test_acc = evaluate_model(transfer_model, testloader, criterion)
print(f'Transfer Learning Test Accuracy: {transfer_test_acc:.2f}%')

# Compare the two models
print(f'Simple CNN Test Accuracy: {test_acc:.2f}%')
print(f'Transfer Learning Test Accuracy: {transfer_test_acc:.2f}%')
```

#### Step 6: Feature Visualization

```python
# Visualize activations
def visualize_layer_outputs(model, image, layer_name):
    # Register a hook to get the output of a specific layer
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register the hook
    if layer_name == 'conv1':
        handle = model.conv1.register_forward_hook(get_activation(layer_name))
    elif layer_name == 'conv2':
        handle = model.conv2.register_forward_hook(get_activation(layer_name))
    elif layer_name == 'conv3':
        handle = model.conv3.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    
    # Remove the hook
    handle.remove()
    
    # Get the activations
    act = activations[layer_name].squeeze().cpu()
    
    # Plot the activations
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Activations for layer: {layer_name}', fontsize=16)
    
    num_filters = min(act.size(0), 32)  # Show at most 32 filters
    for i in range(num_filters):
        ax = fig.add_subplot(4, 8, i+1)
        ax.imshow(act[i], cmap='viridis')
        ax.axis('off')
    
    plt.savefig(f'{layer_name}_activations.png')
    plt.show()

# Get a sample image
dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]

# Visualize activations for different layers
visualize_layer_outputs(model, img, 'conv1')
visualize_layer_outputs(model, img, 'conv2')
visualize_layer_outputs(model, img, 'conv3')
```

### Challenges and Extensions

1. **Improve Model Performance**:
   - Experiment with different architectures
   - Try different optimizers and learning rates
   - Implement more advanced data augmentation

2. **Model Interpretability**:
   - Implement Grad-CAM to visualize which parts of the image are important for classification
   - Analyze misclassified images to understand model weaknesses

3. **Deploy the Model**:
   - Create a simple web application to classify uploaded images
   - Optimize the model for inference speed

## Project 2: Natural Language Processing with Transformers

### Overview
In this project, you'll build a text classification system using transformer models. You'll fine-tune a pre-trained model to classify text into different categories, leveraging your RTX 3080 GPU for efficient training.

### Learning Objectives
- Understand transformer architecture
- Apply transfer learning with pre-trained language models
- Process and tokenize text data
- Fine-tune models for specific NLP tasks
- Evaluate and interpret NLP model results

### Requirements
- Python 3.8+
- PyTorch
- Transformers library (Hugging Face)
- CUDA drivers for GPU acceleration
- Jupyter Notebook

### Dataset
We'll use the AG News dataset, which consists of news articles categorized into 4 classes:
1. World
2. Sports
3. Business
4. Sci/Tech

### Implementation Steps

#### Step 1: Setup and Data Preparation

```python
# text_classification.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import time
import random

# Set random seeds for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load AG News dataset
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer

# Download dataset
train_iter, test_iter = AG_NEWS(root='./data', split=('train', 'test'))

# Convert to pandas DataFrame
train_data = []
for label, text in train_iter:
    train_data.append({'label': label - 1, 'text': text})  # Labels are 1-indexed, convert to 0-indexed

test_data = []
for label, text in test_iter:
    test_data.append({'label': label - 1, 'text': text})

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Display class distribution
plt.figure(figsize=(10, 6))
train_df['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(4), ['World', 'Sports', 'Business', 'Sci/Tech'])
plt.savefig('class_distribution.png')
plt.show()

# Sample some examples
print("\nSample examples:")
for i in range(4):
    print(f"Class: {train_df.iloc[i]['label']} - Text: {train_df.iloc[i]['text'][:100]}...")
```

#### Step 2: Create Dataset and DataLoader

```python
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = NewsDataset(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer
)

test_dataset = NewsDataset(
    texts=test_df['text'].values,
    labels=test_df['label'].values,
    tokenizer=tokenizer
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

# Check a batch
sample_batch = next(iter(train_loader))
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Attention mask shape: {sample_batch['attention_mask'].shape}")
print(f"Labels shape: {sample_batch['label'].shape}")
```

#### Step 3: Fine-tune BERT Model

```python
# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False
)

# Move model to GPU
model = model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Total number of training steps
total_steps = len(train_loader) * 3  # 3 epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader), correct_predictions.double() / total_predictions

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    return (
        total_loss / len(data_loader),
        correct_predictions.double() / total_predictions,
        all_preds,
        all_labels
    )

# Train the model
epochs = 3
best_accuracy = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    
    start_time = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        device
    )
    
    print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')
    
    # Evaluate
    val_loss, val_acc, _, _ = evaluate(model, test_loader, device)
    
    print(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')
    
    # Save the best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_bert_model.pt')
        print('Saved best model!')
    
    end_time = time.time()
    print(f'Epoch completed in {end_time - start_time:.2f} seconds')
    print()

print(f'Best validation accuracy: {best_accuracy:.4f}')
```

#### Step 4: Evaluate and Analyze Results

```python
# Load the best model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False
)
model.load_state_dict(torch.load('best_bert_model.pt'))
model = model.to(device)

# Evaluate on test set
test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, device)
print(f'Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}')

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['World', 'Sports', 'Business', 'Sci/Tech'],
            yticklabels=['World', 'Sports', 'Business', 'Sci/Tech'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('bert_confusion_matrix.png')
plt.show()

# Classification report
print(classification_report(all_labels, all_preds, 
                           target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# Analyze some predictions
def analyze_predictions(model, dataset, tokenizer, device, num_examples=5):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1)
    
    examples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_examples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].item()
            text = batch['text'][0]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred = torch.max(outputs.logits, dim=1)
            pred = pred.item()
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            
            examples.append({
                'text': text,
                'true_label': class_names[label],
                'predicted_label': class_names[pred],
                'correct': label == pred,
                'probabilities': {class_names[i]: probs[0][i].item() for i in range(4)}
            })
    
    return examples

# Analyze some examples
examples = analyze_predictions(model, test_dataset, tokenizer, device, num_examples=5)

for i, example in enumerate(examples):
    print(f"Example {i+1}:")
    print(f"Text: {example['text'][:100]}...")
    print(f"True label: {example['true_label']}")
    print(f"Predicted label: {example['predicted_label']}")
    print(f"Correct: {example['correct']}")
    print("Probabilities:")
    for label, prob in example['probabilities'].items():
        print(f"  {label}: {prob:.4f}")
    print()
```

#### Step 5: Text Generation with the Model

```python
# Function to predict class for new text
def predict_class(text, model, tokenizer, device):
    # Tokenize the text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Convert to class name
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    predicted_class = class_names[preds.item()]
    
    # Get probabilities for each class
    probabilities = {class_names[i]: probs[0][i].item() for i in range(4)}
    
    return predicted_class, probabilities

# Test with some new examples
new_texts = [
    "The stock market reached a new high today as investors reacted to positive economic data.",
    "The team won the championship after a thrilling overtime victory in the final game.",
    "Scientists discovered a new species of deep-sea creatures near the Mariana Trench.",
    "World leaders gathered for a summit to discuss climate change and global warming solutions."
]

for text in new_texts:
    predicted_class, probabilities = predict_class(text, model, tokenizer, device)
    print(f"Text: {text}")
    print(f"Predicted class: {predicted_class}")
    print("Probabilities:")
    for label, prob in probabilities.items():
        print(f"  {label}: {prob:.4f}")
    print()
```

### Challenges and Extensions

1. **Try Different Pre-trained Models**:
   - Compare BERT with RoBERTa, DistilBERT, or other transformer models
   - Experiment with different model sizes (base vs. large)

2. **Multi-label Classification**:
   - Modify the model to handle texts that belong to multiple categories

3. **Text Summarization**:
   - Extend the project to generate summaries of news articles

4. **Sentiment Analysis**:
   - Adapt the model to analyze sentiment in text

## Project 3: Reinforcement Learning for Game Playing

### Overview
In this project, you'll implement reinforcement learning algorithms to train an agent to play a simple game. You'll use your RTX 3080 GPU to accelerate the training process and visualize the agent's learning progress.

### Learning Objectives
- Understand reinforcement learning concepts
- Implement Q-learning and Deep Q-Network (DQN) algorithms
- Design reward systems for effective learning
- Visualize and analyze agent behavior
- Apply GPU acceleration to reinforcement learning

### Requirements
- Python 3.8+
- PyTorch
- Gymnasium (formerly OpenAI Gym)
- Matplotlib
- CUDA drivers for GPU acceleration

### Implementation Steps

#### Step 1: Setup and Environment Creation

```python
# reinforcement_learning.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('CartPole-v1')
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# Set random seeds for reproducibility
seed = 42
env.reset(seed=seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Test the environment
observation, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

#### Step 2: Implement Q-Learning (Tabular Method)

```python
# Create a simpler environment for tabular Q-learning
env = gym.make('FrozenLake-v1', is_slippery=False)
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Q-learning parameters
alpha = 0.8  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 10000

# Training the agent
rewards = []
epsilons = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit
        
        # Take action and observe new state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Track progress
    rewards.append(total_reward)
    epsilons.append(epsilon)
    
    if episode % 1000 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Plot training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(epsilons)
plt.title('Epsilon per Episode')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.savefig('q_learning_training.png')
plt.show()

# Print the Q-table
print("Q-table:")
print(q_table)

# Test the trained agent
def test_agent(env, q_table, num_episodes=10):
    total_rewards = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.argmax(q_table[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    print(f"Average Reward: {total_rewards / num_episodes}")

# Test the agent
test_agent(env, q_table)
```

#### Step 3: Implement Deep Q-Network (DQN)

```python
# Create CartPole environment for DQN
env = gym.make('CartPole-v1')

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        
        # Update frequency
        self.update_target_every = 10
        self.target_update_counter = 0
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return torch.argmax(action_values).item()
    
    def learn(self):
        # If buffer is not large enough, skip learning
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Get Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

# Training the DQN agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 500
max_steps = 500
batch_size = 64

# Training metrics
rewards = []
epsilons = []
losses = []
steps_per_episode = []

start_time = time.time()

for episode in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])[0]
    total_reward = 0
    episode_losses = []
    
    for step in range(max_steps):
        # Select action
        action = agent.act(state)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])[0]
        
        # Store experience
        agent.buffer.add(state, action, reward, next_state, done)
        
        # Learn
        if len(agent.buffer) >= batch_size:
            loss = agent.learn()
            if loss:
                episode_losses.append(loss)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Track metrics
    rewards.append(total_reward)
    epsilons.append(agent.epsilon)
    if episode_losses:
        losses.append(np.mean(episode_losses))
    else:
        losses.append(0)
    steps_per_episode.append(step + 1)
    
    # Print progress
    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Steps: {step + 1}, Epsilon: {agent.epsilon:.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save the trained model
torch.save(agent.policy_net.state_dict(), 'dqn_cartpole.pth')

# Plot training metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2, 2, 2)
plt.plot(epsilons)
plt.title('Epsilon per Episode')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.subplot(2, 2, 3)
plt.plot(losses)
plt.title('Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.savefig('dqn_training.png')
plt.show()

# Test the trained agent
def test_dqn_agent(env, agent, num_episodes=10, render=False):
    total_rewards = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])[0]
        done = False
        episode_reward = 0
        
        while not done:
            # Use policy network with no exploration
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(agent.policy_net(state_tensor)).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
        
        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    print(f"Average Reward: {total_rewards / num_episodes}")

# Test the agent
test_dqn_agent(env, agent)
```

#### Step 4: Implement Policy Gradient Method (REINFORCE)

```python
# Implement REINFORCE algorithm
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, action_size).to(device)
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Episode history
        self.states = []
        self.actions = []
        self.rewards = []
    
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy_net(state).cpu().detach().numpy()[0]
        action = np.random.choice(self.action_size, p=probs)
        return action
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def discount_rewards(self):
        discounted_rewards = []
        cumulative_reward = 0
        
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        return discounted_rewards
    
    def learn(self):
        # Convert episode history to tensors
        states = torch.tensor(np.vstack(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        discounted_rewards = self.discount_rewards()
        
        # Calculate loss
        probs = self.policy_net(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs)
        loss = -torch.sum(log_probs * discounted_rewards)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode history
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss.item()

# Training the REINFORCE agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = REINFORCEAgent(state_size, action_size)

episodes = 1000
max_steps = 500

# Training metrics
episode_rewards = []
episode_lengths = []
losses = []

start_time = time.time()

for episode in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])[0]
    total_reward = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.act(state)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])[0]
        
        # Remember experience
        agent.remember(state, action, reward)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Learn from episode
    loss = agent.learn()
    
    # Track metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    losses.append(loss)
    
    # Print progress
    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Steps: {step + 1}")
    
    # Early stopping if solved
    if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195:
        print(f"Environment solved in {episode} episodes!")
        break

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save the trained model
torch.save(agent.policy_net.state_dict(), 'reinforce_cartpole.pth')

# Plot training metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(episode_rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2, 2, 2)
plt.plot(episode_lengths)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.subplot(2, 2, 3)
plt.plot(losses)
plt.title('Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')

# Plot moving average of rewards
window_size = 100
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(2, 2, 4)
    plt.plot(moving_avg)
    plt.title(f'Moving Average of Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

plt.tight_layout()
plt.savefig('reinforce_training.png')
plt.show()

# Test the trained agent
def test_reinforce_agent(env, agent, num_episodes=10, render=False):
    total_rewards = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])[0]
        done = False
        episode_reward = 0
        
        while not done:
            # Use policy network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs = agent.policy_net(state_tensor).cpu().detach().numpy()[0]
            action = np.argmax(probs)  # Greedy action for testing
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
        
        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    print(f"Average Reward: {total_rewards / num_episodes}")

# Test the agent
test_reinforce_agent(env, agent)
```

#### Step 5: Compare Different RL Algorithms

```python
# Load the trained models
dqn_model = DQN(state_size, action_size).to(device)
dqn_model.load_state_dict(torch.load('dqn_cartpole.pth'))
dqn_model.eval()

reinforce_model = PolicyNetwork(state_size, action_size).to(device)
reinforce_model.load_state_dict(torch.load('reinforce_cartpole.pth'))
reinforce_model.eval()

# Function to evaluate a model
def evaluate_model(model, model_type, num_episodes=100):
    rewards = []
    steps = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])[0]
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            if model_type == 'dqn':
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()
            elif model_type == 'reinforce':
                with torch.no_grad():
                    probs = model(state_tensor).cpu().numpy()[0]
                    action = np.argmax(probs)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if episode_steps >= 500:  # Max steps in CartPole-v1
                break
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
    
    return np.mean(rewards), np.std(rewards), np.mean(steps), np.std(steps)

# Evaluate both models
dqn_mean_reward, dqn_std_reward, dqn_mean_steps, dqn_std_steps = evaluate_model(dqn_model, 'dqn')
reinforce_mean_reward, reinforce_std_reward, reinforce_mean_steps, reinforce_std_steps = evaluate_model(reinforce_model, 'reinforce')

print("DQN Performance:")
print(f"Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")
print(f"Mean Steps: {dqn_mean_steps:.2f} ± {dqn_std_steps:.2f}")

print("\nREINFORCE Performance:")
print(f"Mean Reward: {reinforce_mean_reward:.2f} ± {reinforce_std_reward:.2f}")
print(f"Mean Steps: {reinforce_mean_steps:.2f} ± {reinforce_std_steps:.2f}")

# Plot comparison
plt.figure(figsize=(12, 6))

# Bar chart for mean rewards
plt.subplot(1, 2, 1)
algorithms = ['DQN', 'REINFORCE']
mean_rewards = [dqn_mean_reward, reinforce_mean_reward]
std_rewards = [dqn_std_reward, reinforce_std_reward]

plt.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=10)
plt.title('Mean Reward by Algorithm')
plt.ylabel('Mean Reward')
plt.ylim([0, 500])

# Bar chart for mean steps
plt.subplot(1, 2, 2)
mean_steps = [dqn_mean_steps, reinforce_mean_steps]
std_steps = [dqn_std_steps, reinforce_std_steps]

plt.bar(algorithms, mean_steps, yerr=std_steps, capsize=10)
plt.title('Mean Steps by Algorithm')
plt.ylabel('Mean Steps')
plt.ylim([0, 500])

plt.tight_layout()
plt.savefig('algorithm_comparison.png')
plt.show()
```

### Challenges and Extensions

1. **Try Different Environments**:
   - Implement the algorithms for more complex environments like LunarLander or Atari games

2. **Implement Advanced Algorithms**:
   - Add Actor-Critic methods
   - Implement Proximal Policy Optimization (PPO)
   - Try Soft Actor-Critic (SAC) for continuous action spaces

3. **Hyperparameter Tuning**:
   - Systematically explore different hyperparameters
   - Implement a grid search or random search

4. **Multi-Agent Reinforcement Learning**:
   - Extend to environments with multiple agents
   - Implement cooperative or competitive scenarios

## Project 4: Generative AI with GANs

### Overview
In this project, you'll implement a Generative Adversarial Network (GAN) to generate realistic images. You'll leverage your RTX 3080 GPU to train the model efficiently and visualize the generated images.

### Learning Objectives
- Understand GAN architecture and training dynamics
- Implement generator and discriminator networks
- Train GANs effectively
- Visualize and evaluate generated images
- Apply GPU acceleration to GAN training

### Requirements
- Python 3.8+
- PyTorch
- Matplotlib
- CUDA drivers for GPU acceleration

### Implementation Steps

#### Step 1: Setup and Data Preparation

```python
# gan.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Function to display images
def show_images(images, num_images=25, size=(5, 5)):
    plt.figure(figsize=(10, 10))
    
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(size[0], size[1], i+1)
        plt.imshow(np.transpose((images[i] * 0.5 + 0.5).cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('real_images.png')
    plt.show()

# Display some real images
real_batch = next(iter(dataloader))
show_images(real_batch[0])
```

#### Step 2: Define GAN Architecture

```python
# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is latent vector z
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: 3 x 64 x 64
        )
    
    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is 3 x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # State size: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Initialize networks
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Print model summaries
print(generator)
print(discriminator)
```

#### Step 3: Train the GAN

```python
# Loss function and optimizers
criterion = nn.BCELoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training parameters
num_epochs = 50
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Lists to track progress
img_list = []
g_losses = []
d_losses = []
d_real_accuracies = []
d_fake_accuracies = []

# Training loop
print("Starting Training...")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with real batch
        discriminator.zero_grad()
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), 1.0, device=device)
        
        output = discriminator(real_images)
        d_real_accuracy = (output > 0.5).float().mean().item()
        errD_real = criterion(output, label)
        errD_real.backward()
        
        # Train with fake batch
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(0.0)
        
        output = discriminator(fake_images.detach())
        d_fake_accuracy = (output < 0.5).float().mean().item()
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        # Update D
        errD = errD_real + errD_fake
        optimizer_d.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(1.0)  # Fake labels are real for generator cost
        
        output = discriminator(fake_images)
        errG = criterion(output, label)
        errG.backward()
        
        # Update G
        optimizer_g.step()
        
        # Save losses for plotting
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D_real: {d_real_accuracy:.4f} D_fake: {d_fake_accuracy:.4f}')
            
            g_losses.append(errG.item())
            d_losses.append(errD.item())
            d_real_accuracies.append(d_real_accuracy)
            d_fake_accuracies.append(d_fake_accuracy)
    
    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    img_list.append(torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
    
    # Save images at the end of each epoch
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.axis('off')
    plt.title(f"Epoch {epoch}")
    plt.savefig(f'gan_epoch_{epoch}.png')
    plt.close()
    
    epoch_end_time = time.time()
    print(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
```

#### Step 4: Visualize Results and Generate Images

```python
# Plot training losses
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(g_losses, label='Generator')
plt.plot(d_losses, label='Discriminator')
plt.title('GAN Losses')
plt.xlabel('Iterations (x50)')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(d_real_accuracies, label='Real')
plt.plot(d_fake_accuracies, label='Fake')
plt.title('Discriminator Accuracy')
plt.xlabel('Iterations (x50)')
plt.ylabel('Accuracy')
plt.legend()

# Show progression of generated images
plt.subplot(2, 2, 3)
plt.imshow(np.transpose(img_list[0], (1, 2, 0)))
plt.axis('off')
plt.title('Generated Images (Start)')

plt.subplot(2, 2, 4)
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.axis('off')
plt.title('Generated Images (End)')

plt.tight_layout()
plt.savefig('gan_training_results.png')
plt.show()

# Create animation of training progression
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# Save animation
ani.save('gan_training.gif', writer='pillow', fps=5)

# Generate new images
num_images = 16
noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
with torch.no_grad():
    generated_images = generator(noise).detach().cpu()

# Display generated images
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(4, 4, i+1)
    plt.imshow(np.transpose((generated_images[i] * 0.5 + 0.5).numpy(), (1, 2, 0)))
    plt.axis('off')
plt.tight_layout()
plt.savefig('new_generated_images.png')
plt.show()

# Interpolation between two random points in latent space
def interpolate_latent_space(generator, num_steps=10):
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    
    # Generate images for interpolated points
    images = []
    alphas = np.linspace(0, 1, num_steps)
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = z1 * (1 - alpha) + z2 * alpha
            image = generator(z_interp).detach().cpu()
            images.append(image[0])
    
    # Display interpolated images
    plt.figure(figsize=(20, 4))
    for i, image in enumerate(images):
        plt.subplot(1, num_steps, i+1)
        plt.imshow(np.transpose((image * 0.5 + 0.5).numpy(), (1, 2, 0)))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('latent_space_interpolation.png')
    plt.show()

# Interpolate between two points in latent space
interpolate_latent_space(generator)
```

### Challenges and Extensions

1. **Improve GAN Stability**:
   - Implement Wasserstein GAN (WGAN) or WGAN-GP
   - Try Spectral Normalization

2. **Conditional GAN**:
   - Modify the GAN to generate images of specific classes

3. **Style Transfer with GANs**:
   - Implement CycleGAN for unpaired image-to-image translation

4. **High-Resolution Image Generation**:
   - Implement Progressive GAN or StyleGAN for higher quality images

## Resources and References

For each project, here are additional resources to deepen your understanding:

### Image Classification
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

### Natural Language Processing
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### Generative AI
- [GAN Lab: Play with Generative Adversarial Networks](https://poloclub.github.io/ganlab/)
- [NVIDIA GAN Research](https://research.nvidia.com/research-area/generative-models)
- [GAN Hacks: Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

These practical projects will help you apply the theoretical knowledge from the earlier modules and build a strong portfolio of AI projects. Remember to experiment with different approaches and hyperparameters to deepen your understanding of each technique.
