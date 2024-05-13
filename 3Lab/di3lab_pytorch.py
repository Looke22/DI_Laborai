import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

BATCH_SIZE = 128
EPOCHS = 1

#region Prepare the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Data augmentation transformations
transform_augmented = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.ColorJitter(contrast= 0.005),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Prepare the data with augmentation
train_set_augmented = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmented)

# Combine original and augmented datasets
train_set_combined = torch.utils.data.ConcatDataset([train_set, train_set_augmented])
#endregion
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Create data loaders
#Original
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
#Augmented
train_augmented_loader = torch.utils.data.DataLoader(train_set_augmented, batch_size=BATCH_SIZE, shuffle=False)
#Combined
train_loader = torch.utils.data.DataLoader(train_set_combined, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Load a batch of data
data_iter = iter(train_set_loader)
images, labels = next(data_iter)

# Define a list to store original images
original_images = []
for i in range(len(images)):
    original_images.append(images[i].numpy())  # Convert torch tensor to numpy array
    
# Load a batch of augmented data
data_iter = iter(train_augmented_loader)
images, labels = next(data_iter)

augmented_images = []
for i in range(len(images)):
    augmented_images.append(images[i].numpy()) # Convert torch tensor to numpy array


def plot_images(original_images, augmented_images, labels, num_images=3):
    plt.figure(figsize=(12, 5))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].squeeze(), cmap='gray_r')
        plt.title("Original: {}".format(int(labels[i])))
        plt.axis("off")
        
        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[i].squeeze(), cmap='gray_r')
        plt.title("Augmented: {}".format(int(labels[i])))
        plt.axis("off")
    # Save the figure
    plt.savefig('original_vs_augmentedTorch.png')
    plt.show()

plot_images(original_images, augmented_images, labels, 3)


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=2)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool2(nn.functional.relu(self.conv1(x)))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool2(nn.functional.relu(self.conv4(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
best_loss = float('inf')
best_model_path = 'best_model.pth'
last_model_path = 'last_model.pth'
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    # Validation
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()
    
    val_accuracy = 100 * correct / total
    val_loss /= len(test_loader)
    print('Epoch %d - Validation Loss: %.3f, Validation Accuracy: %.2f %%' % (epoch + 1, val_loss, val_accuracy))
    
    # Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
    
    # Save the last model
    torch.save(model.state_dict(), last_model_path)

print('Finished Training')


# Evaluate the best model
correct_best = 0
total = 0
best_model = Net()
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_best += (predicted == labels).sum().item()

best_accuracy = 100 * correct_best / total

print('Accuracy of the best model on the test images: {:.2f} %'.format(best_accuracy))

# Evaluate the last model
correct_last = 0
total = 0
last_model = Net()
last_model.load_state_dict(torch.load(last_model_path))
last_model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = last_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_last += (predicted == labels).sum().item()

last_accuracy = 100 * correct_last / total

print('Accuracy of the last model on the test images: {:.2f} %'.format(last_accuracy))

# Compare the results
if best_accuracy > last_accuracy:
    print("Best model performs better.")
elif best_accuracy < last_accuracy:
    print("Last model performs better.")
else:
    print("Both models have the same accuracy.")


def plot_predictions(model, images, num_images=3):
    plt.figure(figsize=(15, 9))
    num_total_images = images.shape[0]
    
    random_indices = np.random.choice(num_total_images, size=num_images*2, replace=False)
    
    best_indices = np.argsort(np.max(num_total_images, axis=1))[-num_images:][::-1]
    worst_indices = np.argsort(np.max(num_total_images, axis=1))[:num_images]
    
    
    model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(best_indices):
            image = images[idx].unsqueeze(0)  # Add batch dimension
            
            # Prediction for the best model
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_classes = probs.topk(3)
            top_probs = top_probs.numpy()
            top_classes = top_classes.numpy()
            
            # Plot for the best model
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
            
            title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(top_classes, top_probs)])
            plt.title(f'Best Model\n{title_text}', color='#017653')
            plt.axis("off")
            
    #Plot the worst predictions
    for i, idx in enumerate(worst_indices):
            image = images[idx].unsqueeze(0)  # Add batch dimension
            
            # Prediction for the best model
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_classes = probs.topk(3)
            top_probs = top_probs.numpy()
            top_classes = top_classes.numpy()
            
            # Plot for the best model
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
            
            title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(top_classes, top_probs)])
            plt.title(f'Best Model\n{title_text}', color='#017653')
            plt.axis("off")
    
    plt.savefig('last_and_best_comparison_images.png')  
    plt.show()
    
plot_predictions(best_model, images, 3)