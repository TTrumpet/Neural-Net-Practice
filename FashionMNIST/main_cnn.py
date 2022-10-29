#---------#
# Imports #
#---------#
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

#---------------------------------------#
# Create Fully Connected Neural Network #
#---------------------------------------#
class CNN(nn.Module):
    def __init__(self, in_channels, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(input_size,120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
	x = F.relu(x)
        x = self.fc3(x)
        return x

#------------#
# Set Device #
#------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----------------#
# Hyperparameters #
#-----------------#

# Classifies fashion from 10 options:
# 	0 T-shirt/top
# 	1 Trouser
# 	2 Pullover
# 	3 Dress
# 	4 Coat
# 	5 Sandal
# 	6 Shirt
# 	7 Sneaker
# 	8 Bag
# 	9 Ankle boot 
in_channels = 1
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

#-----------#
# Load Data #
#-----------#

# Training dataset
train_dataset = datasets.FashionMNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Testing dataset
test_dataset = datasets.FashionMNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#--------------------#
# Initialize Network #
#--------------------#
model = CNN(in_channels=in_channels, input_size=input_size, num_classes=num_classes).to(device)

#------------------#
# Loss & Optimizer #
#------------------#
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#----------------#
# Check Accuracy #
#----------------#
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return float(num_correct)/float(num_samples)*100
#---------------#
# Train Network #
#---------------#
loss_median = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):

    loss_values = []

    for batch_idx, (data, targets) in enumerate((train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        loss_values.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    train_accuracy.append(check_accuracy(train_loader, model))
    test_accuracy.append(check_accuracy(test_loader, model))

    loss_median.append(stat.median(loss_values))

#----------------------------------------#
# Create Visualization using matplotlib #
#----------------------------------------#

# Loss over Epochs
plt.plot(loss_median, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Training Accuracy over Epochs
plt.plot(train_accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.show()

# Test Accuracy over Epochs
plt.plot(test_accuracy, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.show()
