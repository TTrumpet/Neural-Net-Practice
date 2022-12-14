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

class NN(nn.Module):
	def __init__(self, input_size, num_classes): 
		super(NN, self).__init__()
		self.fc1 = nn.Linear(input_size, 50)
		self.fc2 = nn.Linear(50, num_classes)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
# 28 x 28 pixel pictures 
input_size = 784
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
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Training dataset
train_dataset = datasets.FashionMNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Testing dataset
test_dataset = datasets.FashionMNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
			x = x.reshape(x.shape[0], -1)
		
			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)
	model.train()
	return float(num_correct)/float(num_samples)*100

loss_median = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):

	loss_values = []

	for batch_idx, (data, targets) in enumerate(train_loader):
		data = data.to(device=device)
		targets = targets.to(device=device)
		
		data = data.reshape(data.shape[0], -1)
		
		scores = model(data)
		loss = criterion(scores, targets)

		loss_values.append(loss.item())

		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

	train_accuracy.append(check_accuracy(train_loader, model))
	test_accuracy.append(check_accuracy(test_loader, model))


	loss_median.append(stat.median(loss_values))

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
