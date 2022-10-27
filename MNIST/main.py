#---------#
# Imports #
#---------#

import torch

# Neural Network functions
import torch.nn as nn 

# Optimization functions
import torch.optim as optim

# Functions without any parameters (etc. ReLu)
import torch.nn.functional as F

# Easier data set management
from torch.utils.data import DataLoader

# Importing datasets (e.g. MNIST, FashionMNIST)
import torchvision.datasets as datasets

# Contains transformations to perform on data
import torchvision.transforms as transforms


# Visualizing the data!
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

#---------------------------------------#
# Create Fully Connected Neural Network #
#---------------------------------------#

# class NN inherits from nn.Module
class NN(nn.Module):
	def __init__(self, input_size, num_classes): #input_size = 28 x 28 = 784 nodes

		# super calls initialization method of the parent class
		super(NN, self).__init__()

		# First layer of the neural net
		# - Linear
		# - Shrinks 784 > 50 nodes
		self.fc1 = nn.Linear(input_size, 50)

		# Hidden layer 
		# - Linear
		# - Shrinks 50 > num_classes nodes
		self.fc2 = nn.Linear(50, num_classes)

	# Forward propagation aka feeding in the input data
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

#------------#
# Set Device #
#------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Note: CUDA device is a single unit device that can support CUDA
# Mostly will just default to your CPU

#-----------------#
# Hyperparameters #
#-----------------#

# 28 x 28 pixel pictures
input_size = 784
# Either a handwritten digit 0-9
num_classes = 10
# Number determining step size 
learning_rate = 0.001
# Number of pictures per batch
batch_size = 64
# Number of runs through data
num_epochs = 10

#-----------#
# Load Data #
#-----------#

# Training dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Testing dataset
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#--------------------#
# Initialize Network #
#--------------------#
model = NN(input_size=input_size, num_classes=num_classes).to(device)

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
	model.eval() # evaluation mode

	# don't need to calculate gradient, unnecessary computation
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)
			x = x.reshape(x.shape[0], -1)
		
			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

		# Can't concatenate str with Tensors so must use an f string
		# print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
	model.train()
	return float(num_correct)/float(num_samples)*100

# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)

#---------------#
# Train Network #
#---------------#

loss_median = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):

	loss_values = []

	for batch_idx, (data, targets) in enumerate(train_loader):
		# Get data to cuda if possible
		data = data.to(device=device)
		targets = targets.to(device=device)
		
		# Get to correct shape
		data = data.reshape(data.shape[0], -1)
		
		# forward
		scores = model(data)
		loss = criterion(scores, targets)

		loss_values.append(loss.item())

		# backward
		optimizer.zero_grad() # set gradients to zero
		loss.backward()

		# gradient descent or adam step
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
