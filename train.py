#
#  Model based on TowardsDataScience article:
#  https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
#

# Imports
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
#from torch.package import PackageExporter  # only needed if exporting the package

# Load Data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('/global/cscratch1/sd/sconlon/torch_deploy_toy/data/train', download=True, train=True, transform=transform)
valset = datasets.MNIST('/global/cscratch1/sd/sconlon/torch_deploy_toy/data/eval', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print('image shape:', images.shape)
print('label shape:', labels.shape)

# Define Model
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print('model:', model)

# Train Model
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

print("\nTraining Time (in minutes) =",(time()-time0)/60)


# Script model
scripted_model = torch.jit.script(model)
print("scripted model")
print(scripted_model.code)

# Save scripted model
scripted_model.save("scripted_mnist.pt")

"""
Run inference on a example image
(only neccessary if using torch.deploy)

# Test Infer Image
images, labels = next(iter(valloader))
example_input = images[0].view(1, 784)
with torch.no_grad():
    logps = model(example_input)
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
"""



"""
Save the model using torch.package 
(only neccessary if using torch.deploy)

# Model Export
with PackageExporter('mnist') as e:
    # Configure how to export the source code
    e.extern(['torch.**'])
    # instead of saving the source code for the
    # torch libraries, let the package link to
    # the libraries of the loading process.

    # Replace these modules with mock implementations.
    # They are not actually used.
    e.mock(['numpy', 'librosa.**', 'scipy.**'])

    e.save_pickle('model', 'model.pkl', model)
    # dependency resolution will walk the pickled objects
    # and find all the required source files

    # also save example tensor for testing
    e.save_pickle('model', 'eg.pkl', example_input)
"""
