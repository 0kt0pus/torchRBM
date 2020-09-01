import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from rbm import RBM

## CONFIG ##
BATCH_SIZE = 64
VISIBLE_UNITS = 784
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = 10

DATA_FOLDER = 'data/mnist'
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

## set GPU
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

## Download and load the mnist examples
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

## Training ##
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0
    for batch, _ in train_loader:
        ## flatten the input data
        batch = batch.view(len(batch), VISIBLE_UNITS)
        ## set GPU
        if CUDA:
            batch = batch.cuda()
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))