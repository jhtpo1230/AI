import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(root='MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
mnist_test = datasets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

BATSIZE = 10000

train_loader = DataLoader(dataset=mnist_train,
                          batch_size=BATSIZE,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=mnist_test,
                          batch_size=len(mnist_test),
                          shuffle=False,
                          num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
