import matplotlib.pyplot as plt
import time
import torch
import torchvision
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from PIL import Image
import numpy as np
import utility
import argparse

ap = argparse.ArgumentParser(description='train model')
ap.add_argument('data_dir', nargs='*', action="store", default="flowers")
ap.add_argument('--gpu', action="store_true")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

pa = ap.parse_args()
directory = pa.data_dir
checkpoint = pa.save_dir
lrate = pa.learning_rate
architecture = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

model, optimizer, criterion = utility.design_model(architecture, dropout, lrate, hidden_units)

utility.train_model(model, criterion, optimizer, 1, power)

utility.save_checkpoint(model, optimizer, epochs)