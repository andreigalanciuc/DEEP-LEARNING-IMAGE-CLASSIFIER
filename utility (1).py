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
import PIL
from PIL import Image
import numpy as np
import argparse

ap = argparse.ArgumentParser(description='utility help')
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', action="store_true")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
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

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
data_transformations = {'train_transform':transforms.Compose([transforms.RandomRotation(40),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
                       'valid_transform': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
                       'test_transform': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])}

image_datasets = {'train_image': datasets.ImageFolder(train_dir, transform = data_transformations['train_transform']),
                 'valid_image': datasets.ImageFolder(valid_dir, transform = data_transformations['valid_transform']),
                 'test_image': datasets.ImageFolder(test_dir, transform = data_transformations['test_transform'])}

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train_image'], batch_size = 64, shuffle = True),
              'valid': torch.utils.data.DataLoader(image_datasets['valid_image'], batch_size = 32),
              'test': torch.utils.data.DataLoader(image_datasets['test_image'], batch_size = 32)}
    
def design_model(architecture = 'vgg16', dropout=0.5, learningrate = 0.001, hidden_units = 4096):
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False

        
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout)),
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
     
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        
        optimizer = optim.Adam(model.classifier.parameters(), learningrate)
        
        model.cuda()
        
        return model , optimizer ,criterion
    
def train_model(model, criterion, optimizer, epochs, gpu):
    
    print_every = 40
    steps = 0
    if gpu and torch.cuda.is_available():
        model.cuda()

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
        
            inputs,labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                loss_v = 0
                accuracy=0
            
            
                for ii, (inputs_v,labels_v) in enumerate(dataloaders['valid']):
                    optimizer.zero_grad()
                
                    inputs_v, labels_v = inputs_v.to('cuda:0') , labels_v.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs_v)
                        loss_v = criterion(outputs,labels_v)
                        ps = torch.exp(outputs).data
                        equality = (labels_v.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                loss_v = loss_v / len(dataloaders['valid'])
                accuracy = accuracy /len(dataloaders['valid'])
            
                    
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(loss_v),
                   "Validation Accuracy: {:.4f}".format(accuracy))
            
            
                running_loss = 0
                
def save_checkpoint(model, optimizer, epochs):
    model.class_to_idx = image_datasets['train_image'].class_to_idx
    torch.save({
                'architecture': 'vgg16',
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer': optimizer.state_dict(),
                'epochs' : epochs},
                'checkpoint.pth')
        
def load_checkpoint(filepath):
    checkpoint = torch.load('checkpoint.pth')
    architecture = checkpoint['architecture']   
    model, optimizer, criterion = design_model(architecture, dropout, lrate, hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    img_pil = Image.open(image_path)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)