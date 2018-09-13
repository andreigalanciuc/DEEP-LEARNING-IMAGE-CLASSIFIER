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
import utility
import argparse
import json

ap = argparse.ArgumentParser(description='predict class of flower')
ap.add_argument('path_img', default='aipnd-project/flowers/test/10/image_07117.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--labels', dest="labels", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", nargs = '*', action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.path_img
path = pa.checkpoint

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
model = utility.load_checkpoint(path)

utility.predict(path_image, model, topk=5)
                   
