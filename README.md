# DEEP-LEARNING-IMAGE-CLASSIFIER

## SOFTWARE REQUIREMENTS: pytorch

## PROJECT SUMMARY
In this project I build a deep learning model that classifies flowers.
I first normalized the data and segmented into training, validation and testing sets. The resizing is important here because if we do not resize the image and directly crop it, we might lose important information about the data image. Resizing reduces the size of a image while still holding full information of the image unlike a crop which blindly extracts one part of the image. I used a pretrained network from the torchvision.models and I added the LogSoftMax layer in the last layer of the network.
