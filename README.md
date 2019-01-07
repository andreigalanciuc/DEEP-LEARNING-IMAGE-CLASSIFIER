# DEEP-LEARNING-IMAGE-CLASSIFIER

## SOFTWARE REQUIREMENTS: pytorch

## PROJECT SUMMARY
In this project I built a deep learning model that classifies flower images.
I first normalized the data and segmented into training, validation and testing sets. The resizing is important here because if we do not resize the image and directly crop it, we might lose important information about the data image. Resizing reduces the size of a image while still holding full information of the image unlike a crop which blindly extracts one part of the image. I used a pretrained network from the torchvision.models and I added the LogSoftMax layer in the last layer of the network.
I defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout.
I trained the classifier layers using backpropagation using the pre-trained network to get the features.
I tracked the loss and accuracy on the validation set to determine the best hyperparameters.
I tested my trained model on the 1000 test images and got the accuracy of 86%. Finally I used the trained model for inferece, i.e pass an image into the network and predict the class of the flower in the image. 

