A clean and simple PyTorch code to train a convolutional neural net (CNN) on the CIFAR-10 image dataset. This thing slaps for educational purposes or as a starting point for deeper experiments with CNNs.

### Features
- Uses PyTorch + torchvision
- Trains a 3-layer CNN with batch norm, ReLU, dropout
- Tracks training and validation accuracy/loss
- Trains on GPU if available
- Custom training and validation loops (no magic, all manual)

CIFAR-10 image dataset 60,000 32x32 color images in 10 classes
Split:
45k training
5k validation
10k test (unused in training loop but loaded)

Did this to learn how to code a neural network in pytorch so the accuracy might be questionable
