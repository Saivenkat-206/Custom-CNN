if __name__ == '__main__':
    import torch
    #import matplotlib.pyplot as plt
    #import numpy as np
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    batch_size = 12

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    class CNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()

            self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling squashes everything to 1x1 feature maps
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

            self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(1024, num_classes)
        )

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = self.flatten(x)
            x = self.classifier(x)
            return x

    net = CNN()
    net.to(device)

    for i, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        print("input shape: ", inputs.shape)
        print("after network shape: ", net(inputs).shape)
        
        break

    # batch of 6 of 32 images have been converted to 6 batches of 10 dim vectors

    # checking the parameters of the model
    num_params = 0
    for x in net.parameters():
        num_params += len(torch.flatten(x))

    print("Number of parameters in the model: ", num_params)

    # training the model
    criterion = nn.CrossEntropyLoss() # this is the loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # this is the optimizer

    def train_one_epoch():
        net.train(True) # set the model to training mode
        running_loss = 0.0
        running_accuracy = 0.0

        #iterate over the training set
        for batch_index, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)

            #zero the parameter gradients
            optimizer.zero_grad()
            #forward pass
            outputs = net(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct/batch_size
            
            loss = criterion(outputs, labels) #avg loss of each batch
            running_loss += loss.item() #avg loss of each batch
            loss.backward()
            optimizer.step()

            if batch_index % 500 == 499:
                avg_loss_across_batches = running_loss / 500
                avg_acc_accross_batches = (running_accuracy / 500) * 100
                print(f"Batch: {batch_index+1}, Loss: {avg_loss_across_batches:.3f}, Accuracy: {avg_acc_accross_batches:.3f}%")

                running_loss = 0.0
                running_accuracy = 0.0

        print()
    
    def validate_one_epoch():
        net.train(False)
        running_loss = 0.0
        running_accuracy = 0.0
        
        for i,data in enumerate(valLoader):
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad(): # don't worry about the gradients
                outputs = net(inputs)
                correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                running_accuracy += correct/batch_size

                loss = criterion(outputs, labels)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(valLoader)
        avg_acc_across_batches = (running_accuracy / len(valLoader)) * 100

        print(f"Validation Loss: {avg_loss_across_batches:.3f}, Validation Accuracy: {avg_acc_across_batches:.3f}%")
        print()

    #training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1)

        train_one_epoch()
        validate_one_epoch()

    print("Finished training")
