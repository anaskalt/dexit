import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy

def softmax_temperature(logits, temperature=5):
    return torch.exp(logits / temperature) / torch.sum(torch.exp(logits / temperature))

class EarlyExitNet(nn.Module):
    def __init__(self,network,input_shape,device,thresholds,exit_layers = 16,scaler=None):
        def get_output_flattened( network, input_shape):
            x = torch.rand(input_shape).to(self.device)  # Add the batch size dimension here
            return network(x).view(x.size(0), -1).size(1)
        
        super(EarlyExitNet, self).__init__()  
        self.device = device
        self.thresholds  = thresholds
        self.network = network.to(self.device)
        x = torch.rand(input_shape).to(self.device)
        
        if isinstance(exit_layers,int):
            layers  =  exit_layers
            exit_layers = [[layers] for _ in range(len(network)-1)]
        self.output_shape  =  network(x).size(1)
        
        self.len = len(network)
        assert len(exit_layers) == len(network)-1

        self.exits = nn.ModuleList([])
        for i, net in enumerate(network[:-1]):
            layers = [nn.Flatten()]
            last_layer_size  = get_output_flattened(net,input_shape)
            for next_layer_size in exit_layers[i]:
                  
                layers.append(nn.Linear(last_layer_size, next_layer_size))
                layers.append(nn.ReLU())
                last_layer_size = next_layer_size
            layers.append(nn.Linear(last_layer_size, self.output_shape))
            
            input_shape  = net(torch.rand(input_shape).to(self.device)).size()
        
            self.exits.append(nn.Sequential(*layers).to(self.device))
            
        
        if scaler is None:
            self.scaler  = softmax_temperature
        else:   
            self.scaler = scaler
                    
    def forward(self,x):
        outputs =[]
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit = self.scaler(early_exit)
            outputs.append(early_exit)
            
        x = self.network[-1](x)
        x = self.scaler(x)
        outputs.append(x)
        return outputs
                
    def segmented_forward(self,x):
        x =  x.to(self.device)
        x = x.unsqueeze(0)
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit= early_exit.squeeze(0)
            early_exit = self.scaler(early_exit)

            if early_exit.max() > self.thresholds[i]:
                return early_exit,i
        x = self.network[-1](x)
        x = self.scaler(x)
        return x,i+1
    
def train(  model,
            epochs,
            trainloader,
            optimizer=None,
            criterion=None):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = 0
            for i in range(len(outputs)):
                loss += criterion(outputs[i], labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(epoch,'loss: ',running_loss)
        running_loss = 0.0

    print('Finished Training')

def test_accuracy(model, dataloader):
    with torch.no_grad():     
        images,_ = next(iter(dataloader))
        images  = images.to(model.device)
        num_exits = len(model(images))
        correct = [0] * num_exits
        total = [0] * num_exits
        for data in dataloader:
            images, labels = data
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output.data, 1)
                total[i] += labels.size(0)
                correct[i] += (predicted == labels).sum().item()

        for i, (c, t) in enumerate(zip(correct, total)):
            print(f'Accuracy of exit {i}: {100 * c / t}%')



def segmented_test_accuracy(model, dataloader):
    exits_chosen =  np.zeros(model.len)
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in dataloader:
            images, labels = images.to(model.device), labels.to(model.device)
            for img,lbl in zip(images,labels):
                output,exit = model.segmented_forward(img)  # Use the segmented_forward function
                prediction = torch.argmax(output).item()  # Get the predictions from the output
                total += 1
                correct += (prediction == lbl)
                exits_chosen[exit] +=1
    plt.bar(range(model.len),exits_chosen)
    plt.show()
    return correct / total ,exits_chosen


class EarlyExitNetworkSegmentor(nn.Module):
        def __init__(self,network,exit=None,threshold=None,device='cpu',scaler=None):
            super(EarlyExitNetworkSegmentor, self).__init__()
            self.network = copy.deepcopy(network)
            self.exit = copy.deepcopy(exit)
            self.threshold = threshold
            self.device=  device
            self.scaler = scaler
                
        def forward(self,x):
            x = x.to(self.device)
            x = x.unsqueeze(0)
            x = self.network(x)
            if self.exit != None:
                early_exit = self.exit(x)
                early_exit=  early_exit.squeeze(0)
                early_exit = self.scaler(early_exit)
                if early_exit.max() > self.threshold:
                            return early_exit,True
                return x.squeeze(0),False
            x = x.squeeze(0)
            return x,True
        
        
def seperate_networks(eenet):
    segmented_networks =[]
    scaler  = eenet.scaler
    for i in range((len(eenet.network)-1)):
        network = EarlyExitNetworkSegmentor(eenet.network[i],eenet.exits[i],eenet.thresholds[i],eenet.device,scaler)
        segmented_networks.append(network)
    network  = EarlyExitNetworkSegmentor(eenet.network[-1],device=eenet.device)
    segmented_networks.append(network)
    return segmented_networks


def main():
    return  

if __name__ == "main":
    main()