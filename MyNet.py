# import statements
import torch.nn as nn

# class definition for our network
class MyNet(nn.Module):
    # default constructor, taking as input the number of layers and the number
    #   of nodes per layer.
    def __init__(self,NumberOfLayers,Neurons):
        super(MyNet,self).__init__()

        # the input layer. We flatten the input images into 1-dimension tensors,
        #   and then pass them into a linear layer with a ReLU.
        self.input = nn.Sequential(nn.Flatten(start_dim=1),
                                   nn.Linear(373248,Neurons),
                                   nn.ReLU())
        
        # for the hidden layers, they all have the below structure
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(Neurons,Neurons),
                                                   nn.ReLU()) for i in range(NumberOfLayers)])
        
        # The output layer is just a linear activation 
        self.output = nn.Linear(Neurons,10)
    
    # definition of function to handle forward pass of the network.
    def forward(self,data):
        # these are the dimensions of the gztan images
        x = data.reshape(len(data),3,288,432)
        x = self.input(x)
        for layer in self.hidden:
            x = layer(x)
        guess = self.output(x)

        # returning the network's output
        return guess
    
    # this function is the training loop for the model
    # TODO: what even is the point of having this and the validation loop here?
    # It takes as input:
    #   dataset: the data being used, which corresponds to the specific dataloader
    #   TODO model: why are you passing the model to itself????? can you not just self.___?
    #   criterion: The loss function (cross-entropy)
    #   optimizer: the optimizer being used.
    def trainloop(self,dataset,model,criterion,optimizer):
        # declaring lists for holding the intermediary metrics
        running_loss = []
        running_accuracy = []

        # going through the data batches in the dataloader
        for i,data in enumerate(dataset):
            # for the accuracy calculation
            num_correct = 0
            num_samples = 0

            inputs,labels = data

            # initializing the gradients to 0.
            optimizer.zero_grad()

            # executing the prediction and and calculating the loss
            pred = model(inputs)
            loss = criterion(pred,labels)
            loss.backward()
            optimizer.step()

            # calculating the accuracy
            _,predictions = pred.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            accuracy = float(num_correct/num_samples)*100

            # appending the training accuracy and loss
            running_accuracy.append(accuracy)
            running_loss.append(loss.item())
            print(f' Training Batch {i+1}: {accuracy}')

        # returning the lists of the calculated loss and accuracy
        return running_loss,running_accuracy
    
    # function used for calculating the validation accuracies and loss
    # takes as input:
    #   dataset: the dataloader
    #   model: the model being used
    #   criterion: the loss function (cross entropy)
    def validationloop(self,dataset,model,criterion):
        # declaring lists to hold the calculated metrics
        running_loss = []
        running_accuracy = []

        # loop for going through the batches 
        for i,data in enumerate(dataset):
            # for calculating the validation accuracy
            num_correct = 0
            num_samples = 0
            inputs,labels = data

            # exectuing the prediction and calculating the loss.
            pred = model(inputs)
            loss = criterion(pred,labels)
            running_loss.append(loss.item())

            # calculating the accuracy and recording it
            _,predictions = pred.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            accuracy = float(num_correct/num_samples)*100
            running_accuracy.append(accuracy)
            print(f'Validation: {accuracy}\n')

        # returning the calculated loss and accuracy lists
        return running_loss,running_accuracy
    
