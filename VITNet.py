# import statements
import torch.nn as nn
import torchvision.models as tvm

# class definition for our network
class VITNet(nn.Module):
    # default constructor, taking as input the number of layers and the number
    #   of nodes per layer.
    def __init__(self,NumberOfLayers,Neurons):
        super(VITNet,self).__init__()
        # we initialize the vit_b_16, remake the 'head' then attach an MLP for analysis.
        # we can also just use the 'head' and disregard the MLP if we want.
        # the input of the 'head' is locked to 768 based on the ViT output
        # we set the requires_grad to false in order to tell the loss.backward() step not to look at those parameters
        self.vit = tvm.vit_b_16(weights = 'IMAGENET1K_V1')
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads = nn.Linear(768,Neurons)
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(Neurons,Neurons),
                                                   nn.ReLU()) for i in range(NumberOfLayers)])
        self.output = nn.Linear(Neurons,10)
    
    # TODO: Where is this function actually used?
    # definition of function to handle forward pass of the network.
    def forward(self,data):
        x = self.vit(data)
        for layer in self.hidden:
            x = layer(x)
        guess = self.output(x)

        # returning the network's output
        return guess