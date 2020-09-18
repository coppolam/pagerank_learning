import torch

class net(torch.nn.Module):
    def __init__(self,n_inputs,n_outputs,layer_size=30,layers=3):
        '''Initialization function. Set here the hyperparameters'''
        super(net,self).__init__() # Get nn.Module properties
        
        # Initialize the network
        self.fc_in = torch.nn.Linear(n_inputs,layers)
        self.fc_mid = torch.nn.Linear(layers,layers)
        self.fc_out = torch.nn.Linear(layers,n_outputs)
        self.relu = torch.nn.ReLU()
        self.layers = layers

    def forward(self,x):
        # Run input layer
        x = self.relu(self.fc_in(x))

        # Run middle layers
        for i in range(self.layers):
            x = self.relu(self.fc_mid(x))

        # Output layer
        return self.fc_out(x)