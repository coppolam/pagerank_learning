import torch

class net(torch.nn.Module):
    def __init__(self,n_inputs,n_outputs,layer_size=30,layers=3):
        '''Initialization function. Set here the hyperparameters'''
        super(net,self).__init__()
        
        # Layers
        self.fc_in = torch.nn.Linear(n_inputs,layer_size)
        self.fc_mid = torch.nn.Linear(layer_size,layer_size)
        self.fc_out = torch.nn.Linear(layer_size,n_outputs)

        # ReLU
        self.relu = torch.nn.ReLU()

        # Number of layers
        self.layers = layers

    def forward(self,x):
        # Run input layer
        x = self.relu(self.fc_in(x))

        # Run middle layers
        for i in range(self.layers):
            x = self.relu(self.fc_mid(x))

        # Output layer
        return self.fc_out(x)