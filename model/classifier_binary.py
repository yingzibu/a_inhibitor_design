import torch.nn as nn

class Classifier_binary(nn.Module):
    """
    simple classifier, for prediction using latent space z
    https://dejanbatanjac.github.io/2019/07/04/softmax-vs-sigmoid.html
    """ 
    def __init__(self, dims):
        super(Classifier_binary, self).__init__()
        [in_dim, h_dims] = dims
        # assert out_dim == 1
        neurons = [in_dim, *h_dims] 
        linear_layers = [nn.Linear(
            neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        
        self.hidden = nn.ModuleList(linear_layers)
        self.final = nn.Linear(h_dims[-1], 1)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
            # print(x.shape) [batch_size, h_dim[-1]]
        x = self.final(x)
        x = torch.sigmoid(x)
        return x
