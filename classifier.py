import torch
import torch.nn as nn


class Classification(nn.Module):
    """One-layer classification model
    Parameters:
        input_size: Input dimension
        num_classes: Number of categories
    return:
        logists: The maximum probability corresponds to the label
    """
    def __init__(self, input_size, num_classes):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        logists = torch.log_softmax(self.fc1(x), 1)
        return logists
    
class Predictor(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Predictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        output = self.mlp(x)

        return output
