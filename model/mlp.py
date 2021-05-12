import torch
import torch.nn as nn
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(MLP, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.fc = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs*num_inputs)
            )
        self.soft = nn.Softmax(dim=-1)
    
    def forward(self, x):
        probs = self.fc(x)
        probs = self.soft(probs.view(-1, self.num_inputs, self.num_outputs))
        dist = Categorical(probs)
        return dist

if __name__ == "__main__":
    inp = torch.rand(1, 1)
    net = MLP(1, 10, 256)
    out = net(inp)
    action = out.sample()
    log_prob = out.log_prob(action)
    print(log_prob)