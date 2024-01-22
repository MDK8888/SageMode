import torch
import torch.nn as nn
import torch.distributed as dist

class DistributedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DistributedLinear, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Perform the forward pass using the original linear layer
        output = self.linear(x)

        # Synchronize parameters using all_reduce
        self.all_reduce_parameters()

        return output

    def all_reduce_parameters(self):
        # All-reduce the weight and bias parameters
        dist.all_reduce(self.linear.weight.data)
        if self.linear.bias is not None:
            dist.all_reduce(self.linear.bias.data)

        # Normalize the parameters (optional, depends on your use case)
        self.linear.weight.data /= dist.get_world_size()
        if self.linear.bias is not None:
            self.linear.bias.data /= dist.get_world_size()
