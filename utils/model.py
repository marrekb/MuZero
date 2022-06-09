import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x) + input
        return F.relu(x)

'''
methods support_to_scalar, scalar_to_support and normalize_output are
implemented by Davaud Werner in github.com/werner-duvaud/muzero-general
'''
def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits.view(-1, 2 * support_size + 1)

def normalize_output(x):
    _, dim1, dim2, dim3 = x.shape
    x_v = x.view(-1, dim1, dim2 * dim3)
    x_max = x_v.max(2, keepdim=True)[0].view(-1, dim1, 1, 1)
    x_min = x_v.min(2, keepdim=True)[0].view(-1, dim1, 1, 1)
    x_scale = x_max - x_min
    x_scale[x_scale < 1e-5] += 1e-5
    return (x - x_min) / x_scale
