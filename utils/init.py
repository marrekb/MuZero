import torch.nn.init as init

def init_orthogonal_head(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 0.01)
        init.zeros_(layer.bias)

def init_orthogonal_features(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 2**0.5)
        init.zeros_(layer.bias)
