import torch
import torch.nn as nn

# this module is base on architecture github.com/michalnand/reinforcement_learning

class RND_Model(torch.nn.Module):
    def __init__(self, count_of_features):
        super(RND_Model, self).__init__()

        self.model =  nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8,
                      stride = 4, padding = 2),
            nn.ELU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size =3 ,
                      stride = 2, padding = 1),
            nn.ELU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(count_of_features, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)
        )

        self.target_model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8,
                      stride = 4, padding = 2),
            nn.ELU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size =3 ,
                      stride = 2, padding = 1),
            nn.ELU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(count_of_features, 512)
        )

        for i in range(len(self.target_model)):
            if hasattr(self.target_model[i], "weight"):
                wa, wb = self._coupled_ortohogonal_init(self.model[i].weight.shape, 2.0**0.5)

                self.model[i].weight        = torch.nn.Parameter(wa, requires_grad = True)
                self.target_model[i].weight = torch.nn.Parameter(wb, requires_grad = True)

                torch.nn.init.zeros_(self.model[i].bias)
                torch.nn.init.zeros_(self.target_model[i].bias)

        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.eval()

    def forward(self, x):
        return self.model(x), self.target_model(x)

    def _coupled_ortohogonal_init(self, shape, gain):
        w = torch.zeros((2*shape[0], ) + shape[1:])
        torch.nn.init.orthogonal_(w, gain)

        w = w.reshape((2, ) + shape)
        return w[0], w[1]