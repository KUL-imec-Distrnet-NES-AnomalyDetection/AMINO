import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionGMM(nn.Module):
    def __init__(
        self, hidden_dims, reconstruct_dropout, latent_dim, gmm_dropout, n_gmm
    ):
        super().__init__()
        num_layer = len(hidden_dims) - 1
        # Decoder network
        layers = []
        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=reconstruct_dropout, inplace=False))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*layers)

        # Estimation network
        layers = []
        layers += [nn.Linear(hidden_dims[0] + 2, latent_dim)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=gmm_dropout)]
        layers += [nn.Linear(latent_dim, n_gmm)]
        layers += [nn.Softmax(dim=-1)]
        self.estimation = nn.Sequential(*layers)

    def forward(self, x, h):
        # x: batch, time, faeture
        x_hat = self.decoder(h)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([h, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=-1)
        gamma = self.estimation(z)
        return x_hat, z, gamma
