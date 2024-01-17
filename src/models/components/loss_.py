import torch
import torch.nn as nn


class GMMLoss(nn.Module):
    def __init__(self, lambda_energy, lambda_cov, n_gmm, eps=1e-9):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.n_gmm = n_gmm
        self.eps = eps

    def forward(self, z, gamma):
        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return loss

    def compute_energy(self, z, gamma):
        phi, mu, cov = self.compute_params(z, gamma)
        z_mu = z.unsqueeze(-2) - mu

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(gamma.size(-1)):
            cov_k = cov[k] + torch.eye(cov[k].size(-1)).to(z.device) * self.eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((torch.linalg.det(cov_k.cpu()) + self.eps).unsqueeze(0))
            cov_diag += torch.sum(1 / (cov_k.diag() + self.eps))

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov)

        E_z = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1,
        )
        E_z = torch.exp(E_z)
        E_z = -torch.log(
            torch.sum(phi * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=-1) + self.eps
        )
        E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        phi = torch.sum(gamma, dim=[0, 1, 2], keepdim=True) / (
            gamma.size(0) * gamma.size(1) * gamma.size(2)
        )

        mu = torch.sum(
            z.unsqueeze(-2) * gamma.unsqueeze(-1), dim=[0, 1, 2], keepdim=True
        )
        mu /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1)

        z_mu = z.unsqueeze(-2) - mu
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=[0, 1, 2])
        cov /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov


if __name__ == "__main__":
    n_gmm = 4
    compute_loss = GMMLoss(0.1, 0.1, n_gmm)

    z = torch.randn(10, 1, 1024, 10)
    gamma = torch.randn(10, 1, 1024, n_gmm)

    loss = compute_loss(z, gamma)

    print(loss)
