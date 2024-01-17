import torch
import torch.nn as nn


class GMMLoss(nn.Module):
    def __init__(self, lambda_energy, lambda_cov, n_gmm):
        super().__init__()
        self.lambda_energy = nn.Parameter(
            torch.tensor(lambda_energy), requires_grad=False
        )
        self.lambda_cov = nn.Parameter(torch.tensor(lambda_cov), requires_grad=False)
        self.n_gmm = nn.Parameter(torch.tensor(n_gmm), requires_grad=False)
        self.criterion_MSE = nn.MSELoss()

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = self.criterion_MSE(x_hat, x)  # output, train_data
        sample_energy, cov_diag = self.compute_energy(z, gamma)
        loss = (
            reconst_loss
            + self.lambda_energy * sample_energy
            + self.lambda_cov * cov_diag
        )
        return loss

    def compute_energy(self, z, gamma):
        """Computing the sample energy function"""
        phi, mu, cov = self.compute_params(z, gamma)

        z_mu = z.unsqueeze(-2) - mu

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(
                (Cholesky.apply(cov_k.cpu() * (2 * torch.pi)).diag().prod()).unsqueeze(
                    0
                )
            )
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.lambda_energy.device)

        E_z = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1,
        )
        E_z = torch.exp(E_z)
        E_z = -torch.log(
            torch.sum(phi * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=-1) + eps
        )
        E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function"""
        # phi = D
        phi = torch.sum(gamma, dim=[0, 1, 2], keepdim=True) / (
            gamma.size(0) * gamma.size(1) * gamma.size(2)
        )
        # phi = gamma.mean()

        # mu = KxD
        mu = torch.sum(
            z.unsqueeze(-2) * gamma.unsqueeze(-1), dim=[0, 1, 2], keepdim=True
        )
        mu /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1)

        a = z.unsqueeze(-2)
        z_mu = z.unsqueeze(-2) - mu  # z:B,C,T,H (combine dimension), mu:(1,1,1,H,)
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        a = gamma.unsqueeze(-1).unsqueeze(-1)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=[0, 1, 2])
        cov /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        # l = torch.cholesky(a, False)
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    """
       L = torch.cholesky(A)
       should be replaced with
       L = torch.linalg.cholesky(A)
       and
       U = torch.cholesky(A, upper=True)
       should be replaced with
       U = torch.linalg.cholesky(A).transpose(-2, -1).conj().
       """

    def backward(ctx, grad_output):
        (l,) = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - l.data.new(l.size(1)).fill_(0.5).diag()
        )
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
