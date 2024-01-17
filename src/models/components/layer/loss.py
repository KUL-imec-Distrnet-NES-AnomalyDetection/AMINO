import logging

import torch
import torch.nn as nn

class GMMLoss(nn.Module):
    def __init__(self, weight_energy, weight_cov, eps=1e-9, cov_eps=1e-6):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_cov = weight_cov
        self.eps = eps
        self.cov_eps = cov_eps

    def forward(self, z, gamma):
        sample_energy, cov_diag = self.compute_energy(z, gamma)

        # Adding a check for NaN in reconst_loss, sample_energy, and cov_diag
        if not torch.isfinite(sample_energy):
            logging.warning("NaN or inf found in sample_energy")
            sample_energy = torch.zeros_like(sample_energy)
        if not torch.isfinite(cov_diag):
            logging.warning("NaN or inf found in cov_diag")
            cov_diag = torch.zeros_like(cov_diag)

        loss = (
            self.weight_energy * sample_energy
            + self.weight_cov * cov_diag
        )
        return loss

    def compute_energy(self, z, gamma):
        phi, mu, cov = self.compute_params(z, gamma)
        z_mu = z.unsqueeze(-2) - mu

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(gamma.size(-1)):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)).to(device=z.device) * self.eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(
                (Cholesky.apply(cov_k * (2 * torch.pi)).diag().prod()).unsqueeze(0)
            )
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov)
        assert torch.all(det_cov > 0)

        E_z = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1,
        )

        E_z = torch.exp(E_z)

        # Adding a larger epsilon before the log operation for stability
        E_z = -torch.log(
            torch.sum(phi * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=-1)
            + self.cov_eps
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

        # Regularization to ensure positive-definiteness
        for k in range(gamma.size(-1)):
            cov[k] += torch.eye(cov[k].size(-1)).to(cov.device) * self.cov_eps

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        try:
            logging.warning("Cholesky forward")
            l = torch.linalg.cholesky(a)
            ctx.save_for_backward(l)
            return l
        except RuntimeError as err:
            if "positive-definite" in str(err):
                # Handle non positive-definite case
                # Return a default value or perform alternative computation
                logging.warning("error in Cholesky forward with non positive-definite matrix")
                return torch.eye(a.size(0)).to(
                    a.device
                )  # Example: return identity matrix
            else:
                raise err  # Re-raise exception if it's a different error

    @staticmethod
    def backward(ctx, grad_output):
        (l,) = ctx.saved_tensors
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - l.data.new(l.size(1)).fill_(0.5).diag()
        )
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


if __name__ == "__main__":
    # Initialize the ComputeLoss class
    n_gmm = 4
    compute_loss = GMMLoss(0.1, 0.1)

    # Create some random tensors for x, x_hat, z, and gamma
    z = torch.randn(10, 1, 1024, 10)
    gamma = torch.randn(10, 1, 1024, n_gmm)

    # Print NaN checks
    print("NaN in z:", torch.isnan(z).any())
    print("Inf in z:", torch.isinf(z).any())
    print("NaN in gamma:", torch.isnan(gamma).any())
    print("Inf in gamma:", torch.isinf(gamma).any())

    # Compute the loss
    loss = compute_loss(z, gamma)

    # Print the loss
    print(loss)
