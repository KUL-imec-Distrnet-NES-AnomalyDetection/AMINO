import torch
import torch.nn as nn


# inspired by:
# https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/model/models.py
# Orignal paper:
# https://arxiv.org/abs/1807.03748
class CPCloss(nn.Module):
    def __init__(self, enc_dimension, ar_dimension, predict_size):
        super().__init__()
        self.gru = nn.GRU(
            enc_dimension, 
            ar_dimension, 
            batch_first=True
        )
        self.Wk = nn.ModuleList([
            nn.Linear(ar_dimension, enc_dimension) for _ in range(predict_size)
        ])
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.predict_size = predict_size
        self.enc_dimension = enc_dimension

    def forward(self, feature, dummy=None):
        target = feature[:,-self.predict_size:,:].transpose(0,1) # target is (W2,B,D)
        forward_sequence = feature[:,:-self.predict_size,:] # forward_sequence is (B,W1,D)

        output = self.gru(forward_sequence)[0]
        context = output[:,-1,:].view(feature.size(0), -1) # context is (B,D) (take last hidden state)
        pred = torch.empty(
            (self.predict_size, feature.size(0), self.enc_dimension),
            dtype = torch.float,
            device = feature.device,
        ) # pred (empty container) is (W2,B,D)
        # loop of prediction
        for i in range(self.predict_size):
            pred[i] = self.Wk[i](context) # Wk*context is (B,D)
        loss = self.info_nce(pred, target)
        # return loss, accuracy
        return loss.mean()
    
    def info_nce(self, prediction, target):
        k_size, batch_size, hidden_size = target.shape
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target.device)
        # compute nce
        logits = torch.matmul(
            prediction.reshape([-1, hidden_size]), 
            target.reshape([-1, hidden_size]).transpose(-1, -2)
        )
        loss = self.ce_loss(logits, label)
        # process for split loss and accuracy into k pieces (useful for logging)
        nce = list()
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size+batch_size
            nce.append(torch.sum(loss[start:end]) / batch_size)
        return torch.stack(nce).unsqueeze(0)


class GMMLoss(nn.Module):
    def __init__(self, n_gmm, weight_energy=0.7, weight_cov=0.3):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_cov = weight_cov
        self.n_gmm = n_gmm
        self.eps = 1e-9

    def forward(self, z, gamma):
        """Computing the loss function for DAGMM."""
        sample_energy, cov_diag = self.compute_energy(z, gamma)
        loss = self.weight_energy * sample_energy + self.weight_cov * cov_diag
        return loss

    def compute_energy(self, z, gamma):
        """Computing the sample energy function"""
        phi, mu, cov = self.compute_params(z, gamma)
        z_mu = (z.unsqueeze(-2) - mu)

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)).to(device=z.device) * self.eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k * (2 *torch.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=-1) + self.eps)
        E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        #  z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=[0, 1, 2], keepdim=True) / (gamma.size(0)*gamma.size(1)*gamma.size(2))

        # mu = KxD
        mu = torch.sum(z.unsqueeze(-2) * gamma.unsqueeze(-1), dim=[0, 1, 2], keepdim=True)
        mu /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1)

        a = z.unsqueeze(-2)
        z_mu = (z.unsqueeze(-2) - mu) # z:B,C,T,H (combine dimension), mu:(1,1,1,H,)
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        a = gamma.unsqueeze(-1).unsqueeze(-1)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=[0, 1, 2])
        cov /= torch.sum(gamma, dim=[0, 1, 2]).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        #l = torch.cholesky(a, False)
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    '''
       L = torch.cholesky(A)
       should be replaced with
       L = torch.linalg.cholesky(A)
       and
       U = torch.cholesky(A, upper=True)
       should be replaced with
       U = torch.linalg.cholesky(A).transpose(-2, -1).conj().
       '''

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - l.data.new(l.size(1)).fill_(0.5).diag())
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

if __name__ == "__main__":
    cpc = CPCloss(512, 33, 5)

    # batch, seq, feature
    feature = torch.randn(16, 750, 512)
    loss = cpc(feature)
    print(loss)
