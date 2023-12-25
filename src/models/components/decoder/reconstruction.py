import torch.nn.functional as F

def compute_reconstruction(x, x_hat):
    # cosine_similarity: (batch, time)
    cosine_similarity = F.cosine_similarity(x, x_hat, dim=-1)
    # relative_euclidean_distance: (batch, time)
    relative_euclidean_distance = (x - x_hat).norm(2, dim=-1) / x.norm(2, dim=-1)
    return relative_euclidean_distance, cosine_similarity
