import torch
import torch.nn.functional as F
from ckpt_analysis import load_ckpt


def gram_linear(X):
    """Compute linear kernel (Gram matrix)."""
    return X @ X.T

def gram_rbf(X, sigma=None):
    """Compute RBF (Gaussian) kernel (Gram matrix)."""
    GX = X @ X.T
    KX_diag = torch.diag(GX)
    sq_dists = KX_diag[:, None] - 2 * GX + KX_diag[None, :]
    if sigma is None:
        # Median heuristic for bandwidth
        sigma = torch.median(sq_dists[sq_dists != 0]).sqrt()
    K = torch.exp(-sq_dists / (2 * sigma ** 2))
    return K

def center_gram(G):
    """Center the Gram matrix."""
    n = G.size(0)
    I = torch.eye(n, device=G.device)
    one_n = torch.ones((n, n), device=G.device) / n
    return G - one_n @ G - G @ one_n + one_n @ G @ one_n

def cka(X, Y, kernel='linear'):
    """
    Compute CKA similarity between two matrices X and Y.
    X: [n, d1]
    Y: [n, d2]
    kernel: 'linear' or 'rbf'
    """
    assert X.size(0) == Y.size(0), "X and Y must have the same number of samples"
    
    if kernel == 'linear':
        K = gram_linear(X)
        L = gram_linear(Y)
    elif kernel == 'rbf':
        K = gram_rbf(X)
        L = gram_rbf(Y)
    else:
        raise ValueError("Unsupported kernel. Use 'linear' or 'rbf'.")

    Kc = center_gram(K)
    Lc = center_gram(L)

    hsic = (Kc * Lc).sum()
    norm_K = torch.norm(Kc)
    norm_L = torch.norm(Lc)

    return hsic / (norm_K * norm_L + 1e-10)  # add epsilon for stability


ckpt_p1 = "logs/convnext_s_fulltune/iter_160000.pth"
# ckpt_p2 = "logs/convnext_s_fulltune/iter_144000.pth"
ckpt_p2 = "pretrain/convnext_small_22k_1k_384.pth"

ckpt1, ckpt2 = load_ckpt(ckpt_p1), load_ckpt(ckpt_p2)

pwconv1_states, pwconv2_states = [], []
if "state_dict" in ckpt1:
    ckpt1 = ckpt1["state_dict"]
if "state_dict" in ckpt2:
    ckpt2 = ckpt2["state_dict"]

if "model" in ckpt1:
    ckpt1 = ckpt1["model"]
if "model" in ckpt2:
    ckpt2 = ckpt2["model"]

for name, param in ckpt1.items():
    if "pwconv1.weight" in name:
        p2 = ckpt2[name.split("backbone.")[-1]]
        # cka_val = cka(param, p2, kernel='rbf')
        cka_val = (p2 * param).mean()
        pwconv1_states.append((name.split(".")[-2], cka_val.item()))
    if "pwconv2.weight" in name:
        p2 = ckpt2[name.split("backbone.")[-1]]
        # cka_val = cka(param, p2, kernel='rbf')
        cka_val = (param * p2).mean()
        pwconv2_states.append((name.split(".")[-2], cka_val.item()))
print(pwconv1_states, pwconv2_states)



