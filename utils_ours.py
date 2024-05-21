import numpy as np
from scipy.ndimage import gaussian_filter
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import radon

def build_blurring_operator(size, sig_blurr):
    # builds radon transform operator into matrix of suitable shape
    # args:
    # im_size: pixels of quadratic image
    # sig_blurr: standard deviation of gaussian blurring
    
    A = np.zeros((size**2, size**2))

    for i in range(size):
        for j in range(size):
            x = np.zeros((size, size))
            x[i, j] = 1
            y = gaussian_filter(x, sigma=sig_blurr)
            A[: , i*size + j] = y.flatten()    
    return torch.from_numpy(A).float()

def build_CT_operator(size, angles, max_angle):
   # builds radon transform operator into matrix of suitable shape
   # args:
   # size: pixels of quadratic image
   # angles: number of projections
   # max_angle: maximum angle at which projectios are drawn. Usuall less than 180 degrees in our case to model limited angle tomography
    A = np.zeros((size*angles, size**2))
    for i in range(size):
        for j in range(size):
            x = np.zeros((size, size))
            x[i, j] = 1
            theta = np.linspace(0., max_angle, angles, endpoint=False)
            y = radon(x, theta=theta)
            A[: , i*size + j] = y.flatten()   
    _, normaliz, _ = np.linalg.svd(A)
    A = A / normaliz[0]
    #for large dimensional problems save the forward map to avoid recomputation
    torch.save(A, "CT_forward" + str(size) + "angles" + str(angles) + "max_angle" + str(max_angle) +".pt")
    return A


def uncond_loss_fn(model, x0, T, eps=1e-5):
#The loss function for training score-based generative models.
#   Args:
#    
#    model:   A PyTorch model instance that represents a 
#             time-dependent score-based model.
#    x0:      A mini-batch of training data.    
#    T:       final time of forward SDE
#    eps:     A tolerance value for numerical stability.
    
    #Force the same time for all the batch; TODO allow different time
    #random_t = torch.rand(x0.shape[0], device=x0.device) * (T - eps) + eps
    random_t = torch.rand(x0.shape[0]) * (T - eps) + eps
    
    z = torch.randn_like(x0)
    z = z.reshape((z.shape[0], z.shape[1], z.shape[2] * z.shape[3]))

    std = torch.sqrt(1 - torch.exp(-random_t))
    noise = z * std[:, None, None]

    x_t = torch.exp(-random_t/2)[:, None, None, None] * x0 + noise.reshape(x0.shape)
    score = model(x_t, random_t)

    loss = torch.mean(torch.sum((score * std[:, None, None, None].to(score.device) + z.reshape(x0.shape).to(score.device))**2, dim = (1, 2, 3)))
    return loss, random_t.to(loss.device)

def our_loss_fn(model, x0, A, Gam, T, eps=1e-5):
#The loss function for training score-based generative models.
#   Args:
#    
#    model:   A PyTorch model instance that represents a 
#             time-dependent score-based model.
#    x0:      A mini-batch of training data.    
#    A:       Forward operator
#    Gam:     observational noise
#    T:       final time of forward SDE
#    eps:     A tolerance value for numerical stability.
    
    #Force the same time for all the batch; TODO allow different time
    random_t = torch.rand(1, device = A.device) * (T - eps) + eps

    z = torch.randn_like(x0).to(A.device)
    z = z.reshape((z.shape[0], z.shape[1], z.shape[2] * z.shape[3]))

    d = A.shape[1]
    sig_post2_inv = 1 / (torch.exp(random_t) - 1) * torch.eye(d, device = A.device) + A.T @ torch.linalg.inv(Gam).to(A.device) @ A
    std_inv = torch.linalg.cholesky(sig_post2_inv).float()

    ## transforming where to train accordingly ##
    x_t_dash = x0.reshape(z.shape).to(A.device) @ sig_post2_inv.float() + z @ std_inv
    x_t_dash = x_t_dash.reshape(x0.shape)
    score = model(x_t_dash, random_t.repeat(x0.shape[0]))
    target = x0 

    loss = torch.mean(torch.sum((score - target.to(score.device))**2, dim = (1, 2, 3))) 
    return loss, random_t.to(loss.device)

