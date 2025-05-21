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
    # size: pixels of quadratic image
    # sig_blurr: standard deviation of gaussian blurring
    try:
        A = torch.load("Blurr_forward" + str(size) + "sig:" + str(sig_blurr) +".pt")
        return A
         
    except:
        A = np.zeros((size**2, size**2))
    
        for i in range(size):
            for j in range(size):
                x = np.zeros((size, size))
                x[i, j] = 1
                y = gaussian_filter(x, sigma=sig_blurr)
                A[: , i*size + j] = y.flatten() 
        A = torch.from_numpy(A).float()
        torch.save(A, "Blurr_forward" + str(size) + "sig:" + str(sig_blurr) +".pt", pickle_protocol=4)
        return A


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
    A = A / size
    #for large dimensional problems save the forward map to avoid recomputation
    torch.save(A, "CT_forward" + str(size) + "angles" + str(angles) + "max_angle" + str(max_angle) +".pt", pickle_protocol=4)
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
    random_t = torch.rand(x0.shape[0], device = x0.device) * (T - eps) + eps
    
    z = torch.randn_like(x0)
    int_beta = random_t * 0.05 + (10-0.05) * random_t**2 / (2*T)

    std = torch.sqrt(1 - torch.exp(-int_beta))
    noise = z * std[:, None, None, None]

    x_t = torch.exp(-int_beta/2)[:, None, None, None] * x0 + noise
    score = model(x_t, random_t)

    loss = torch.mean(torch.sum((score + z.reshape(x0.shape).to(score.device))**2, dim = (1, 2, 3)))
    return loss

def cond_loss_fn(model, x0, A, sig_obs, T, d3 = False, eps=1e-5):
#The loss function for training score-based generative models.
#   Args:
#    
#    model:   A PyTorch model instance that represents a 
#             time-dependent score-based model.
#    x0:      A mini-batch of training data.    
#    T:       final time of forward SDE
#    eps:     A tolerance value for numerical stability.
    origshape = x0.shape
    random_t = torch.rand(x0.shape[0], device = x0.device) * (T - eps) + eps
    
    z = torch.randn_like(x0)
    x0 = x0.permute(1, 2, 0, 3)
    prev_shape = x0.shape
    x0 = x0.reshape((-1, x0.shape[2] * x0.shape[3]))
    y = A @ x0
    y = y.reshape(prev_shape)
    y = y.permute(2, 0, 1, 3)

    x0 = x0.reshape((origshape[1], origshape[2], origshape[0], origshape[3]))
    x0 = x0.permute(2, 0, 1, 3)
    y = y + sig_obs * torch.randn_like(y)


    int_beta = random_t * 0.05 + (10-0.05) * random_t**2 / (2*T)
    
    std = torch.sqrt(1 - torch.exp(-int_beta))
    noise = z * std[:, None, None, None]

    x_t = torch.exp(-int_beta/2)[:, None, None, None] * x0 + noise#.reshape(x0.shape)
    if d3:
        y = y.reshape(x0.shape)
    score = model(x_t, y, random_t)

    loss = torch.mean(torch.sum((score + z.reshape(x0.shape).to(score.device))**2, dim = (1, 2, 3)))
    return loss

def our_loss_fn(model, x0, A, sig_obs, T, eps=1e-5):
#The loss function for training score-based generative models.
#   Args:
#    
#    model:   A PyTorch model instance that represents a 
#             time-dependent score-based model.
#    x0:      A mini-batch of training data.    
#    A:       Forward operator
#    sig_obs:     observational noise; scalar
#    T:       final time of forward SDE
#    eps:     A tolerance value for numerical stability.
    origshape = x0.shape
    x00 = x0
    random_t = torch.rand(x0.shape[0], device = A.device) * (T - eps) + eps
    bmax = 10
    bmin = 0.05
    
    int_beta = random_t * bmin + (bmax-bmin) * random_t**2 / (2*T) 
    scale = torch.clamp((torch.exp(int_beta) - 1), max = 1.)

    x0 = x0.permute(0, 3, 1, 2)
    x0 = x0.reshape(x0.shape[0], x0.shape[1], -1)
    noise = (1 / torch.sqrt(torch.exp(int_beta) - 1))[:, None, None] * torch.randn_like(x0).to(A.device) 
    noise = noise + torch.randn(size = (x0.shape[0], x0.shape[1], A.shape[0]), device = A.device) @ A / sig_obs
    # transforming where to train accordingly ##

    
    x_t_dash = x0.to(A.device) / (torch.exp(int_beta) - 1)[:, None, None] + x0.to(A.device) @ A.T @ A / sig_obs**2
    x_t_dash = x_t_dash + noise

    x_t_dash = x_t_dash.permute(0, 2, 1)
    x_t_dash = x_t_dash.reshape(origshape).to(x0.device) 

    x_t_dash = x_t_dash / (torch.std(x_t_dash, dim = (1, 2, 3), keepdim = True) + 1)
    score = model(x_t_dash, random_t)

    target = x00
    loss = torch.mean(torch.sum((score - target.to(score.device))**2, dim = (1, 2, 3))) 
    return loss


class LungDataset(Dataset):

    def __init__(self, root_dir = "/fabian/work/Project CT Diffusion/csgm/csgm-main/lungdata", train = True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.train = train
        if train:
            self.root_dir = os.path.join(root_dir, 'train512')
        else:
            self.root_dir = os.path.join(root_dir, 'test512')
        self.file_names = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_names[idx])
        image = torch.load(img_name) 

        if self.transform:
            sample = self.transform(image)

        return (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))


