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

def cond_loss_fn(model, x0, A, sig_obs, T, eps=1e-5):
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
    y = x0.reshape((x0.shape[0], -1)) @ A.T
    y = y.reshape((x0.shape[0], x0.shape[1], -1, 1))
    y = y + sig_obs * torch.randn_like(y)


    int_beta = random_t * 0.05 + (10-0.05) * random_t**2 / (2*T)
    
    std = torch.sqrt(1 - torch.exp(-int_beta))
    noise = z * std[:, None, None, None]

    x_t = torch.exp(-int_beta/2)[:, None, None, None] * x0 + noise#.reshape(x0.shape)
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
    x0 = x0.reshape(x0.shape[0], -1)
    random_t = torch.rand(x0.shape[0], device = A.device) * (T - eps) + eps
    bmax = 10
    bmin = 0.05
    
    int_beta = random_t * bmin + (bmax-bmin) * random_t**2 / (2*T) 
    scale = torch.clamp((torch.exp(int_beta) - 1)[:, None], max = 1.)
    
    noise = (1 / torch.sqrt(torch.exp(int_beta) - 1))[:, None] * torch.randn_like(x0).to(A.device) 
    noise = noise + torch.randn(size = (x0.shape[0], A.shape[0]), device = A.device) @ A / sig_obs
    # transforming where to train accordingly ##
    x_t_dash = x0.to(A.device) / (torch.exp(int_beta) - 1)[:, None] + x0.to(A.device) @ A.T @ A / sig_obs**2
    x_t_dash = x_t_dash + noise
    x_t_dash = x_t_dash * scale#* (torch.exp(int_beta) - 1)[:, None]
    x_t_dash = x_t_dash.reshape(origshape).to(x0.device) 

    score = model(x_t_dash, random_t)

    target = x0.reshape(origshape) 

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


class GP(Dataset):

    def __init__(self, size = 32, train = True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.train = train
        self.size = size

        def k(x, y, sig=32, l=size**2 / 10):
            return 1 / sig * torch.exp(-torch.norm(x - y)**2 / l)

        # Create a 2D grid of points
        t = torch.linspace(0, size-1, size)
        X, Y = torch.meshgrid(t, t)
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Initialize the covariance matrix
        K = 0.00001 * torch.eye(size**2)
        
        # Fill the covariance matrix based on the kernel
        for i in range(size**2):
            for j in range(size**2):
                K[i, j] = K[i, j] + k(grid[i], grid[j])
        
        # Cholesky decomposition
        self.K12 = torch.linalg.cholesky(K)

    def __len__(self):
        if self.train:
            return 100000
        else:
            return 1000

    def __getitem__(self, idx):
        sample = self.K12 @ torch.randn(size=(self.size**2, 1))

        # Reshape the result to a 2D grid
        sample = sample.reshape(self.size, self.size, 1)
        if self.transform:
            sample = self.transform(sample)

        return sample
