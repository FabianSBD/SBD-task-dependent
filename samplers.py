import torch
import numpy as np
import math
from tqdm import tqdm

def Plain_Euler_Maruyama_sampler(score_model, 
                           T,
                           nsamples,
                           batch_size, 
                           num_steps, 
                           device, 
                           size,
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
  Args:
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, size, size, color_channels, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
    
      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)):
          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
            
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step
          f = betas[i]
          g = math.sqrt(betas[i])
    
          std = torch.sqrt(1 - torch.exp(-int_beta))
          score = score_model(x, batch_time_step) / std
            
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f * x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))


def LD_sampler(score_model, 
                           T, 
                           y,
                           A,
                           sig_obs,
                           nsamples,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, color_channels, size * size, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
    
      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)):
          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step
          f = betas[i]
          g = math.sqrt(betas[i])
    
    
          std = torch.sqrt(1 - torch.exp(-int_beta))
          score = score_model(x.reshape(batch_size, size, size, color_channels), batch_time_step) / std
          score = score.reshape(x.shape)
            
          yhat = A @ x[:, 0, :].T
          ald = (y[:, 0, :] - yhat.T) @ A / sig_obs**2
          score = score + ald[:, None, :] 
            
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f * x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(mean_x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))


def ALD_sampler(score_model, 
                           T, 
                           y,
                           A,
                           sig_obs,
                           nsamples,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
  Args:
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
    
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, color_channels, size * size, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
    
      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)):
          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step
          f = betas[i]
          g = math.sqrt(betas[i])
    
          std = torch.sqrt(1 - torch.exp(-int_beta))
          score = score_model(x.reshape(batch_size, size, size, color_channels), batch_time_step) / std
          score = score.reshape(x.shape)
            
          yhat = A @ x[:, 0, :].T
          ald = (y[:, 0, :] - yhat.T) @ A / sig_obs**2
          score = score + ald[:, None, :] * torch.norm(score) / torch.norm(ald)  #https://arxiv.org/pdf/2304.11751
            
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f*x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(mean_x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))

def DPM_sampler(score_model, 
                           T, 
                           y,
                           A,
                           sig_obs,
                           p,
                           nsamples,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
  Args:
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, color_channels, size * size, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x

      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      for param in score_model.parameters():
        param.requires_grad = False
          
      for (i, time_step) in enumerate(tqdm(time_steps)):
        int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
        batch_time_step = torch.ones(batch_size, device=x.device) * time_step
        
        f = betas[i]
        g = math.sqrt(betas[i])
    
        x = x.detach().clone().requires_grad_(True)
        std = torch.sqrt(1 - torch.exp(-int_beta))
        score = score_model(x.reshape(batch_size, size, size, color_channels), batch_time_step) / std
        score = score.reshape(x.shape)
    
        x0hat = torch.exp(int_beta/2) * (x + (1-torch.exp(-int_beta)) * score)
        loss = torch.sum(((x0hat @ A.T - y))**2)
        loss.backward()
        score = score - p * x.grad / (torch.sum(((x0hat @ A.T - y))**2, axis = (1, 2))[:, None, None])**1/2

          
        diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
        drift_change = (f*x/2. + score * g**2) * step_size
        mean_x = x + drift_change 
        x = mean_x + diff_change
      samples.append(mean_x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))

def Projection_sampler(score_model,
                           lamb,
                           T, 
                           y,
                           A,
                           sig_obs,
                           nsamples,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  Source: https://arxiv.org/pdf/2404.00471
  
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  samples = [] 
  lamb = 1 - lamb / lamb
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, color_channels, size * size, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x

      #too expensive and requires Batch_size 1
      #oper = A.T @ A
      #oper.mul_(lamb * oper)
      #oper[range(oper.size(0)), range(oper.size(0))] += (1 - lamb)

      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)):
          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step
          f = betas[i]
          g = math.sqrt(betas[i])
        
          yt = torch.exp(-int_beta/2) * y + torch.sqrt(1 - torch.exp(-int_beta)) * torch.randn_like(x) @ A.T
        
          #too expensive and requires Batch_size 1
          #b = (1 - lamb) * x + lamb * yt @ A
          #x = torch.linalg.solve(
          #  oper.T,
          #  b,
          #  left=False
          #)
          xt = x
          for i in range(10):
              x = x - ((x @ A.T - yt)  @ A + (x - xt))
    
          std = torch.sqrt(1 - torch.exp(-int_beta)) 
          score = score_model(x.reshape(batch_size, size, size, color_channels), batch_time_step) / std
          score = score.reshape(x.shape)
          
            
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f*x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))


def cond_sampler(score_model, 
                           T, 
                           y,
                           A,
                           sig_obs,
                           nsamples,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0
      init_x = torch.randn(batch_size, size, size, color_channels, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
      
      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)):
          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
          
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step
          f = betas[i]
          g = math.sqrt(betas[i])
    
          std = torch.sqrt(1 - torch.exp(-int_beta))
          score = score_model(x, y, batch_time_step) / std
          
            
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f * x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(mean_x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))



def Our_sampler(score_model, 
                           T,
                           y,
                           A,
                           sig_obs,
                           nsamples,
                           batch_size,
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  
  Args:
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    y:                 evidence term
    nsamples:          Number of samples to be generated
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    size:              pixels of quadratic image
    color_channels:    should be set to 1
    eps:               The smallest time step for numerical stability.
    
    Assumes prior model of Ornstein Uhlenbeck process
    
  
  Returns:
    Samples.    
  """
  samples = [] 
  for _ in torch.utils.data.DataLoader(range(nsamples),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True):
      t = T * torch.ones(batch_size, device=device)
      #mean for init x is 0 
      init_x = torch.randn(batch_size, color_channels, size * size, device=device) 
      time_steps = torch.linspace(T, eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
         
      A_Gam_inv_y = y @ A / sig_obs**2

      bmax = 10
      bmin = 0.05
      betas = bmin + (time_steps / T) * (bmax - bmin)
      with torch.no_grad():
        for (i, time_step) in enumerate(tqdm(time_steps)): 
          batch_time_step = torch.ones(batch_size, device=x.device) * time_step

          int_beta = time_step * bmin + (bmax-bmin) * time_step **2 / (2*T)
          scale = torch.clamp((torch.exp(int_beta) - 1), max = 1.)

          f = betas[i]
          g = math.sqrt(betas[i])
          lamb = 1 / (torch.exp(int_beta/2) - torch.exp(-int_beta/2))
          m_t = A_Gam_inv_y + x * lamb
          m_t = m_t.reshape(batch_size, size, size, color_channels) * scale
            
          score = score_model(m_t, batch_time_step).reshape(x.shape)
          score = (score - torch.exp(int_beta/2) * x) * lamb
    
          diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
          drift_change = (f* x/2. + score * g**2) * step_size
          mean_x = x + drift_change 
          x = mean_x + diff_change
      samples.append(mean_x) 
  samples = torch.cat(samples, dim=0)
  return samples.reshape((nsamples, color_channels, size, size))


