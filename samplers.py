import torch
import numpy as np

def Plain_Euler_Maruyama_sampler(score_model, 
                           T,
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
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=x.device) * time_step
      g = 1

      score = score_model(x.reshape(batch_size, color_channels, size, size), batch_time_step)
      score = score.reshape(x.shape)
        
      diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
      drift_change = (x/2. + score * g**2) * step_size
      mean_x = x + drift_change 
      x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))


def LD_sampler(score_model, 
                           T, 
                           y,
                           A,
                           Gam_inv,
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
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=x.device) * time_step
      g = 1

      score = score_model(x.reshape(batch_size, color_channels, size, size), batch_time_step)
      score = score.reshape(x.shape)
        
      yhat = A @ x[:, 0, :].T
      ald = (y[:, 0, :] - yhat.T) @ Gam_inv @ A
      score = score + ald[:, None, :] 
        
      diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
      drift_change = (x/2. + score * g**2) * step_size
      mean_x = x + drift_change 
      x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))


def ALD_sampler(score_model, 
                           T, 
                           y,
                           A,
                           Gam_inv,
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
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=x.device) * time_step
      g = 1

      score = score_model(x.reshape(batch_size, color_channels, size, size), batch_time_step)
      score = score.reshape(x.shape)
        
      yhat = A @ x[:, 0, :].T
      ald = (y[:, 0, :] - yhat.T) @ Gam_inv @ A
      score = score + ald[:, None, :] * torch.norm(score) / torch.norm(ald)  #https://arxiv.org/pdf/2304.11751
        
      diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
      drift_change = (x/2. + score * g**2) * step_size
      mean_x = x + drift_change 
      x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))

def DPM_sampler(score_model, 
                           T, 
                           y,
                           A,
                           Gam_inv,
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
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  #xi = 1
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  #with torch.no_grad():
  for param in score_model.parameters():
    param.requires_grad = False
      
  for time_step in time_steps:
    batch_time_step = torch.ones(batch_size, device=x.device) * time_step
    g = 1

    x = x.detach().clone().requires_grad_(True)

    score = score_model(x.reshape(batch_size, color_channels, size, size), batch_time_step)
    score = score.reshape(x.shape)

    x0hat = torch.exp(time_step/2) * (x + (1-torch.exp(-time_step)) * score)
    #loss = torch.sum(((x0hat @ A.T - y))**2)
    loss = torch.sum(((x0hat @ A.T - y) @ torch.linalg.cholesky(Gam_inv))**2)
    loss.backward()
    #print(x.grad)
    #normali = torch.sum(((x0hat @ A.T - y))**2, axis = (1, 2))[:, None, None].expand(batch_size, color_channels, A.shape[1])
    #score = score - xi * x.grad / torch.sqrt(normali)
    score = score - x.grad
      
    diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
    drift_change = (x/2. + score * g**2) * step_size
    mean_x = x + drift_change 
    x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))

def Projection_sampler(score_model,
                           lamb,
                           T, 
                           y,
                           A,
                           Gam_inv,
                           batch_size, 
                           size,
                           num_steps, 
                           device, 
                           color_channels,
                           eps):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  Source: https://arxiv.org/pdf/2404.00471
  
  Args:
    score_model:       A PyTorch model that represents the time-dependent score-based model. Corresponds to the model for 
                       the score of p_0 \conv \tilde k_t 
    T:                 final time such that the reverse SDE starts at X_T
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model of Ornstein Uhlenbeck process
  
  Returns:
    Samples.    
  """
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  #oper = torch.linalg.inv(lamb * A.T @ A + (1-lamb) * torch.eye(size**2, device = A.device))
  oper_inv = lamb * A.T @ A + (1-lamb) * torch.eye(size**2, device = A.device)
  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=x.device) * time_step
      g = 1
    
      yt = torch.exp(-time_step/2) * y + torch.sqrt(1 - torch.exp(-time_step)) * torch.randn_like(x) @ A.T
      #x = T_inv @ (lamb * [y, 0] + (1- lamb) * [A @ x, 0] + [0, A_rest @ x] ) 
      x = torch.linalg.solve(oper_inv.T, (1 - lamb) * x + lamb * yt @ A, left = False)

      score = score_model(x.reshape(batch_size, color_channels, size, size), batch_time_step)
      score = score.reshape(x.shape)
      
        
      diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
      drift_change = (x/2. + score * g**2) * step_size
      mean_x = x + drift_change 
      x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))

def Posterior_Euler_Maruyama_sampler(score_model, 
                           T,
                           y,
                           A,
                           Gam_inv,
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
    y:                 noisy observation
    A:                 forward operator
    Gam_inv:           Inverse noise covariance
    batch_size:        The number of samplers to generate by calling this function once.
    num_steps:         The number of sampling steps. 
                       Equivalent to the number of discretized time steps.
    device:            'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps:               The smallest time step for numerical stability.
    
    Assumes posterior model dX_t = dW_t 
    
    I am making use of the equality A x = (x.T A.T).T, whenever my x is batched, eg x = (64, 784), A = (28, 784)
    then A @ x does not work, but x @ A.T works
  
  Returns:
    Samples.    
  """
  t = T * torch.ones(batch_size, device=device)
  #mean for init x is 0
  init_x = np.sqrt((1-np.exp(-T))) * torch.randn(batch_size, color_channels, size * size, device=device) 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  x = init_x
     
  A_Gam_inv_y = y @ Gam_inv.T @ A

  with torch.no_grad():
    for time_ind in range(num_steps): 
      time_step = time_steps[time_ind]
    
      if time_ind > 1:
          step_size = time_steps[time_ind - 1 ] - time_steps[time_ind]
      else:
          step_size = time_steps[time_ind] - time_steps[time_ind + 1]
      g = 1
     
      score = score_wrapper(x, time_step, A_Gam_inv_y, score_model, batch_size, color_channels, size)
      
      diff_change = torch.sqrt(step_size) * torch.randn_like(x) * g
      drift_change = (x/2. + score * g**2) * step_size
      mean_x = x + drift_change 
      x = mean_x + diff_change
  return mean_x.reshape((batch_size, color_channels, size, size))

def score_wrapper(x, time_step, A_Gam_inv_y, score_model, batch_size, color_channels, size):
    batch_time_step = torch.ones(batch_size, device=x.device) * time_step

    m_t = A_Gam_inv_y + torch.exp(time_step/2) / (torch.exp(time_step) - 1) * x
    m_t = m_t.reshape(batch_size, color_channels, size, size)
    
    score = score_model(m_t, batch_time_step).reshape((batch_size, color_channels, size**2))

    return (score - torch.exp(time_step/2) * x) / ((torch.exp(time_step/2)-torch.exp(-time_step/2)))
    

