import torch
import numpy as np
from scipy.integrate import solve_ivp

def setup_cuda():
    """Return CPU device"""
    print("Using CPU")
    return torch.device("cpu")

def batch_hill_function(x, T, n, device=None):
    """Vectorized Hill function computation on CPU"""
    x = torch.tensor(x, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)
    n = torch.tensor(n, dtype=torch.float32)
    
    return (x**n / (T**n + x**n)).numpy()

def batch_integrate_system(system, initial_conditions, t_span, device=None):
    """Batch integrate multiple initial conditions in parallel on CPU"""
    x0 = torch.tensor(initial_conditions, dtype=torch.float32)
    
    dt = 0.01
    t = torch.arange(t_span[0], t_span[1], dt)
    
    trajectories = torch.zeros((len(initial_conditions), len(t), 2))
    trajectories[:, 0] = x0
    
    for i in range(1, len(t)):
        dx = system(trajectories[:, i-1])
        trajectories[:, i] = trajectories[:, i-1] + dt * dx
    
    return trajectories.numpy()

def compute_nullclines_gpu(system, x_range, y_range, n_points=100, device=None):
    """Compute nullclines using CPU"""
    x = torch.linspace(x_range[0], x_range[1], n_points)
    y = torch.linspace(y_range[0], y_range[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        derivatives = system(points)
    
    dx = derivatives[:, 0].reshape(n_points, n_points)
    dy = derivatives[:, 1].reshape(n_points, n_points)
    
    return (X.numpy(), Y.numpy(), 
            dx.numpy(), dy.numpy())
    
def torch_jacobian(system, point, device):
    """Compute Jacobian using automatic differentiation on CPU"""
    x = torch.tensor(point, dtype=torch.float32, requires_grad=True)
    y = system(x)
    y = y.view(-1)
    jacobian = torch.zeros((len(y), len(x)), dtype=torch.float32)
    for i in range(len(y)):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1.0
        y.backward(grad_outputs, retain_graph=True)
        jacobian[i] = x.grad.clone()
        x.grad.zero_()
    
    return jacobian

def integrate_system_rk45(system, x0, t_span, device, rtol=1e-4, atol=1e-6):
    """Integrate ODE system using solve_ivp (RK45) on CPU"""
    def numpy_system(t, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        dx_tensor = system(x_tensor)
        return dx_tensor.numpy()
    
    sol = solve_ivp(
        numpy_system,
        t_span,
        x0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=0.1
    )
    
    final_deriv = np.linalg.norm(numpy_system(sol.t[-1], sol.y[:, -1]))
    has_converged = final_deriv < 1e-3
    
    return sol.y[:, -1], has_converged 