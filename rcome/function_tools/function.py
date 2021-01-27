'''
This script contains general math usefull function 
'''
import torch 


def arc_cosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def arc_tanh(x):
    return 0.5 * torch.log((1+x)/(1-x))

def arc_sinh(x):
    x = torch.clamp(x, -2000, 20000)
    return torch.log(x +1e-4+ torch.sqrt(x**2 + 1))

torch.arc_cosh = arc_cosh
torch.arc_sinh = arc_sinh
torch.arc_tanh = arc_tanh
