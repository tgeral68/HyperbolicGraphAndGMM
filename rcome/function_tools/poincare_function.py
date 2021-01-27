''' Poincare ball model associated functions. 
Those function are deprecated in favor of manifold package functions.
'''

import math
import torch
from torch.autograd import Function
from rcome.function_tools import function


class PoincareDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        with torch.no_grad():
            x_norm = torch.clamp(torch.sum(x ** 2, dim=-1), 0, 1-5e-4)
            y_norm = torch.clamp(torch.sum(y ** 2, dim=-1), 0, 1-5e-4)
            d_norm = torch.sum((x-y) ** 2, dim=-1)
            cc = 1+2*d_norm/((1-x_norm)*(1-y_norm)) 
            dist = torch.log(cc + torch.sqrt(cc**2-1))
            ctx.save_for_backward(x, y, dist)
            return  dist
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            x, y, dist = ctx.saved_tensors
            dist_unsqueeze = dist.unsqueeze(-1).expand_as(x)

            res_x, res_y =  (- (log(x, y)/(dist_unsqueeze)) * grad_output.unsqueeze(-1).expand_as(x),
                     - (log(y, x)/(dist_unsqueeze)) * grad_output.unsqueeze(-1).expand_as(x))
            if((dist == 0).sum() != 0):
                # it exist example having same representation
                res_x[dist == 0 ] = 0 
                res_y[dist == 0 ] = 0
            return res_x, res_y 


def poincare_distance(x, y):
    return PoincareDistance.apply(x, y)


def distance(x, y):
    return PoincareDistance.apply(x, y)


def poincare_distance_radius(x, y, c=1):
    return (2/math.sqrt(c)) * torch.arc_tanh(math.sqrt(c)*add(-x, y).norm(2,-1))


def poincare_retractation(x, y):
    return ((1 - torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(y))**2) * y


def renorm_projection(x, eps=1e-4):
    x_n = x.norm(2, -1)
    if(len(x[x_n>=1.])>0):
        x[x_n>=1.] /= (x_n.unsqueeze(-1).expand_as(x[x_n>=1.]) + eps)
    return x


def add(x, y):
    nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x) * 1 
    ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x) * 1
    xy = (x * y).sum(-1, keepdim=True).expand_as(x)*1
    return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

def log(base_point, point):
    kpx = add(-base_point, point)
    norm_kpx = kpx.norm(2, -1, keepdim=True).expand_as(kpx)
    norm_k = base_point.norm(2, -1, keepdim=True).expand_as(kpx)
    res = (1-norm_k**2) * ((torch.arc_tanh(norm_kpx))) * (kpx/norm_kpx)
    if(0 != len(torch.nonzero(norm_kpx == 0))):
        res[norm_kpx == 0] = 0
    return res


def exp(base_point, vector):
    norm_k = base_point.norm(2, -1, keepdim=True).expand_as(base_point) * 1
    lambda_k = 1/(1-norm_k**2)
    norm_x = vector.norm(2, -1, keepdim=True).expand_as(vector) * 1
    direction = vector/norm_x
    factor = torch.tanh(lambda_k * norm_x)
    res = add(base_point, direction*factor)
    if(0 != len(torch.nonzero((norm_x == 0)))):
        res[norm_x == 0] = base_point[norm_x == 0]
    return res


def exp_0(x):
    norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
    direction = x/norm_x
    factor = torch.tanh(norm_x)
    return factor * direction


def lambda_k(k):
    return 2/(1-torch.sum(k ** 2, dim=-1, keepdim=True))


def _lambda(x, keepdim=False):
    return 2 / (1 - x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(1e-15)


def _gyration(u, v, w, k, dim=-1):
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    K2 = k ** 2
    a = -K2 * uw * v2 + k * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - k * uw
    d = 1 + 2 * k * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)



def parallel_transport(from_point, to_point,
                        vector):
    """Implementation of parallel transport for Poincare Ball.

    Transport a vector in the tangent point of from_point 
    to the tangent space of to_point.
    The code below come from geopt code
    """
    # to_point = -to_point
    # u2 = from_point.pow(2).sum(dim=-1, keepdim=True)
    # v2 = to_point.pow(2).sum(dim=-1, keepdim=True)
    # uv = (from_point * to_point).sum(dim=-1, keepdim=True)
    # uw = (from_point * vector).sum(dim=-1, keepdim=True)
    # vw = (to_point * vector).sum(dim=-1, keepdim=True)
    # a = -uw * v2 - vw + 2 * uv * vw
    # b = -vw * u2 + uw
    # d = 1 - 2 * uv + u2 * v2

    # transported_gyrovector = vector + 2 * (a * from_point + b* to_point)/d.clamp_min(1e-15)
    
    transported_gyrovector = _gyration(to_point, -from_point, vector, 1.)
    lambda_x = _lambda(from_point, keepdim=True)
    lambda_y = _lambda(to_point, keepdim=True)
    return transported_gyrovector * (lambda_x / lambda_y)




def norm(from_point, point):
    return _lambda(from_point, keepdim=True)**2\
        * (point**2).sum(-1, keepdim=True)


def test_gradient():
    import torch
    from torch import nn
    # testing gradient
    print("Unit gradient test")
    x = torch.Tensor([[0.6325,0.6325]])
    y = torch.Tensor([[-0.6325,-0.6325]])
    z = torch.Tensor([[-0.7,-0.7]])
    print("Norm")
    print("\t |x| ", x.norm(2,-1).item())
    print("\t |y| ", y.norm(2,-1).item())
    print("\t |z| ", z.norm(2,-1).item())

    print("d(x,y)=", distance(x, y).item())
    print("d(y,x)=", distance(y, x).item())
    xm = nn.Parameter(x)

    dxydx = (distance(xm,y)**2).backward()
    print("dxydx", xm.grad)
    print("x updated ", exp(xm.data, 0.1*-xm.grad))

    print("d(x,z)=", distance(x, z).item())
    print("d(z,x)=", distance(z, x).item())

    xm = nn.Parameter(x)
    dxzdx = (distance(xm,z)**2).backward()
    print("dxzdx", xm.grad)
    print("x updated ", exp(xm.data, 0.1 * -xm.grad))


if __name__ == "__main__":
    test_gradient()