'''
rcome.manifold.tools
    define common tools for manifolds
'''
@staticmethod
def gyration(u, v, w, k, dim=-1):
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