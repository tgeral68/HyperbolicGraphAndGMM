
def distance(x, y):
    return (x-y).norm(2,-1)

def invprod(x, y):
    return 1/(((x*y).sigmoid()).sum(-1))