import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
import matplotlib.colors as plt_colors
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle, PathPatch
import numpy as np
import torch 
from rcome.function_tools import poincare_function as pf
from rcome.function_tools import euclidean_function as ef
from rcome.function_tools import distribution_function as df
import math
import os

from rcome.manifold.poincare_ball import PoincareBallExact


def plot_poincare_gmm(z, gmm, labels=None, n_estim=100, marker='.', 
                      marker_size=20, save_folder=".", file_name="default.png",
                      close=True):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    X = np.linspace(-1, 1 ,n_estim)
    Y = np.linspace(-1, 1 ,n_estim)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.zeros((n_estim, n_estim))

    X0, Y0, radius = 0, 0, 1
    r = np.sqrt((X - X0)**2 + (Y * Y0)**2)
    disc = r < 1
    for z_index in range(len(Z)):

        point =  torch.cat((torch.FloatTensor(X[z_index]).unsqueeze(-1), torch.FloatTensor(Y[z_index]).unsqueeze(-1)), -1)
        # print(point.shape)
        p = gmm.get_density(point)
        # print(p.shape)
        p[p != p ]= 0
        Z[z_index] = p.numpy()  

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=plt.get_cmap("viridis"))    
    z_circle = -0.8
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z = z_circle, zdir="z")
    if labels is not None:
        n_cluster = len(np.unique([labels[i] for i in range(len(labels))]))
        
    for q in range(len(z)):
        c_color = [plt_colors.hsv_to_rgb([float(labels[q][0])/(n_cluster),0.5,0.8])] if(labels is not None) else "C0"
        ax.scatter(z[q][0].item(), z[q][1].item(), z_circle , c=c_color, marker=marker, s=marker_size)    
    
    mu = gmm._mu
    for j in range(len(mu)):
        ax.scatter(mu[j][0].item(), mu[j][1].item(), z_circle, c='r', marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')
    filepath = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(filepath, format="png")
    if(close):
        plt.close()

def plot_poincare_disc_embeddings(z, labels=None, centroids=None, save_folder=".", file_name="default.png",
                    marker='.', s=2000., draw_circle=True, axis=False, geodesics=None, close=True):
    fig = plt.figure(" Embeddings ", figsize=(20, 20))
    fig.patch.set_visible(False)

    # draw circle
    theta = np.linspace(0, 2*np.pi, 100)

    r = np.sqrt(1.0)

    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    if(draw_circle):
        plt.plot(x1, x2)

    if labels is not None:
        n_cluster = len(np.unique([labels[i] for i in range(len(labels))]))
    # print(n_cluster)
    # plotting embeddings
    for q in range(len(z)):
        # print(float(labels[q][0]))
        c_color = [plt_colors.hsv_to_rgb([float(labels[q][0])/(n_cluster),0.5,0.8])] if(labels is not None) else "C0"
        plt.scatter(z[q][0].item(), z[q][1].item(), c=c_color, marker=marker, s=s)    
    if(centroids is not None):
        plt.scatter(centroids[:, 0], centroids[:,1],marker='D', s=800., c='red')      
    if(geodesics is not None):
        for x in geodesics:
            plt.plot(x[:,0].numpy(), x[:,1].numpy(), linewidth=3, c="C1")
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, file_name)
    if(not axis):
        plt.axis('off')
    plt.savefig(filepath, format="png")
    if(close):
        plt.close()

def plot_geodesic(from_point, to_point, manifold, ax=None):
    factors = torch.arange(1e-3, 1-1e-3, 1e-2)


    points = []
    for f in factors:
        points.append(manifold.riemannian_exp(from_point, f *  manifold.riemannian_log(from_point, to_point)))
    points = torch.cat(points)
    if(ax is None):
        plt.plot(points[:, 0], points[:, 1], c='red')
    else:
        ax.plot(points[:, 0], points[:, 1], c='red')
    return points


### TESTING METHODS ###
def test_geodesics():

    manifold = PoincareBallExact
    from_point = torch.Tensor([[0.7, 0.1]])
    to_point = torch.Tensor([[0.1, 0.4]])
    plot_poincare_disc_embeddings(torch.cat((from_point, to_point),0), close=False)
    points = plot_geodesic(from_point, to_point, manifold)
    print(points)
    plt.savefig("LOG/geodesic.png")


# test_geodesics()