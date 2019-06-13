import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib as mpl

def norm_colours(values, lims=(-1,1)):
    # Normalize colour values to between 0 and 1
    maxcol = np.max(np.abs(values))
    if type(lims) != tuple:
        raise ValueError("Colour limits were not a valid tuple")
    if lims==(-1, 1):
        colours = (values/maxcol + 1)/2
        abs_max = np.max(np.abs(values))
        col_min = -abs_max
        col_max = abs_max
    else:
        colours = values/maxcol
        col_min = 0
        col_max = np.max(np.abs(values))
    return colours, col_min, col_max

def plot_3d_fields(mesh, colour, ax=None, cmap=cm.seismic, col_scale=(-1,1)):
    # create figure if there is no axis provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
    
    # add mesh shape
    poly = mplot3d.art3d.Poly3DCollection(verts=mesh, alpha=1)
    ax.add_collection3d(poly)
    
    # scale axes to shape
    coords = mesh.flatten()
    scale_max = np.max(coords)
    scale_min = np.min(coords)
    scale = [scale_min, scale_max]
    ax.set_position([0, 0.05, 0.8, 0.9])
    ax.auto_scale_xyz(scale, scale, scale)
    
    # scale colours and set facecolours
    data_plus, col_min, col_max = norm_colours(colour, lims=col_scale)
    poly.set_facecolors(cmap(data_plus))

    # turn off axis markers
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    #set axis labels and positions
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #add colourbar
    cbar_ax = plt.axes([0.8, 0.1, 0.1, 0.8])
    norm = mpl.colors.Normalize(vmin=col_min, vmax=col_max)
    mpl.colorbar.ColorbarBase(ax=cbar_ax, cmap=cmap, norm=norm)
    
    #show plot and return figure
    plt.show()
    return fig

