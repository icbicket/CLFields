import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib as mpl

def norm_colours(values, lims=(-1,1)):
    # Normalize colour values to between 0 and 1
    maxcol = np.max(np.abs(values))
    if lims==(-1, 1):
        colours = (values/maxcol + 1)/2
        col_min = -np.max(np.abs(values))
        col_max = np.max(np.abs(values))
    else:
        colours = values/maxcol
        col_min = 0
        col_max = np.max(np.abs(colours))
    print(colours)
    return colours, col_min, col_max

def plot_3d_fields(mesh1, mesh2, colour1, colour2, fig_name, cmap=cm.seismic, col_scale=(-1, 1)):
    '''
    mesh1: vectors defining the faces of a 3d particle (n by 3 (or 4) by 3 for n faces with 3 (or 4) vertices with (x,y,z) coordinates for each)
    mesh2: vectors defining the faces of a second 3d particle (n by 3 (or 4) by 3 for n faces with 3 (or 4) vertices with (x,y,z) coordinates for each)
    colour1: 1d vector of colour/intensity values for mapping to faces mesh1 (length n), or valid matplotlib colour value
    colour2: 1d vector of colour/intensity values for mapping to faces of mesh2 (length n), or valid matplotlib colour value
    fig_name: string to name the figure
    cmap: matplotlib colourmap instance
    col_scale: the relevant scaling of the colourmap (-1 to 1 for blue-red, 0 to 1 for dark-light)
    '''
    
    # set up the figure and 3d axis
    fig = plt.figure(fig_name)
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    #add the two meshes as 3d surface plots
    poly1 = mplot3d.art3d.Poly3DCollection(verts=mesh1, alpha=1)
    poly2 = mplot3d.art3d.Poly3DCollection(verts=mesh2, alpha=1)
    ax.add_collection3d(poly1)
    ax.add_collection3d(poly2)
    
    # Scale and position the 3d axes
    scale_max = np.max(np.append(mesh1.flatten(), mesh2.flatten()))
    scale_min = np.min(np.append(mesh1.flatten(), mesh2.flatten()))
    scale = [scale_min, scale_max]
    ax.set_position([0, 0.05, 0.8, 0.9])
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Set facecolours
    data_plus, col_min, col_max = norm_colours(colour1, lims=col_scale)
    norm = mpl.colors.Normalize(vmin=col_min, vmax=col_max)    
    poly1.set_facecolors(cmap(data_plus))
    poly2.set_facecolors(cmap(data_plus))
    
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

    mpl.colorbar.ColorbarBase(ax=cbar_ax, cmap=cmap, norm=norm)
    
    #show plot and return figure
    plt.show()
    return fig
