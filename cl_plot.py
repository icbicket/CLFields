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
        colours = (values)/(2 * np.max(np.abs(values))) + 0.5
        abs_max = np.max(np.abs(values))
        col_min = -abs_max
        col_max = abs_max
    else:
        colours = (values - np.min(values))/(np.max(values)-np.min(values))
        col_min = np.min(values)
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

def ar_plot(theta, phi, magnitude, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = ax.get_figure()
#    scatterplot = ax.scatter(
#        phi, 
#        np.degrees(theta),
#        c = np.real(magnitude),
#        **kwargs)
    colourplot = ax.pcolormesh(
        phi,
        np.degrees(theta),
        np.real(magnitude),
        **kwargs
        )
    ax.set_rmax(90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    plt.colorbar(colourplot, ax=ax)
    return fig

def format_AR_plot(fig, axis):
    axis.set_position([0.05, 0.1, 0.85, 0.8])
    axis.set_facecolor('k')
    axis.tick_params(
        axis='y', 
        grid_color=[0.7, 0.7, 0.7], 
        labelsize=15, 
        grid_linewidth=1.5,
        pad=0
        )
    axis.tick_params(
        axis='x', 
        #colors='w', 
        grid_color=[0.7, 0.7, 0.7], 
        labelsize=15, 
        grid_linewidth=1.5, 
        pad=10
        )
    axis.spines['polar'].set_linewidth(2)
    axis.spines['polar'].set_color([0.7, 0.7, 0.7])
    linesr, labelsr = plt.rgrids((30, 60, 90), [None, None, None])
    cbar_axis = fig.axes[-1]
    cbar_axis.remove()
