import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib as mpl

def ar_mask_calc(theta, phi, holein=True, slit=None, orientation=0):
    
    #solve equality ar^2-1/(4a)=x. 
    #c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^
    #2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Than
    #we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    #square root of a^2 
    
    a = 0.1
    xcut = 10.75
    phi = phi + orientation

    ##thetacutoff can actually be calculated
    holesize = 0.6
    holeheight = np.sqrt(2.5/a)
#    thetacutoffhole=np.arctan(holesize/(2*holeheight))*180/np.pi
    thetacutoffhole = 4
    dfoc = 0.5

#    phi,theta=np.meshgrid(phi1,theta1) ##Toon
    c = 1./(2*(a*np.cos(phi)*np.sin(theta)+a))

    z = np.cos(theta)*c
    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
    y = np.sin(theta)*np.sin(phi)*c
    
    condition = (-x > xcut) | (z < dfoc)
    
    if slit is not None:
        ycut = slit/2.
        condition = (condition | (np.abs(y) > ycut))
    else:
        pass

    if holein is True:
        condition = (condition | (theta < (thetacutoffhole*np.pi/180)))
    else:
        pass

    mask = np.ones(np.shape(phi))
    mask[condition] = False
        
    return mask  

def ar_plot(theta, phi, magnitude, lim=(0,1)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    scatterplot = ax.scatter(
        phi, 
        np.degrees(theta),
        c = np.real(magnitude),
        s=100,
        vmin=lim[0],
        vmax=lim[1])
    ax.set_rmax(90)
    ax.set_theta_zero_location('N')
    plt.colorbar(scatterplot, ax=ax)
    return fig

def cartesian_to_spherical_coords(vectors):
    r = np.sqrt(np.sum(np.square(vectors), axis=-1))
    theta = np.arctan(vectors[:, 1]/vectors[:, 2])
    phi = np.arccos(vectors[:, 0]/r)
    phi[(vectors[:, 1] > 0) * (vectors[:, 0] < 0)] += np.pi # +y, -z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] > 0)] += 2*np.pi #-y, +z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] < 0)] += np.pi # -y, -z
    return r, theta, phi

def cartesian_to_spherical_vector_field(theta, phi, fx, fy, fz):
    fr = (np.sin(theta) * np.cos(phi) * fx + 
       np.sin(theta) * np.sin(phi) * fy + 
       np.cos(theta) * fz)
    e_thz = (np.cos(theta) * np.cos(phi) * fx + 
        np.cos(theta) * np.sin(phi) * fy - 
        np.sin(theta) * fz)
    e_phz = (-np.sin(phi) * fx + 
        np.cos(phi) * fy)

def degree_of_polarization(S0, S1, S2, S3):
    DoP = np.sqrt(np.square(S1) + np.square(S2) + np.square(S3))/S0
    DoLP = np.sqrt(np.square(S1) + np.square(S2))/S0
    DoCP = S3/S0
    ellipticity = S3/(S0 + np.sqrt(np.square(S1) + np.square(S2)))
    return DoP, DoLP, DoCP, ellipticity

def expand_quadrant_symmetry(mag):
    Q1 = np.average(np.real(mag), axis=0)
    Q2 = np.flip(Q1, axis=1)[:, :-1]
    Q3 = np.flip(Q1, axis=0)[:-1, :]
    Q4 = np.flip(np.flip(Q1, axis=0), axis=1)[:-1, :-1]
    Q12 = np.append(Q2, Q1, axis=1)
    Q34 = np.append(Q4, Q3, axis=1)
    total = np.append(Q34, Q12, axis=0)
    return total

def field_magnitude(f):
    f_mag = np.sqrt(np.sum(f * np.conj(f), axis=-1))
    f_mag = np.real(f)
    return f_mag

def mirror_mask(theta, phi, slit=None, orientation=0):
    mirror = np.logical_not(ar_mask_calc(theta, phi, slit=slit, orientation=orientation))
    mirror = np.expand_dims(mirror, axis=1)
    mirror = np.zeros(np.shape(mirror)).astype('bool')
    mirror_3 = np.expand_dims(mirror, axis=1)
    mirror_3 = np.repeat(mirror_f, 3, axis=1)
    mirror_3 = np.repeat(mirror_f, 3, axis=2)
    return mirror, mirror_3

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

def stokes_parameters(E_theta, E_phi):
    S0 = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    S1 = np.real(E_theta * np.conj(E_theta) - E_phi * np.conj(E_phi))
    S2 = 2 * np.real(E_theta * np.conj(E_phi))
    S3 = -2 * np.imag(E_theta * np.conj(E_phi))
    return S0, S1, S2, S3
