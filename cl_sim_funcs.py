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



def degree_of_polarization(S0, S1, S2, S3):
    DoP = np.sqrt(np.square(S1) + np.square(S2) + np.square(S3))/S0
    DoLP = np.sqrt(np.square(S1) + np.square(S2))/S0
    DoCP = S3/S0
    ellipticity = S3/(S0 + np.sqrt(np.square(S1) + np.square(S2)))
    return DoP, DoLP, DoCP, ellipticity

def mirror_mask(theta, phi, slit=None, orientation=0):
    mirror = np.logical_not(ar_mask_calc(theta, phi, slit=slit, orientation=orientation))
    mirror = np.expand_dims(mirror, axis=1)
    mirror = np.zeros(np.shape(mirror)).astype('bool')
    mirror_3 = np.expand_dims(mirror, axis=1)
    mirror_3 = np.repeat(mirror_f, 3, axis=1)
    mirror_3 = np.repeat(mirror_f, 3, axis=2)
    return mirror, mirror_3

def stokes_parameters(E_theta, E_phi):
    S0 = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    S1 = np.real(E_theta * np.conj(E_theta) - E_phi * np.conj(E_phi))
    S2 = 2 * np.real(E_theta * np.conj(E_phi))
    S3 = -2 * np.imag(E_theta * np.conj(E_phi))
    return S0, S1, S2, S3
