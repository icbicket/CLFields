import numpy as np

def ar_mask_calc(theta, phi, holein=True, slit=None, orientation=0):
    '''
    Define a mask representing the mirror used in SEM-CL system at AMOLF
    x and y axes are exchanged relative to the normal spherical coordinate convention (is it normal?)
    mirror opening points along -x direction
    input: 
        theta and phi are angles of emission
        holein: True or False boolean, whether or not to take out the electron beam hole from the mirror mask
        slit: None or a  positive integer or float value, if None, there is no slit, if there is a value, this is the width of the slit in mm
        orientation: a value in radians, to rotate the mirror around the sample (eg to get different responses to simulate rotating the stage in the microscope to change the orientation of the mirror relative to the sample)
    '''
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
    
    return np.logical_not(mask)

def degree_of_polarization(S0, S1, S2, S3):
    DoP = np.zeros(np.shape(S0))
    DoLP = np.zeros(np.shape(S0))
    DoCP = np.zeros(np.shape(S0))
    ellipticity = np.zeros(np.shape(S0))
    DoP[S0 != 0] = np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0]) + np.square(S3[S0 != 0]))/S0[S0 != 0]
    DoLP[S0 != 0] = np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0]))/S0[S0 != 0]
    DoCP[S0 != 0] = S3[S0 != 0]/S0[S0 != 0]
    ellipticity[S0 != 0] = S3[S0 != 0]/(S0[S0 != 0] + np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0])))
    return DoP, DoLP, DoCP, ellipticity

def mirror_mask3d(theta, phi, **kwargs):
    mirror = ar_mask_calc(theta, phi, **kwargs)
    mirror = np.expand_dims(mirror, axis=1)
    mirror_3 = np.expand_dims(mirror, axis=1)
    mirror_3 = np.repeat(mirror_3, 3, axis=1)
    mirror_3 = np.repeat(mirror_3, 3, axis=2)
    return mirror_3

def stokes_parameters(E_theta, E_phi):
    S0 = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    S1 = np.real(E_theta * np.conj(E_theta) - E_phi * np.conj(E_phi))
    S2 = np.real(-(E_theta * np.conj(E_phi)) - (E_phi * np.conj(E_theta)))
#    S3 = np.real(-1j * (E_phi * np.conj(E_theta) - 1j * E_theta * np.conj(E_phi)))
    S3 = np.real(-1j * (E_theta * np.conj(E_phi) - E_phi * np.conj(E_theta)))
#    S2 = 2 * np.real(E_theta * np.conj(E_phi))
#    S3 = -2 * np.imag(E_theta * np.conj(E_phi))
    return S0, S1, S2, S3

def normalize_stokes_parameters(S0, S1, S2, S3):
    s1 = np.zeros(np.shape(S1))
    s2 = np.zeros(np.shape(S2))
    s3 = np.zeros(np.shape(S3))
    s1[S0 != 0] = S1[S0 != 0]/S0[S0 != 0]
    s2[S0 != 0] = S2[S0 != 0]/S0[S0 != 0]
    s3[S0 != 0] = S3[S0 != 0]/S0[S0 != 0]
    return s1, s2, s3
