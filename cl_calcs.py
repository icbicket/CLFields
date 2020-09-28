import numpy as np

def ar_mask_calc(theta, phi, holein=True, slit=None, slit_center=0, orientation=0):
    '''
    Define a mask representing the mirror used in SEM-CL system at AMOLF
    x and y axes are exchanged relative to the normal spherical coordinate convention (is it normal?)
    mirror opening points along -x direction
    input: 
        theta and phi are angles of emission
        holein: True or False boolean, whether or not to take out the electron beam hole from the mirror mask
        slit: None or a  positive integer or float value, if None, there is no slit, if there is a value, this is the width of the slit in mm
        orientation: a value in radians, to rotate the mirror around the sample (eg to get different responses to simulate rotating the stage in the microscope to change the orientation of the mirror relative to the sample)
    
    solve equality ar^2-1/(4a)=x. 
    c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^
    2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Then
    we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    square root of a^2
    '''
    
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
        ycut_positive = slit_center + slit/2.
        ycut_negative = slit_center - slit/2.
        condition = (condition | (y > ycut_positive))
        condition = (condition | (y < ycut_negative))
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

def mirror_outline(holein=True, slit=3, slit_center=0, orientation=0 ):
    '''
    provide the coordinates for an outline of the mirror and slit combination
    '''
    # hole
    r = 4*np.ones((1000))
    phi = np.linspace(0, 2*np.pi, 1000)
    r_phi = np.transpose(np.array([phi, r]))
    
    #outsides of mirror
    theta_patch = np.linspace(np.radians(14), np.pi/2, 1000)
    phi_patch = np.linspace(0, 2*np.pi, 1000)
    theta_mesh, phi_mesh = np.meshgrid(theta_patch, phi_patch)

    mirror4patch = np.invert(ar_mask_calc(theta_mesh, phi_mesh, holein=holein, slit=slit, slit_center=slit_center, orientation=orientation))
    theta_mesh *= mirror4patch
    phi_mesh *= mirror4patch
    
    th_sort = np.argsort(theta_mesh.flatten())[::-1]
    
    diff_phi, diff_ind = np.unique(
        np.degrees(phi_mesh.flatten()[th_sort]).astype(int), 
        return_index=True,
        )
    
    th_max = theta_mesh.flatten()[th_sort][diff_ind]
    ph_max = phi_mesh.flatten()[th_sort][diff_ind]
    maxthph = np.transpose(np.array([ph_max, np.degrees(th_max)]))
    return maxthph, r_phi

def stokes_parameters(E_theta, E_phi):
    S0 = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    S1 = np.real(E_theta * np.conj(E_theta) - E_phi * np.conj(E_phi))
    S2 = np.real((E_theta * np.conj(E_phi)) + (E_phi * np.conj(E_theta)))
#    S2 = -2*np.real(np.real(E_theta) * np.real(E_phi) - np.imag(E_theta)*np.imag(E_phi))
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
