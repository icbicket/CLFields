import numpy as np
import coord_transforms as coord

def ar_mask_calc(theta, phi, holein=True, slit=None, slit_center=0, orientation=0):
    '''
    Define a mask representing the mirror used in SEM-CL system at AMOLF
    x and y axes are exchanged relative to the normal spherical coordinate convention (is it normal?)
    mirror opening points along -x direction
    input: 
        theta and phi are angles of emission
        holein: True or False boolean, whether or not to take out the electron beam hole from the mirror mask
        slit: None or a positive integer or float value, if None, there is no slit, if there is a value, this is the width of the slit in mm
        orientation: a value in radians, to rotate the mirror around the sample (eg to get different responses to simulate rotating the stage in the microscope to change the orientation of the mirror relative to the sample)
    
    solve equality ar^2-1/(4a)=x. 
    c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^
    2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Then
    we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    square root of a^2
    True: outside mirror
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
##    phi,theta=np.meshgrid(phi1,theta1) ##Toon
#    c = np.empty(np.shape(phi))
#    c_denominator = a*np.cos(phi)*np.sin(theta) + a
#    c[c_denominator==0] = np.inf
#    c[c_denominator!=0] = 1/(2*c_denominator[c_denominator!=0])
##    c = 1./(2*(a*np.cos(phi)*np.sin(theta)+a))

#    z = np.cos(theta)*c
#    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
#    y = np.sin(theta)*np.sin(phi)*c
    x, y, z, c = mirror_xyz(theta, phi, a)
    condition = (-x > xcut) | (z < dfoc)
#    print('x', x, '\ny', y, '\nz', z)
    if slit is not None:
        ycut_positive = slit_center + slit/2.  ##
        ycut_negative = slit_center - slit/2.  ##
        condition = (condition | (y > ycut_positive)) ##
        condition = (condition | (y < ycut_negative))  ##
    else:
        pass

    if holein is True:
        condition = (condition | (theta <= (thetacutoffhole*np.pi/180)))
    else:
        pass

    return condition

def mirror_xyz(theta, phi, a=0.1):
    c = np.empty(np.shape(phi))
    c_denominator = a*np.cos(phi)*np.sin(theta) + a
    c[c_denominator==0] = np.inf
    c[c_denominator!=0] = 1/(2*c_denominator[c_denominator!=0])
#    spherical_vector = 
    z = np.cos(theta)*c
    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
    y = np.sin(theta)*np.sin(phi)*c
    return x, y, z, c

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

def mirror_mask3d(theta, phi, **kwargs): ##
    mirror = ar_mask_calc(theta, phi, **kwargs)
    mirror = np.expand_dims(mirror, axis=1)
    mirror_3 = np.expand_dims(mirror, axis=1)
    mirror_3 = np.repeat(mirror_3, 3, axis=1)
    mirror_3 = np.repeat(mirror_3, 3, axis=2)
    return mirror_3

def mirror_outline(holein=True, slit=3, slit_center=0, orientation=0 ): ##
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

def angle_of_incidence(incident_vector, normal):
    '''
    calculate the angle of incidence of 'incident_vector' onto the surface 
    with a surface normal described by 'normal'
    '''
    incident_vector_magnitude = coord.field_magnitude(incident_vector)
    normal_magnitude = coord.field_magnitude(normal)
    cosine_angle = np.array((np.sum(incident_vector * normal, axis=-1))
        /(incident_vector_magnitude * normal_magnitude))
    float_error_condition = np.logical_and(
        (abs(cosine_angle)-1) < 1e-5, 
        (abs(cosine_angle)-1)>0)
    if np.any(float_error_condition):
        cosine_angle[float_error_condition] = np.round(  ##
            cosine_angle[float_error_condition], 2)
    angle = np.array(np.arccos(cosine_angle))
    if np.any(cosine_angle < 0):
        angle[cosine_angle < 0] = np.pi-angle[cosine_angle < 0]
    return angle

def snells_law(incidence_angles, n_surface, n_environment=1):
    '''
    given the incident angle of light, calculate the angle of refraction
    incidence_angles: angle or angles of incidence in radians, single value or 
        1D numpy array
    n_surface: refractive index of the surface upon which light impinges
    n_environment: refractive index of the environment through which light 
        travels to impinge upon the surface
    '''
    angle_refraction = np.arcsin(n_environment/n_surface * np.sin(incidence_angles))
    return angle_refraction
    
def brewsters_angle(n_surface, n_environment=1):
    '''
    Calculate Brewster's angle given two refractive indices
    n_surface: refractive index of the medium that the light is hitting
    n_environment: refractive index of the medium the incident light travels
        through
    '''
    brewster = np.arctan(n_surface/n_environment)
    return brewster

def reflection_coefficients(incidence_angle, n_surface, n_environment=1):
    '''
    calculate the reflection coefficients for light incident at angles given
    by 'incidence_angles' from a medium with refractive index given by 
    'n_environment' onto the surface of a medium with refractive index given by 
    'n_surface'
    incidence_angle: the angle of incidence of light (in radians) onto the 
        surface, can be a single integer or float or a 1D numpy array
    n_surface: refractive index of the surface upon which light impinges
    n_environment: refractive index of the medium through which light travels
        before it impinges on the surface
    r_s: reflection coefficient for s-polarized light (perpendicular to the 
        plane of incidence)
    r_p: reflection coefficient for p-polarized light (parallel to the plane of
        incidence)
    Note: r_s**2 and r_p**2 produce the Fresnel reflection coefficients R_s, R_p
    '''
    if not isinstance(incidence_angle, np.ndarray):
        incidence_angle = np.array([incidence_angle])
    refraction_angle = snells_law(incidence_angle, n_surface, n_environment)
    r_s = np.empty(np.shape(incidence_angle))
    r_p = np.empty(np.shape(incidence_angle))
    r_s[incidence_angle==0] = (n_surface - n_environment)/(n_surface + n_environment)
    r_p[incidence_angle==0] = r_s[incidence_angle==0]
    r_s[incidence_angle!=0] = (
                -np.sin(incidence_angle[incidence_angle!=0] - refraction_angle[incidence_angle!=0])
            ) / (
                np.sin(incidence_angle[incidence_angle!=0] + refraction_angle[incidence_angle!=0])
            )
    r_p[incidence_angle!=0] = (
                np.sin(2*incidence_angle[incidence_angle!=0]) - np.sin(2*refraction_angle[incidence_angle!=0])
            ) / (
                np.sin(2*refraction_angle[incidence_angle!=0]) + np.sin(2*incidence_angle[incidence_angle!=0])
            )
#    if incidence_angle==0 and refraction_angle==0:
#        r_s = (n_surface - n_environment)/(n_surface + n_environment)
#        r_p = r_s
#    else:
#        r_s = (
#                -np.sin(incidence_angle - refraction_angle)
#            ) / (
#                np.sin(incidence_angle + refraction_angle)
#            )
#        r_p = (
#                np.sin(2*incidence_angle) - np.sin(2*refraction_angle)
#            ) / (
#                np.sin(2*refraction_angle) + np.sin(2*incidence_angle)
#            )
    #    r_p = (
    #            np.tan(incidence_angle - refraction_angle)
    #        ) / (
    #            np.tan(incidence_angle + refraction_angle)
    #        )
    return r_s, r_p

def reflected_e(incident_direction, incident_e, surface_normal, n_surface, n_environment=1):
    '''
    calculate the electric field reflected off an interface
    incident_direction: N by 3 numpy array representing the propagation 
        direction of the electric field vectors of interest (Cartesian)
    incident_e: N by 3 numpy array of electric field vectors to impinge on
        a surface (Cartesian)
    surface_normal: the normals of the surface that the electric field 
        vectors are hitting, either 1 by 3 or N by 3 numpy array
    n_surface: refractive index of the surface
    n_environment: refractive index of the medium which the rays originate
        from
    '''
    incident_angle = angle_of_incidence(incident_direction, surface_normal)
    r_s, r_p = reflection_coefficients(incident_angle, n_surface, n_environment)
    
    # isolate e_s (electric field polarized perpendicular to the plane of incidence)
    s_direction = np.cross(surface_normal, incident_direction)
    s_direction = s_direction/np.expand_dims(coord.field_magnitude(s_direction), axis=-1)
    e_s = np.sum(incident_e * s_direction, axis=-1, keepdims=True) * s_direction
    e_s_reflected = np.expand_dims(r_s, axis=-1) * e_s
    
    # isolate e_p (electric field polarized parallel to the plane of incidence)
    e_p = incident_e - e_s
    e_p_reflected = np.expand_dims(r_p, axis=-1) * e_p
    
#    e_p1 = incident_e - e_s
#    e_p_norm = e_p1/coord.field_magnitude(e_p1)
#    e_p_amp = np.sqrt(np.sum(e_p1*e_p1, axis=-1))
#    e_p = (2 * (np.sum(surface_normal*e_p_norm, axis=-1)) * surface_normal - e_p_norm) * e_p_amp
#    e_p_reflected = e_p * r_p
    
    return e_s_reflected, e_p_reflected

def stokes_parameters(E_theta, E_phi):
    S0 = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    S1 = np.real(E_theta * np.conj(E_theta) - E_phi * np.conj(E_phi))
    S2 = np.real((E_theta * np.conj(E_phi)) + (E_phi * np.conj(E_theta)))
#    S3 = np.real(-1j * (E_phi * np.conj(E_theta) - 1j * E_theta * np.conj(E_phi)))
    S3 = np.real(-1j * (E_theta * np.conj(E_phi) - E_phi * np.conj(E_theta)))
#    S2 = 2 * np.real(E_theta * np.conj(E_phi))
#    S3 = -2 * np.imag(E_theta * np.conj(E_phi))
    return S0, S1, S2, S3

def normalize_stokes_parameters(S0, S1, S2, S3):
    '''
    Return S1, S2, S3 normalized Stokes parameters
    '''
    s1 = np.zeros(np.shape(S1))
    s2 = np.zeros(np.shape(S2))
    s3 = np.zeros(np.shape(S3))
    s1[S0 != 0] = S1[S0 != 0]/S0[S0 != 0]
    s2[S0 != 0] = S2[S0 != 0]/S0[S0 != 0]
    s3[S0 != 0] = S3[S0 != 0]/S0[S0 != 0]
    return s1, s2, s3
