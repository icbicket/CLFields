import numpy as np
import coord_transforms as coord
from dataclasses import dataclass

@dataclass(frozen=True)
class ParabolicMirror:
    """Class for keeping track of mirror parameters."""
    a: float
    dfoc: float
    xcut: float
    thetacutoffhole: float

AMOLF_MIRROR = ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75, thetacutoffhole=4.)

def ar_mask_calc(theta, phi, holein=True, slit=None, slit_center=0, orientation=0, mirror=AMOLF_MIRROR):
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
    
    phi = phi + orientation
    #    a = mirror.a
    #    xcut = 10.75
    #    ##thetacutoff can actually be calculated
    #    holesize = 0.6
    #    holeheight = np.sqrt(2.5/mirror.a)
    ##    thetacutoffhole=np.arctan(holesize/(2*holeheight))*180/np.pi
    #    thetacutoffhole = 4
    #    dfoc = 0.5
##    phi,theta=np.meshgrid(phi1,theta1) ##Toon
#    c = np.empty(np.shape(phi))
#    c_denominator = a*np.cos(phi)*np.sin(theta) + a
#    c[c_denominator==0] = np.inf
#    c[c_denominator!=0] = 1/(2*c_denominator[c_denominator!=0])
##    c = 1./(2*(a*np.cos(phi)*np.sin(theta)+a))

#    z = np.cos(theta)*c
#    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
#    y = np.sin(theta)*np.sin(phi)*c
    x, y, z, c = mirror_xyz(theta, phi, mirror)
    condition = (x < mirror.xcut) | (z < mirror.dfoc)
#    print('x', x, '\ny', y, '\nz', z)
    if slit is not None:
        ycut_positive = slit_center + slit/2.  ##
        ycut_negative = slit_center - slit/2.  ##
        condition = (condition | (y > ycut_positive)) ##
        condition = (condition | (y < ycut_negative))  ##
    else:
        pass

    if holein is True:
        condition = (condition | (theta <= (mirror.thetacutoffhole*np.pi/180)))
    else:
        pass

    return condition

def mirror_xyz(theta, phi, mirror=AMOLF_MIRROR):
    c = np.empty(np.shape(phi))
    c_denominator = mirror.a*np.cos(phi)*np.sin(theta) + mirror.a
    c[c_denominator==0] = np.inf
    c[c_denominator!=0] = 1/(2*c_denominator[c_denominator!=0])
#    spherical_vector = 
    z = np.cos(theta)*c
    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
    y = np.sin(theta)*np.sin(phi)*c
    return x, y, z, c

def degree_of_polarization(S0, S1, S2, S3):
    '''
    Calculate the total degree of polarization, degree of linear polarization, 
     degree of circular polarization, and the ellipticity
    '''
    if np.any(S0 < 0):
        raise ValueError(f'Element {np.where(S0<0)} is less than 0. S0 should be >= 0.')
    S0_check = S1**2 + S2**2 + S3**2
    if np.any(S0**2 < S0_check*0.99):
        raise ValueError(f'{np.where(S0**2 < S0_check*0.99)} is not valid input')
    DoP = np.zeros(np.shape(S0))
    DoLP = np.zeros(np.shape(S0))
    DoCP = np.zeros(np.shape(S0))
    ellipticity = np.zeros(np.shape(S0))
    DoP[S0 != 0] = np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0]) + np.square(S3[S0 != 0]))/S0[S0 != 0]
    DoLP[S0 != 0] = np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0]))/S0[S0 != 0]
    DoCP[S0 != 0] = np.abs(S3[S0 != 0]/S0[S0 != 0])
    # ellipticity from Bass 2010, Handbook of Optics
    ellipticity[S0 != 0] = S3[S0 != 0]/(S0[S0 != 0] + np.sqrt(np.square(S1[S0 != 0]) + np.square(S2[S0 != 0])))
    return DoP, DoLP, DoCP, ellipticity

def mirror_mask3d(theta, phi, **kwargs):
    '''
    Given theta and phi, return a mask representing the mirror in 3D
    Add two axes at the end
    '''
    mirror = ar_mask_calc(theta, phi, **kwargs)
    mirror = np.expand_dims(mirror, axis=-1)
    mirror_3 = np.expand_dims(mirror, axis=-1)
    mirror_3 = np.repeat(mirror_3, 3, axis=-2)
    mirror_3 = np.repeat(mirror_3, 3, axis=-1)
    return mirror_3

def mirror_outline(phi=np.linspace(0, 2*np.pi, 1000), holein=True, slit=None, slit_center=None, orientation=0, mirror=AMOLF_MIRROR):
    '''
    provide the coordinates for an outline of the mirror and slit combination
    Returns an n by 2 array of (theta, phi) values for the edges of the mirror,
    and an n by 2 array of (theta, phi) values for the edges of the hole, if in
    Won't work if the slit is too far off the centre of the mirror (such that 
    it doesn't cover the x-axis): one side of the slit must be on positive y,
    the other on negative y
    '''
    # hole
    if holein:
        hole_theta = 4*np.ones(np.shape(phi))
#        hole_phi = np.linspace(0, 2*np.pi, n)
        hole_phi = phi + orientation
        hole_theta_phi = np.transpose(np.array([hole_theta, hole_phi]))
    else:
        hole_theta_phi = np.array([[None, None]])
    
    x = np.empty(np.shape(phi))
    y = np.empty(np.shape(phi))
    z = np.empty(np.shape(phi))
    
    if slit is None:
        corner1_y = np.sqrt(25-mirror.xcut*10-mirror.dfoc**2)
        _, _, corner1_phi = coord.cartesian_to_spherical_coords(np.array([[mirror.xcut, corner1_y, 1]]))
        corner2_y = -np.sqrt(25-mirror.xcut*10-mirror.dfoc**2)
        _, _, corner2_phi = coord.cartesian_to_spherical_coords(np.array([[mirror.xcut, corner2_y, 1]]))
        
        # zcut/dfoc plane
        z[phi<=corner1_phi] = mirror.dfoc
        z[phi>=corner2_phi] = mirror.dfoc
        y[phi<=corner1_phi] = (
            (10/np.tan(phi[phi<=corner1_phi]) 
            - np.sqrt(np.square(10/np.tan(phi[phi<=corner1_phi])) 
            - 4 * -1 * (25-z[phi<=corner1_phi]**2)))/(2*-1)
            )
        y[phi>=corner1_phi] = (
            (10/np.tan(phi[phi>=corner1_phi]) 
            + np.sqrt(np.square(10/np.tan(phi[phi>=corner1_phi])) 
            - 4 * -1 * (25-z[phi>=corner1_phi]**2)))/(2*-1)
            )
        y[phi==0] = 0
        y[phi == np.pi] = 0
        y[phi == 2*np.pi] = 0
        x[phi<=corner1_phi] = 2.5-(y[phi<=corner1_phi]**2 + z[phi<=corner1_phi]**2)/10
        x[phi>=corner1_phi] = 2.5-(y[phi>=corner1_phi]**2 + z[phi>=corner1_phi]**2)/10
        # xcut plane
        edge_1_2 = np.logical_and((phi>corner1_phi), (phi<corner2_phi))
        x[edge_1_2] = mirror.xcut
        y[edge_1_2] = x[edge_1_2]*np.tan(phi[edge_1_2])
        z[edge_1_2] = np.sqrt(25 - 10*x[edge_1_2] - y[edge_1_2]**2)
    else:
        ycut_positive = slit_center + slit/2
        ycut_negative = slit_center - slit/2
        corner1_x = 2.5 - (ycut_positive**2 + mirror.dfoc**2)/10
        _, _, corner1_phi = coord.cartesian_to_spherical_coords(
            np.array([[corner1_x, ycut_positive, mirror.dfoc]])
            )
        _, _, corner2_phi = coord.cartesian_to_spherical_coords(
            np.array([[mirror.xcut, ycut_positive, mirror.dfoc]])
            )
        _, _, corner3_phi = coord.cartesian_to_spherical_coords(
            np.array([[mirror.xcut, ycut_negative, mirror.dfoc]])
            )
        corner4_x = 2.5 - (ycut_negative**2 + mirror.dfoc**2)/10
        _, _, corner4_phi = coord.cartesian_to_spherical_coords(
            np.array([[corner4_x, ycut_negative, mirror.dfoc]])
            )
        # zcut/dfoc plane
        edge_0_1 = phi<=corner1_phi
        z[edge_0_1] = mirror.dfoc
        y[edge_0_1] = (
            (10/np.tan(phi[edge_0_1]) 
            - np.sqrt(np.square(10/np.tan(phi[edge_0_1])) 
            - 4 * -1 * (25-z[edge_0_1]**2)))/(2*-1)
            )
        y[phi==0] = 0
        y[phi == np.pi] = 0
        y[phi == 2*np.pi] = 0
        x[edge_0_1] = 2.5-(y[edge_0_1]**2 + z[edge_0_1]**2)/10
        
        edge_1_2 = np.logical_and(phi>corner1_phi, phi<corner2_phi)
        y[edge_1_2] = ycut_positive
        x[edge_1_2] = y[edge_1_2]/np.tan(phi[edge_1_2])
        z[edge_1_2] = np.sqrt(25 - 10*x[edge_1_2] - y[edge_1_2]**2)
        
        edge_2_3 = np.logical_and(phi>=corner2_phi, phi<=corner3_phi)
        x[edge_2_3] = mirror.xcut
        y[edge_2_3] = x[edge_2_3]*np.tan(phi[edge_2_3])
        z[edge_2_3] = np.sqrt(25 - 10*x[edge_2_3] - y[edge_2_3]**2)
        
        edge_3_4 = np.logical_and(phi>=corner3_phi, phi<=corner4_phi)
        y[edge_3_4] = ycut_negative
        x[edge_3_4] = y[edge_3_4]/np.tan(phi[edge_3_4])
        z[edge_3_4] = np.sqrt(25 - 10*x[edge_3_4] - y[edge_3_4]**2)
        
        edge_4_0 = phi>corner4_phi
        z[edge_4_0] = mirror.dfoc
        y[edge_4_0] = (
            (10/np.tan(phi[edge_4_0]) 
            + np.sqrt(np.square(10/np.tan(phi[edge_4_0])) 
            - 4 * -1 * (25-z[edge_4_0]**2)))/(2*-1)
            )
        y[phi==0] = 0
        y[phi == np.pi] = 0
        y[phi == 2*np.pi] = 0
        x[edge_4_0] = 2.5-(y[edge_4_0]**2 + z[edge_4_0]**2)/10
    
##    mirror_phi = np.linspace(0, 2*np.pi, n)
#    z = np.ones(np.shape(phi))*mirror.dfoc
#    y = np.zeros(np.shape(phi))
#    y1 = (10/np.tan(phi) - np.sqrt(np.square(10/np.tan(phi)) - 4 * -1 * (25-z**2)))/(2*-1)
#    y[phi<np.pi] = y1[phi < np.pi]
#    y[phi==0] = 0
#    y[phi == np.pi] = 0
#    y2 = (10/np.tan(phi) + np.sqrt(np.square(10/np.tan(phi)) - 4 * -1 * (25-z**2)))/(2*-1)
#    y[phi > np.pi] = y2[phi > np.pi]
#    y[phi == 2*np.pi] = 0
#    x = y / np.tan(phi)
#    x[phi == 0] = 2.5-mirror.dfoc**2/10
#    x[phi == 2*np.pi] = 2.5-mirror.dfoc**2/10
#    x[phi == np.pi] = mirror.xcut
#    x_condition = x<= mirror.xcut
#    x[x_condition] = mirror.xcut
#    y[x_condition] = x[x_condition] * np.tan(phi[x_condition])
#    z[x_condition] = np.sqrt((2.5-x[x_condition])*10-y[x_condition]**2)
#    if slit is not None:
#        y_condition_positive = y > slit/2
#        y_condition_negative = y < slit/2
#        y[y_condition_positive] = slit/2
#        y[y_condition_negative] = -slit/2
#        x[y_condition_positive] = y[y_condition_positive]/np.tan(phi[y_condition_positive])
#        x[y_condition_positive * x_condition] = -10.75
#        x[y_condition_negative] = y[y_condition_negative]/np.tan(phi[y_condition_negative])
#        x[y_condition_negative * x_condition] = -10.75
#        z[y_condition_positive] = np.sqrt((2.5-x[y_condition_positive])*10-y[y_condition_positive]**2)
#        z[y_condition_negative] = np.sqrt((2.5-x[y_condition_negative])*10-y[y_condition_negative]**2)
    vector = np.transpose(np.array([x, y, z]))
    _, theta, _ = coord.cartesian_to_spherical_coords(vector)
    phi = phi + orientation
    mirror_theta_phi = np.transpose(np.array([theta, phi]))
#    #outsides of mirror
#    theta_patch = np.linspace(np.radians(14), np.pi/2, n)
#    phi_patch = np.linspace(0, 2*np.pi, n)
#    theta_mesh, phi_mesh = np.meshgrid(theta_patch, phi_patch)
#    mirror4patch = np.invert(ar_mask_calc(
#        theta_mesh, 
#        phi_mesh, 
#        holein=holein, 
#        slit=slit, 
#        slit_center=slit_center, 
#        orientation=orientation))
#    theta_mesh *= mirror4patch
#    phi_mesh *= mirror4patch
#    
#    th_sort = np.argsort(theta_mesh.flatten())[::-1]
#    
#    diff_phi, diff_ind = np.unique(
#        np.degrees(phi_mesh.flatten()[th_sort]).astype(int), 
#        return_index=True,
#        )
#    
#    th_max = theta_mesh.flatten()[th_sort][diff_ind]
#    ph_max = phi_mesh.flatten()[th_sort][diff_ind]
#    maxthph = np.transpose(np.array([ph_max, th_max]))
    return mirror_theta_phi, hole_theta_phi

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
