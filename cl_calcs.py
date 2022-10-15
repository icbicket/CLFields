import numpy as np
import coord_transforms as coord

import constants
import scipy.interpolate as scint

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

def angle_of_incidence(incident_vector, normal):
    '''
    calculate the angle of incidence of 'incident_vector' onto the surface 
    with a surface normal described by 'normal'
    tested under the assumption that the incident vector and normal vector are Nx3
    '''
    incident_vector_magnitude = coord.field_magnitude(incident_vector, keepdims=True)
    normal_magnitude = coord.field_magnitude(normal, keepdims=True)
    if np.any(incident_vector_magnitude)==0:
        raise ValueError('Magnitude of the incident vector cannot be 0')
    elif np.any(normal_magnitude)==0:
        raise ValueError('Magnitude of the normal vector cannot be 0')
    cosine_angle = np.array((np.sum(incident_vector * normal, axis=-1, keepdims=True))
        /(incident_vector_magnitude * normal_magnitude))
    float_error_condition = np.logical_and(
        (abs(cosine_angle)-1) < 1e-5,
        (abs(cosine_angle)-1) > 0)
    if np.any(float_error_condition):
        cosine_angle[float_error_condition] = np.round(
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
    n_surface = np.real(n_surface)
    n_environment = np.real(n_environment)
    brewster = np.array(np.arctan(n_surface/n_environment))
    brewster[n_surface==n_environment] = np.nan
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
    if isinstance(n_surface, complex) or isinstance(n_environment, complex):
        r_s = r_s.astype(complex)
        r_p = r_p.astype(complex)
    r_s[incidence_angle==0] = (
        (n_environment - n_surface)/(n_surface + n_environment)
        )
    r_p[incidence_angle==0] = r_s[incidence_angle==0]
    r_s[incidence_angle!=0] = (
                np.sin(refraction_angle[incidence_angle!=0]
                - incidence_angle[incidence_angle!=0])
            ) / (
                np.sin(refraction_angle[incidence_angle!=0]
                + incidence_angle[incidence_angle!=0])
            )
    r_p[incidence_angle!=0] = (
                np.sin(2*refraction_angle[incidence_angle!=0]) 
                - np.sin(2*incidence_angle[incidence_angle!=0])
            ) / (
                np.sin(2*incidence_angle[incidence_angle!=0])
                + np.sin(2*refraction_angle[incidence_angle!=0])
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
    s_direction = s_direction/np.expand_dims(coord.field_magnitude(s_direction, keepdims=True), axis=-1)
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

def dielectric_to_refractive(dielectric):
    '''
    Convert a complex dielectric function to complex refractive index
    '''
    index_factor = np.sqrt(load_data[:, 1]**2 + load_data[:, 2]**2)
    n = (np.sqrt(0.5 * (index_factor + load_data[:, 1])) 
        + 1j * np.sqrt(0.5 * (index_factor - load_data[:, 1])))
    return n

def eV_to_wavelength(eV):
    '''
    Convert photon energy in eV to wavelength in m
    '''
    wavelength = constants.PLANCK * constants.LIGHTSPEED / (eV * constants.COULOMB)
    return wavelength

def wavelength_to_eV(wavelength):
    '''
    Convert photon wavelength in m to eV
    '''
    eV = constants.PLANCK * constants.LIGHTSPEED / (wavelength * constants.COULOMB)
    return eV

def interpolate_refractive_index(refractive_index, wavelength_list, desired_wavelength):
    '''
    wavelength: wavelength at which it is desired to extract the refractive index (in m)
    filename: file containing the dielectric function (should point to al_pa.mat)
    returns complex refractive index at the given wavelength
    '''
    interp_function_n = scint.interp1d(wavelength_list, refractive_index, kind='cubic')
    n_wavelength = interp_function_n(desired_wavelength)
    return n_wavelength

