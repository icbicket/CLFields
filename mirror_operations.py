import numpy as np
import coord_transforms as ct
import constants
import cl_calcs as clc
import scipy.interpolate as scint

magn = 2.16
a = 1/10 # paraboloid scaling factor
#a = 1/10*magn
xpixels = 1024;
ypixels = 1024;
pixelsize = 13*10**(-3)
pixelarea = pixelsize**2
xmax = 10.75/magn
focusdistance = 0.5/magn
xcut = 10.75/magn
ni = 1;



def parabola_position(direction):
    '''
    direction: emission direction vector in Cartesian coordinates, a numpy 
    array of shape (N by 3)
    '''
    direction = direction/ct.field_magnitude(direction, keepdims=True)
    r, theta, phi = ct.cartesian_to_spherical_coords(direction)
    c = np.empty(np.shape(direction))
    positive_x_condition = np.logical_not(
        np.logical_and(abs(theta-np.pi/2)<5e-8, abs(phi)<5e-8)
        )
    c[np.logical_not(positive_x_condition)] = np.nan
    c[positive_x_condition] = np.expand_dims(
            np.array(
                1 / (
                    2 * a * (
                        1-np.sin(theta[positive_x_condition])
                        * np.cos(phi[positive_x_condition])
                        )
                    )
                ), -1
        )
    parabola = direction * c
    return parabola

def parabola_normals(parabola_positions):
    '''
    parabola_positions: positions on the parabola, N by 3 numpy array
    R = (a*r**2, r*cos(theta), r*sin(theta))
    '''
#    r, theta = ct.cartesian_to_polar(parabola_positions[:, 1], parabola_positions[:, 2])

#    ## One way to calculate the normals    
#    normalizing_factor = 1/np.sqrt(1 + 4 * a**2 * r**2)
#    normal_x = np.ones(np.shape(r))
#    normal_y = -2 * a * r * np.cos(theta)
#    normal_z = -2 * a * r * np.sin(theta)
#    normals = np.transpose(normalizing_factor * np.vstack((normal_x, normal_y, normal_z)))
#    
    # parameterize x in terms of y and z: x = (a*(y**2 + z**2)-2.5, y, z)
    normalizing_factor = np.expand_dims(1/np.sqrt(
        1 + 4 * a**2 * (
            parabola_positions[:, 1]**2 + parabola_positions[:, 2]**2
            )
        ), axis=-1)
    normal_x = np.ones(np.shape(parabola_positions)[0])
    normal_y = -2 * a * parabola_positions[:, 1]
    normal_z = -2 * a * parabola_positions[:, 2]
    normals = normalizing_factor * np.transpose(
        np.vstack((normal_x, normal_y, normal_z))
        )
    
    ## Another way to calculate the normals
#    # derivative of parabola along r
#    r_gradient_x = 2*a*r
#    r_gradient_y = np.cos(theta)
#    r_gradient_z = np.sin(theta)
#    r_gradient = np.vstack((r_gradient_x, r_gradient_y, r_gradient_z)).transpose()
#    
#    # derivative of parabola along theta
#    theta_gradient_x = np.zeros(np.shape(r_gradient_x))
#    theta_gradient_y = -r * np.sin(theta)
#    theta_gradient_z = r * np.cos(theta)
#    theta_gradient = np.vstack((theta_gradient_x, theta_gradient_y, theta_gradient_z)).transpose()
#    
#    normals = np.cross(r_gradient, theta_gradient) #cross product of r and theta to get the surface normal
#    normals = normals/ct.field_magnitude(normals)[:, None] # normalize the normal vectors
    return normals

def surface_polarization_directions(theta, phi):
    '''
    calculate direction vectors for p- and s- polarized light on the mirror surface
    p-polarized: polarized along theta (is this true for the paraboloid??)
    s-polarized: polarized along phi (is this true for the paraboloid??)
    '''
    p_direction = np.transpose(np.vstack(ct.spherical_to_cartesian_vector_field(
        theta,
        phi,
        np.zeros(np.shape(theta)),
        np.ones(np.shape(theta)),
        np.zeros(np.shape(theta)),
        )))

    s_direction = np.transpose(np.vstack(ct.spherical_to_cartesian_vector_field(
        theta,
        phi,
        np.zeros(np.shape(theta)),
        np.zeros(np.shape(theta)),
        np.ones(np.shape(theta)),
        )))
    return p_direction, s_direction


def fresnel_reflection_coefficients(normal,
                                    e_incident_direction,
                                    n_mirror,
                                    n_environment=1):
    '''
    calculate s and p fresnel reflection coefficients for the parabolic mirror surface
    normals: surface normals of the mirror
    e_incident_direction: incident EM wave in Cartesian coordinates
    n_environment: refractive index of the environment at the desired wavelength
    n_mirror: refractive index of the mirror at the desired wavelength
    '''
    e_incident_direction = e_incident_direction/ct.field_magnitude(e_incident_direction, keepdims=True)
    normal = normal/ct.field_magnitude(normal, keepdims=True)
    incidence_angle = clc.angle_of_incidence(e_incident_direction, -normal)
#    incidence_angle = np.arccos(
#        np.dot(normals, e_incident_direction)
#        ) / (
#            ct.field_magnitude(normals)*ct.field_magnitude(e_incident_direction)
#        )
    if n_mirror == 0:
        raise ValueError("Mirror refractive index cannot be 0")
    n_factor = np.sqrt(1-np.square(n_environment/n_mirror * np.sin(incidence_angle)))
    r_s = np.square(
        (
            n_environment * np.cos(incidence_angle) -
            n_mirror * n_factor
        ) / (
            n_environment * np.cos(incidence_angle) +
            n_mirror * n_factor
            )
        )
    r_p = np.square(
        (
            n_environment * n_factor - 
            n_mirror * np.cos(incidence_angle)
        ) / (
            n_environment * n_factor + 
            n_mirror * np.cos(incidence_angle)
            )
        )
    return r_s, r_p

def import_al_pa(filepath):
    '''
    Import al_pa.mat given the filepath to this file
    - al_pa.mat contains the dielectric function for Aluminium, as given by Palik
    '''
    load_data = np.loadtxt(filepath, skiprows=15, max_rows=141-15)
    return load_data

def interpolate_refractive_index_at_wavelength(refractive_index, wavelength_list, desired_wavelength):
    '''
    
    '''
    pass

