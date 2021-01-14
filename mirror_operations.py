import numpy as np
import coord_transforms as ct
import constants
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
    direction: emission direction vector in Cartesian coordinates (N by M by 3) or (P by 3)
    '''
    direction = direction/ct.field_magnitude(direction[:, None])
    r, theta, phi = ct.cartesian_to_spherical_coords(direction)
    c = np.empty(np.size(theta))
    positive_x_condition = np.logical_and(theta==np.pi/2, phi==0)
    c[positive_x_condition] = np.nan
    c[np.logical_not(positive_x_condition)] = np.array(1/(2*a*(1-np.sin(theta)*np.cos(phi))))
#    c = 1/(2*(a*np.cos(phi)*np.sin(theta) + a))
#    c = np.array(1/(2*a*(1-np.sin(theta)*np.cos(phi))))
    parabola = direction * c[:, None]
    return parabola
    
def parabola_normals(parabola_x, parabola_y, parabola_z):
    '''
    parabola_x,y,z: position of parabola, 1D array
    R = (a*r**2, r*cos(theta), r*sin(theta))
    '''
    r, theta = ct.cartesian_to_polar(parabola_y, parabola_z)
    # derivative of parabola along r
    r_gradient_x = 2*a*r
    r_gradient_y = np.cos(theta)
    r_gradient_z = np.sin(theta)
    r_gradient = np.vstack((r_gradient_x, r_gradient_y, r_gradient_z)).transpose()
    
    # derivative of parabola along theta
    theta_gradient_x = np.zeros(np.shape(r_gradient_x))
    theta_gradient_y = -r * np.sin(theta)
    theta_gradient_z = r * np.cos(theta)
    theta_gradient = np.vstack((theta_gradient_x, theta_gradient_y, theta_gradient_z)).transpose()
    
    normals = np.cross(r_gradient, theta_gradient) #cross product of r and theta to get the surface normal
    normals = normals/ct.field_magnitude(normals)[:, None] # normalize the normal vectors
    return normals
    
def surface_polarization_directions(theta, phi):
    '''
    calculate direction vectors for p- and s- polarized light on the mirror surface
    p-polarized: polarized along theta (is this true for the paraboloid??)
    s-polarized: polarized along phi (is this true for the paraboloid??)
    '''
    p_direction = np.vstack(ct.spherical_to_cartesian_vector_field(
        theta,
        phi,
        np.zeros(np.shape(theta)),
        np.ones(np.shape(theta)),
        np.zeros(np.shape(theta)),
        ))
        
    s_direction = np.vstack(ct.spherical_to_cartesian_vector_field(
        theta,
        phi,
        np.zeros(np.shape(theta)),
        np.zeros(np.shape(theta)),
        np.ones(np.shape(theta)),
        ))
    return p_direction, s_direction
    
def fresnel_reflection_coefficients(normals, e_incident_direction, n_mirror, n_environment=1):
    '''
    calculate s and p fresnel reflection coefficients for the parabolic mirror surface
    normals: surface normals of the mirror
    e_incident_direction: direction vector of the incident electromagnetic wave
    n_environment: refractive index of the environment at the desired wavelength
    n_mirror: refractive index of the mirror at the desired wavelength
    '''
    incidence_angle = np.arccos(
        np.dot(normals, e_incident_direction)
        ) / (
            ct.field_magnitude(normals)*ct.field_magnitude(e_incident_direction)
        )
    n_factor = np.sqrt(1-np.square(n_environment/n_mirror * np.sin(incidence_angle)))
    R_s = np.square(
        (
            n_environment * np.cos(incidence_angle) - 
            n_mirror * n_factor
        ) / (
            n_environment * np.cos(incidence_angle) + 
            n_mirror * n_factor
            )
        )
    R_p = np.square(
        (
            n_environment * n_factor - 
            n_mirror * np.cos(incidence_angle)
        ) / (
            n_environment * n_factor + 
            n_mirror * np.cos(incidence_angle)
            )
        )
    return R_s, r_p

def aluminium_refractive_index(wavelength, filename):
    '''
    import data file containing aluminium's dielectric function
    wavelength: wavelength at which it is desired to extract the refractive index (in m)
    filename: file containing the dielectric function (should point to al_pa.mat)
    returns complex refractive index at the given wavelength
    '''
    load_data = np.loadtxt(filename, skiprows=15, max_rows=141-15) # must fix that to allow other files than al_pa.mat
    index_factor = np.sqrt(load_data[:, 1]**2 + load_data[:, 2]**2)
    n = np.sqrt(0.5 * (index_factor + load_data[:, 1])) + 1j * np.sqrt(0.5 * (index_factor - load_data[:, 1]))
    wavelength_list = constants.PLANCK * constants.LIGHTSPEED / (load_data[:, 0] * constants.COULOMB)
    interp_function_n = scint.interp1d(wavelength_list, n, kind='cubic')
    n_wavelength = interp_function_n(wavelength)
    return n_wavelength
    
    
