import numpy as np
import coord_transforms as ct
import constants
import cl_calcs as clc
import scipy.interpolate as scint
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ParabolicMirror:
    """
    Class for keeping track of mirror parameters.
    dielectric is a numpy array with the energy in eV, and the real and complex components of the material's dielectric function
    """
    a: float
    dfoc: float
    xcut: float
    thetacutoffhole: float
    dielectric: np.array

AMOLF_MIRROR = ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75, thetacutoffhole=4., dielectric=np.loadtxt('al_pa.mat', skiprows=15, max_rows=141-15))

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
        _, _, corner1_phi = ct.cartesian_to_spherical_coords(np.array([[mirror.xcut, corner1_y, 1]]))
        corner2_y = -np.sqrt(25-mirror.xcut*10-mirror.dfoc**2)
        _, _, corner2_phi = ct.cartesian_to_spherical_coords(np.array([[mirror.xcut, corner2_y, 1]]))
        
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
        _, _, corner1_phi = ct.cartesian_to_spherical_coords(
            np.array([[corner1_x, ycut_positive, mirror.dfoc]])
            )
        _, _, corner2_phi = ct.cartesian_to_spherical_coords(
            np.array([[mirror.xcut, ycut_positive, mirror.dfoc]])
            )
        _, _, corner3_phi = ct.cartesian_to_spherical_coords(
            np.array([[mirror.xcut, ycut_negative, mirror.dfoc]])
            )
        corner4_x = 2.5 - (ycut_negative**2 + mirror.dfoc**2)/10
        _, _, corner4_phi = ct.cartesian_to_spherical_coords(
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
    vector = np.transpose(np.array([x, y, z]))
    _, theta, _ = ct.cartesian_to_spherical_coords(vector)
    phi = phi + orientation
    mirror_theta_phi = np.transpose(np.array([theta, phi]))
    return mirror_theta_phi, hole_theta_phi


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

def get_mirror_refractive_index(wavelength, mirror=AMOLF_MIRROR, fit_kind='cubic'):
    '''
    Interpolate the refractive index at the desired wavelength, given the wavelength and the mirror
    '''
#    index_factor = np.sqrt(mirror.dielectric[:, 1]**2 + mirror.dielectric[:, 2]**2)
#    n = np.sqrt(0.5 * (index_factor + mirror.dielectric[:, 1])) + 1j * np.sqrt(0.5 * (index_factor - mirror.dielectric[:, 1]))
    n = clc.dielectric_to_refractive(mirror.dielectric[:, 1:])
    wavelength_list = clc.eV_to_wavelength(mirror.dielectric[:, 0])
#    wavelength_list = constants.PLANCK * constants.LIGHTSPEED / (load_data[:, 0] * constants.COULOMB)
    interp_function_n = scint.interp1d(wavelength_list, n, kind=fit_kind)
    n_wavelength = interp_function_n(wavelength)
    return n_wavelength

