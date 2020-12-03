import numpy as np

def polar_to_cartesion(r, phi):
    '''
    Two 1D vectors of polar r (radial) and phi (angular) coordinates
    Function returns the Cartesian vectors
    '''
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def cartesian_to_polar(x, y):
    '''
    Two 1d vectors of cartesian x and y coordinates
    Function converts them to polar coordinates r and phi
    '''
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    return r, phi

def cartesian_to_spherical_coords(vectors):
    '''
    N by 3 vectors of (x, y, z) are converted to spherical coordinates
    '''
    r = np.sqrt(np.sum(np.square(vectors), axis=-1))
    theta = np.arctan(np.sqrt(np.square(vectors[:, 0]) + np.square(vectors[:, 1]))/vectors[:, 2])
    phi = np.zeros(np.size(vectors[:, 0]))
    phi[vectors[:, 0] != 0] = np.arctan(vectors[:, 1][vectors[:, 0] != 0]/vectors[:, 0][vectors[:, 0] !=0])
#    theta = np.arctan(vectors[:, 1]/vectors[:, 2])
#    phi = np.arccos(vectors[:, 0]/r)
    phi[(vectors[:, 1] > 0) * (vectors[:, 0] < 0)] += np.pi # +y, -z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] >= 0)] += 2*np.pi # -y, +z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] < 0)] += np.pi # -y, -z
    phi[(vectors[:, 1] > 0) * (vectors[:, 0] == 0)] = np.pi/2
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] == 0)] = 3*np.pi/2
    phi[(vectors[:, 1] == 0) * (vectors[:, 0] <= 0)] = np.pi
    theta[theta<0] += np.pi
    return r, theta, phi

def spherical_to_cartesian_coords(vectors):
    '''
    N by 3 vectors of (r, theta, phi)
    '''
    x = vectors[:, 0] * np.sin(vectors[:, 1]) * np.cos(vectors[:, 2])
    y = vectors[:, 0] * np.sin(vectors[:, 1]) * np.sin(vectors[:, 2])
    z = vectors[:, 0] * np.cos(vectors[:, 1])
    return x, y, z

def cartesian_to_spherical_vector_field(theta, phi, fx, fy, fz):
    f_r = (np.sin(theta) * np.cos(phi) * fx + 
       np.sin(theta) * np.sin(phi) * fy + 
       np.cos(theta) * fz)
    f_th = (np.cos(theta) * np.cos(phi) * fx + 
        np.cos(theta) * np.sin(phi) * fy - 
        np.sin(theta) * fz)
    f_ph = (-np.sin(phi) * fx + 
        np.cos(phi) * fy)
    return f_r, f_th, f_ph

def spherical_to_cartesian_vector_field(theta, phi, f_r, f_th, f_ph):
    f_x = (np.sin(theta) * np.cos(phi) * f_r + 
           np.cos(theta) * np.cos(phi) * f_th - 
           np.sin(phi) * f_ph)
    f_y = (np.sin(theta) * np.sin(phi) * f_r + 
           np.cos(theta) * np.sin(phi) * f_th +
           np.cos(phi) * f_ph)
    f_z = (np.cos(theta) * f_r - 
           np.sin(theta) * f_th)
    return f_x, f_y, f_z

def field_magnitude(f, axis=-1):
    f_mag = np.sqrt(np.sum(f * np.conj(f), axis=axis))
    f_mag = np.real(f_mag)
    return f_mag

def rotate_vector(xyz, angle, rotation_axis):
    '''
    xyz: a vectors to be rotated, length 3
    angle: the angle by which to rotate xyz, in radians
    rotation_axis: the axis around which to rotate xyz of length 3
    '''
    if np.count_nonzero(xyz) == 0:
        raise ValueError('Input vector is the null vector')
    term1_rot = xyz*np.cos(angle)
    term2_rot = np.cross(rotation_axis, xyz, axis=0) * np.sin(angle)
    term3_rot = rotation_axis * np.transpose(np.tensordot(rotation_axis, xyz, axes=(0,0)))*(1-np.cos(angle))
    xyz_rot = term1_rot + term2_rot + term3_rot
    return xyz_rot
    
def rotate_vector_Nd(xyz, angle, rotation_axis):
    '''
    xyz: an array of vectors to be rotated, with N dimensions, of which the last has 3 elements
    angle: the angle by which to rotate xyz, in radians
    rotation_axis: the axis around which to rotate xyz, of length 3
    '''
    # Normalize rotation axis
    rotation_axis = rotation_axis/np.sqrt(np.sum(np.square(rotation_axis)))
    # Expand dimensions of rotation axis for broadcasting
    rotation_axis = np.reshape(rotation_axis, (xyz.ndim-1)*[1]+[3])
    # Calculate rotation vector (Rodrigues formula)
    term1_rot = xyz*np.cos(angle)
    term2_rot = np.cross(rotation_axis, xyz, axis=-1) * np.sin(angle)
    term3_rot = rotation_axis * np.expand_dims(np.dot(xyz, np.squeeze(rotation_axis)), axis=-1)*(1-np.cos(angle))
    xyz_rot = term1_rot + term2_rot + term3_rot
    return xyz_rot

def expand_quadrant_symmetry(mag, quadrant_num):
    '''
    Uses symmetry to turn an array representing values in one quadrant of an
    image into values for a whole image/array
    quadrant_num = 1, 2, 3 ,4
    quadrants: 1  2
               3  4
    assumes image should be reflected around two axes, removing double rows/columns of pixels caused by reflection
    '''
    if quadrant_num == 1:
        Q1 = mag
        Q2 = np.flip(Q1, axis = 1)[:, 1:]
        Q3 = np.flip(Q1, axis = 0)[1:, :]
        Q4 = np.flip(np.flip(Q1, axis = 1), axis=0)[1:, 1:]
    elif quadrant_num == 2:
        Q2 = mag
        Q1 = np.flip(Q2, axis = 1)[:, :-1]
        Q4 = np.flip(Q2, axis = 0)[1:, :]
        Q3 = np.flip(np.flip(Q2, axis=1), axis=0)[1:, :-1]
    elif quadrant_num == 3:
        Q3 = mag
        Q4 = np.flip(Q3, axis=1)[:, 1:]
        Q1 = np.flip(Q3, axis=0)[:-1, :]
        Q2 = np.flip(np.flip(Q3, axis=1), axis=0)[:-1, 1:]
    elif quadrant_num == 4:
        Q4 = mag
        Q3 = np.flip(Q4, axis=1)[:, :-1]
        Q2 = np.flip(Q4, axis = 0)[:-1, :]
        Q1 = np.flip(np.flip(Q4, axis=1), axis=0)[:-1, :-1]
    
    Q12 = np.append(Q1, Q2, axis=1)
    Q34 = np.append(Q3, Q4, axis=1)
    total = np.append(Q12, Q34, axis=0)
    return total
