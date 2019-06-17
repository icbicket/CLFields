import numpy as np

def cartesian_to_spherical_coords(vectors):
    r = np.sqrt(np.sum(np.square(vectors), axis=-1))
    theta = np.arctan(np.sqrt(np.square(vectors[:, 0]) + np.square(vectors[:, 1]))/vectors[:, 2])
    phi = np.arctan(vectors[:, 1]/vectors[:, 0])
#    theta = np.arctan(vectors[:, 1]/vectors[:, 2])
#    phi = np.arccos(vectors[:, 0]/r)
    phi[(vectors[:, 1] > 0) * (vectors[:, 0] < 0)] += np.pi # +y, -z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] >= 0)] += 2*np.pi # -y, +z
    phi[(vectors[:, 1] < 0) * (vectors[:, 0] < 0)] += np.pi # -y, -z
    theta[theta<0] += np.pi
    return r, theta, phi

def cartesian_to_spherical_vector_field(theta, phi, fx, fy, fz):
    fr = (np.sin(theta) * np.cos(phi) * fx + 
       np.sin(theta) * np.sin(phi) * fy + 
       np.cos(theta) * fz)
    e_thz = (np.cos(theta) * np.cos(phi) * fx + 
        np.cos(theta) * np.sin(phi) * fy - 
        np.sin(theta) * fz)
    e_phz = (-np.sin(phi) * fx + 
        np.cos(phi) * fy)

def field_magnitude(f):
    f_mag = np.sqrt(np.sum(f * np.conj(f), axis=-1))
    f_mag = np.real(f)
    return f_mag

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
