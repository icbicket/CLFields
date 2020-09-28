import cl_calcs
import unittest
import numpy as np
import coord_transforms

#def DoPTest(unittest.TestCase):
#    def test_degree_of_pol_normal(self):

class StokesParametersTest(unittest.TestCase):
    def test_stokes_x_polarized(self):
        E1 = 1
        E2 = 0
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        S_calc = np.array([S0, S1, S2, S3])
        stokes = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(S_calc, stokes)

    def test_stokes_circular_polarized(self):
        E1 = 1 + 1j
        E2 = 1 - 1j
        stokes = np.array([4, 0, 0, 4])
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        S_calc = np.array([S0, S1, S2, S3])
        np.testing.assert_array_equal(stokes, S_calc)

    def test_stokes_angle_polarized(self):
        E1 = 1 + 1j
        E2 = 1 - 0j
        stokes = np.array([3, 1, 2, 2])
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        S_calc = np.array([S0, S1, S2, S3])
        np.testing.assert_array_equal(stokes, S_calc)

class DoPTest(unittest.TestCase):
    def test_pol(self):
        S0 = np.array([3,])
        S1 = np.array([1,])
        S2 = np.array([2,])
        S3 = np.array([-2,])
        dop = np.transpose(np.array([[1, np.sqrt(5)/S0[0], -2/3, -2/(3+np.sqrt(5))]]))
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        dop_calc = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(dop, dop_calc)

class ARMaskTest(unittest.TestCase):
    '''
    Test angle-resolved mirror masking function
    x and y axes are flipped
    '''
    def test_noslit_nohole(self):
        mask = np.array([True, True, True, True,
                        False, False, False, True, 
                        False])
        mask_calc = cl_calcs.ar_mask_calc(
            self.theta, 
            self.phi, 
            holein=False, 
            slit=None, 
            orientation=0)
        np.testing.assert_array_equal(mask, mask_calc)

    def test_noslit_hole(self):
        mask = np.array([True, True, True, True,
                        False, False, False, True, 
                        True])
        mask_calc = cl_calcs.ar_mask_calc(
            self.theta, 
            self.phi, 
            holein=True, 
            slit=None, 
            orientation=0)
        np.testing.assert_array_equal(mask, mask_calc)

    def test_noslit_nohole_rot90(self):
        mask = np.array([True, True, True, True,
                        True, False, False, False, 
                        False])
        mask_calc = cl_calcs.ar_mask_calc(
            self.theta, 
            self.phi, 
            holein=False, 
            slit=None, 
            orientation=np.pi/2)
        np.testing.assert_array_equal(mask, mask_calc)

    def test_noslit_hole_rot180(self):
        mask = np.array([True, True, True, True,
                        False, True, False, False, 
                        True])
        mask_calc = cl_calcs.ar_mask_calc(
            self.theta, 
            self.phi, 
            holein=True, 
            slit=None, 
            orientation=np.pi)
        np.testing.assert_array_equal(mask, mask_calc)

    def setUp(self):
        self.theta =  np.array([
                       np.pi/2, # +y, side, in xy plane
                       np.pi/2, #+x, back, in xy plane
                       np.pi/2, # -y, side, in xy plane
                       np.pi/2, # -x, front, in xy plane
                       np.pi/2-0.2, # +y, side, slightly out of xy plane (78 deg)
                       np.pi/2-0.2, # +x, back, slightly out of xy plane
                       np.pi/2-0.2, # -y, side, slightly out of xy plane
                       np.pi/2-0.2, # -x, front, slightly out of xy plane
                       0, #hole
                       ])
        self.phi = np.array([
                    np.pi/2, #+y
                    0, #+x
                    3*np.pi/2, #-y
                    np.pi, #-x
                    np.pi/2, #+y
                    0, #+x
                    3*np.pi/2, #-y
                    np.pi, #-x
                    0 #hole
                    ])

if __name__ == '__main__':
    unittest.main()
