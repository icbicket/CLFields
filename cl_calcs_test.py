import cl_calcs
import unittest
import numpy as np
import coord_transforms
from absl.testing import parameterized

#def DoPTest(unittest.TestCase):
#    def test_degree_of_pol_normal(self):

class StokesParametersTest(unittest.TestCase):
    def testS1LinearPolarized(self):
        '''
        check inputs which should result in x or y polarization
        '''
        E1 = np.array([1, -1, 0, 0])
        E2 = np.array([0, 0, 1, -1])
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        S_calc = np.array([S0, S1, S2, S3])
        stokes = np.transpose(np.array([
            [1, 1, 0, 0], 
            [1, 1, 0, 0], 
            [1, -1, 0, 0], 
            [1, -1, 0, 0]
            ]))
        np.testing.assert_array_equal(S_calc, stokes)

    def testS3CircularPolarized(self):
        E1 = np.array([1 + 1j, 1 + 0.5j, 1 - 1j])
        E2 = np.array([1 - 1j, 0.5 - 1j, 1 + 1j])
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        print(S0)
        S_calc = np.array([S0, S1, S2, S3])
        stokes = np.transpose(np.array([[4, 0, 0, 4], [2.5, 0, 0, 2.5], [4, 0, 0, -4]]))
        np.testing.assert_array_equal(stokes, S_calc)

    def test_stokes_angle_polarized(self):
        E1 = 1 + 1j
        E2 = 1 - 0j
        stokes = np.array([3, 1, 2, 2])
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        S_calc = np.array([S0, S1, S2, S3])
        np.testing.assert_array_equal(stokes, S_calc)

    def testS2LinearPolarized(self):
        E1 = np.array([1, 1, -1, -1])
        E2 = np.array([1, -1, 1, -1])
        stokes = np.transpose(np.array([
            [2, 0, 2, 0], 
            [2, 0, -2, 0], 
            [2, 0, -2, 0],
            [2, 0, 2, 0],
            ]))
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

class AngleOfIncidenceTest(parameterized.TestCase):
    '''
    Test the function for calculating the angle of incidence of a wave on a 
    surface, given the incoming wavevector and the surface normal
    '''
    
    @parameterized.named_parameters(
        ('45 degrees', 
            np.array([0, 1, 1]), 
            np.array([0, 0, 1]), 
            np.array(np.pi/4),
        ),
        ('0 degrees',
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0]),
        ),
        ('90 degrees',
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([np.pi/2]),
        ),
        ('45 degrees opposite to normal (+y,+z):(-z)',
            np.array([0, 1, 1]),
            np.array([0, 0, -1]),
            np.array(np.pi/4),
        ),
        ('45 degrees opposite to normal (-y,+z):(-z)',
            np.array([0, -1, 1]),
            np.array([0, 0, -1]),
            np.array(np.pi/4),
        ),
        ('45 degrees to normal (+y,-z):(-z)',
            np.array([0, 1, -1]),
            np.array([0, 0, -1]),
            np.array(np.pi/4),
        ),
        ('45 degrees to normal (-y,-z):(-z)',
            np.array([0, -1, -1]),
            np.array([0, 0, -1]),
            np.array(np.pi/4),
        ),
    )
    def test_angle_of_incidence_single_values(self, incident, normal, expected_angle):
        '''
        check that the angle of incidence is calculated correctly given
        several simple cases
        '''
        angle = cl_calcs.angle_of_incidence(incident, normal)
        np.testing.assert_allclose(angle, expected_angle, atol=1e-7)

    def test_angle_of_incidence_multi_value_array(self):
        '''
        check that the angle of incidence is calculated correctly given an 
        N by 3 array
        '''
        incident = np.array([
            [0, 1, 1], 
            [0, 1, 0], 
            [0, 1, 0], 
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
            ])
        normal = np.array([
            [0, 1, 0], 
            [0, 1, 0], 
            [1, 0, 0], 
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            ])
        expected_angles = np.array([
            np.pi/4, 
            0, 
            np.pi/2, 
            np.pi/4,
            np.pi/4,
            np.pi/4,
            np.pi/4,
            ])
        angles = cl_calcs.angle_of_incidence(incident, normal)
        np.testing.assert_allclose(angles, expected_angles)

class SnellsLawTest(parameterized.TestCase):
    '''
    Test the Snell's law function for calculating the angle of refraction
    '''
    def test_angle_array(self):
        '''
        test the standard case in which an array of angles is used and the 
        surface has a refractive index of 2
        '''
        incidence_angles = np.array([
            0, 
            np.pi/8, 
            np.pi/4, 
            np.pi/3, 
            np.pi/2-0.1,
            np.pi/2,
            ])
        n_surface = 2
        n_environment = 1
        expected_refraction_angles = np.array([
            0, 
            0.192528938, 
            0.361367123, 
            0.447832396, 
            0.520716822,
            0.523598775,
            ])
        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)

    def test_single_angle(self):
        '''
        test a single angle value is used as input, and the surface has a 
        refractive index of 2
        '''
        incidence_angles = np.pi/8
        n_surface = 2
        n_environment = 1
        expected_refraction_angles = 0.192528938
        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)

    @parameterized.named_parameters(
        ('critical angle', np.pi/6, np.pi/2),
        ('below critical angle', np.pi/6-0.1, 0.965067965),
    )
    def test_critical_angle(self, incidence_angle, expected_refraction_angle):
        '''
        the second medium has a smaller refractive index than the first, test
        the angle at which total internal reflection occurs, and a value below 
        that
        '''
        n_surface = 1
        n_environment = 2
        refraction_angle = cl_calcs.snells_law(incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose(refraction_angle, expected_refraction_angle, atol=1e-7)
        
    def test_below_critical_angle(self):
        '''
        the second medium has a smaller refractive index than the first, test
        an angle shallower than that at which total internal reflection occurs
        - the result should be NaN
        '''
        incidence_angles = np.pi/6+0.1
        n_surface = 1
        n_environment = 2
        refraction_angle = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
        assert np.isnan(refraction_angle)

class ReflectionCoefficientsTest(parameterized.TestCase):
    '''
    Test the reflection coefficient calculation
    '''
    def test_brewsters_angle(self):
        '''
        the parallel reflection coefficient should be 0 at Brewster's angle
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = 1.107148718
        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
        expected_r_p = 0
        np.testing.assert_allclose(r_p, expected_r_p, atol=1e-7)

    def test_normal_incidence(self):
        '''
        the reflection coefficients should be equal at normal incidence
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = 0
        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose(r_p, r_s, atol=1e-7)
    
    def test_single_value(self):
        '''
        test the reflection coefficient calculation with a single input angle
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = 1.
        r = np.array(cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment))
        expected_r = np.array([-0.54108004, 0.087243335])
        np.testing.assert_allclose(r, expected_r, atol=1e-7)
    
    def test_array_of_values(self):
        '''
        test the reflection coefficient calculation with a 1D numpy array input
            for incidence angle
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = np.array([1., np.pi/4])
        r = np.array(cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment))
        expected_r = np.transpose(np.array([[-0.54108004, 0.087243335], [-0.451416229, 0.203776612]]))
        np.testing.assert_allclose(r, expected_r, atol=1e-7)
   

class BrewstersAngleTest(parameterized.TestCase):
    '''
    Test the reflection coefficient calculation
    '''
    def test_single_value(self):
        '''
        calculate Brewster's angle for n1=1, n2=2
        '''
        n_surface = 2
        n_environment = 1
        expected_brewsters = 1.107148718
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        np.testing.assert_allclose(brewsters, expected_brewsters)
    
    def test_value_array(self):
        '''
        calculate Brewster's angle for n1=1, n2=2
        '''
        n_surface = np.array([2, 3])
        n_environment = np.array([1, 2])
        expected_brewsters = np.array([1.107148718, 0.982793723])
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        np.testing.assert_allclose(brewsters, expected_brewsters)

if __name__ == '__main__':
    unittest.main()
