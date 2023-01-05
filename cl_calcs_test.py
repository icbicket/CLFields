import cl_calcs
import unittest
import numpy as np
import coord_transforms
from absl.testing import parameterized
import sys
np.set_printoptions(threshold=sys.maxsize)

#def DoPTest(unittest.TestCase):
#    def test_degree_of_pol_normal(self):

'''
Unittests for the following functions:
- ar_mask_calc
- degree_of_polarization
- mirror_mask3d
- mirror_outline
- angle_of_incidence
- snells_law
- brewsters_angle
- reflection_coefficients
- reflected_e
- stokes_parameters
- normalize_stokes_parameters
'''

class DegreeOfPolarizationTest(parameterized.TestCase):
    '''
    fully polarized light-mixed states
    fully linearly polarized light
    fully circularly polarized light
    partially polarized linear light
    partially polarized circular light
    totally unpolarized light
    S0 is 0
    S0 is negative
    square sums of S1-3 greater than square of S0
    single value array
    single float/integer
    n by 1 array
    1 by n array
    n by m by p array
    '''
    def test_mixed_polarized_light(self):
        '''
        Polarized light with mixed polarization states
        '''
        S0 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 0.9273618495495703])
        S1 = np.array([1, 1, -1, 1, -1, 1, -1, -1, 0.6])
        S2 = np.array([2, 2, 2, -2, -2, -2, 2, -2, 0.7])
        S3 = np.array([-2, 2, 2, 2, 2, -2, -2, -2, 0.1])
        expected = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 
                0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 
                0.7453559924999299, 0.7453559924999299, 0.994169046], 
            [2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 0.107832773], 
            [-0.38196601125010515, 0.38196601125010515, 0.38196601125010515, 
                0.38196601125010515, 0.38196601125010515, -0.38196601125010515,
                -0.38196601125010515, -0.38196601125010515, 0.054074038]
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(expected, calculated)

    def test_linear_polarized_light(self):
        '''
        Various types of linearly polarized light
        '''
        S0 = np.array([1, 1, 1, 1, 
            np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2), 1/np.sqrt(2)])
        S1 = np.array([1, 0, -1, 0, 1, 1, -1, -1, 0.5])
        S2 = np.array([0, 1, 0, -1, 1, -1, 1, -1, 0.5])
        S3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # DoP
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # DoLP
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # DoCP
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_circular_polarized_light(self):
        '''
        Various types of pure circularly polarized light
        '''
        S0 = np.array([1, 1, 0.5, 0.5])
        S1 = np.array([0, 0, 0, 0])
        S2 = np.array([0, 0, 0, 0])
        S3 = np.array([1, -1, 0.5, -0.5])
        expected = np.array([
            [1, 1, 1, 1],  # DoP
            [0, 0, 0, 0],  # DoLP
            [1, 1, 1, 1],  # DoCP
            [1, -1, 1, -1],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_partially_linear_polarized_light(self):
        '''
        Partially linearly polarized light - no circular
        '''
        S0 = np.array([2, 2, 2, 2, 3, 3, 3, 3, 2])
        S1 = np.array([1, -1, 0, 0, 1, -1, 1, -1, 0.5])
        S2 = np.array([0, 0, 1, -1, 1, 1, -1, -1, 0.5])
        S3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected = np.array([
            [0.5, 0.5, 0.5, 0.5, 0.47140452079103173, 0.47140452079103173, 
                0.47140452079103173, 0.47140452079103173, 
                0.35355339059327373],  # DoP
            [0.5, 0.5, 0.5, 0.5, 0.47140452079103173, 0.47140452079103173, 
                0.47140452079103173, 0.47140452079103173, 
                0.35355339059327373],  # DoLP
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # DoCP
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_partially_circular_polarized_light(self):
        '''
        Various types of partial circularly polarized light
        '''
        S0 = np.array([1.5, 1.5, 2, 2])
        S1 = np.array([0, 0, 0, 0])
        S2 = np.array([0, 0, 0, 0])
        S3 = np.array([1, -1, 0.5, -0.5])
        expected = np.array([
            [2/3, 2/3, 0.25, 0.25],  # DoP
            [0, 0, 0, 0],  # DoLP
            [2/3, 2/3, 0.25, 0.25],  # DoCP
            [2/3, -2/3, 0.25, -0.25],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_unpolarized_light(self):
        '''
        Unpolarized light
        '''
        S0 = np.array([1, 1.5])
        S1 = np.array([0, 0])
        S2 = np.array([0, 0])
        S3 = np.array([0, 0])
        expected = np.array([
            [0, 0],  # DoP
            [0, 0],  # DoLP
            [0, 0],  # DoCP
            [0, 0],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    def test_S0_zero(self):
        '''
        S0 is 0
        '''
        S0 = np.array([0])
        S1 = np.array([0])
        S2 = np.array([0])
        S3 = np.array([0])
        expected = np.array([
            [0],  # DoP
            [0],  # DoLP
            [0],  # DoCP
            [0],  # ellipticity
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(calculated, expected)

    @parameterized.named_parameters(
        dict(testcase_name='S0: one is less than 0',
            S0 = np.array([-1, 1]),
            S1 = np.array([1, 1]),
            S2 = np.array([0, 0]),
            S3 = np.array([0, 0]),
            ),
        dict(testcase_name='S0: all are less than 0',
            S0 = np.array([-1]),
            S1 = np.array([0]),
            S2 = np.array([1]),
            S3 = np.array([0]),
            ),
        )
    def test_S0_negative(self, S0, S1, S2, S3):
        '''
        S0 is < 0
        '''
        with self.assertRaises(ValueError):
            DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)

    @parameterized.named_parameters(
        dict(testcase_name='S0 less in integers',
            S0 = np.array([2]),
            S1 = np.array([5]),
            S2 = np.array([0]),
            S3 = np.array([0]),
            ),
        dict(testcase_name='S0 less in floats',
            S0 = np.array([0.1]),
            S1 = np.array([1]),
            S2 = np.array([0.5]),
            S3 = np.array([0.2]),
            ),
        dict(testcase_name='S0 less, on the edge',
            S0 = np.array([1.12]),
            S1 = np.array([1]),
            S2 = np.array([0.5]),
            S3 = np.array([0.2]),
            ),
        dict(testcase_name='one S0 value is less, on the edge',
            S0 = np.array([1.12, 1.14]),
            S1 = np.array([1, 1]),
            S2 = np.array([0.5, 0.5]),
            S3 = np.array([0.2, 0.2]),
            ),
        )
    def test_S0_greater_than_sum_squares(self, S0, S1, S2, S3):
        '''
        S0^2 is < (S1^2+S2^2+S3^2)
        '''
        with self.assertRaises(ValueError):
            DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)

    def test_single_value_array(self):
        '''
        input is a single value
        '''
        S0 = np.array([3])
        S1 = np.array([1])
        S2 = np.array([2])
        S3 = np.array([-2])
        expected = np.array([
            [1], 
            [0.7453559924999299], 
            [2/3], 
            [-0.38196601125010515]
            ])
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(expected, calculated)
        
    def test_single_integer(self):
        '''
        input for each Stokes parameter is a single integer
        '''
        S0 = 3
        S1 = 1
        S2 = 2
        S3 = -2
        expected = np.array([
            [1], 
            [0.7453559924999299], 
            [2/3], 
            [-0.38196601125010515]
            ])
        with self.assertRaises(TypeError):
            DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)

    def test_single_float(self):
        '''
        input for each Stokes parameter is a single float
        '''
        S0 = 3.
        S1 = 1.
        S2 = 2.
        S3 = -2.
        expected = np.array([
            [1], 
            [0.7453559924999299], 
            [2/3], 
            [-0.38196601125010515]
            ])
        with self.assertRaises(TypeError):
            DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)

    def test_n_by_1_array(self):
        '''
        Input is an n by 1 array
        '''
        S0 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 0.9273618495495703]).reshape((9,1))
        S1 = np.array([1, 1, -1, 1, -1, 1, -1, -1, 0.6]).reshape((9,1))
        S2 = np.array([2, 2, 2, -2, -2, -2, 2, -2, 0.7]).reshape((9,1))
        S3 = np.array([-2, 2, 2, 2, 2, -2, -2, -2, 0.1]).reshape((9,1))
        expected_DoP = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((9,1))
        expected_DoLP = np.array([0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.994169046]).reshape((9,1))
        expected_DoCP = np.array([2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 0.107832773]).reshape((9,1))
        expected_ell = np.array([-0.38196601125010515, 0.38196601125010515, 
            0.38196601125010515, 0.38196601125010515, 0.38196601125010515, 
            -0.38196601125010515, -0.38196601125010515, -0.38196601125010515, 
            0.054074038]).reshape((9,1))
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(expected_DoP, DoP)
        np.testing.assert_array_almost_equal(expected_DoLP, DoLP)
        np.testing.assert_array_almost_equal(expected_DoCP, DoCP)
        np.testing.assert_array_almost_equal(expected_ell, ell)

    def test_1_by_n_array(self):
        '''
        Input is a 1 by n array
        '''
        S0 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 0.9273618495495703]).reshape((1,9))
        S1 = np.array([1, 1, -1, 1, -1, 1, -1, -1, 0.6]).reshape((1,9))
        S2 = np.array([2, 2, 2, -2, -2, -2, 2, -2, 0.7]).reshape((1,9))
        S3 = np.array([-2, 2, 2, 2, 2, -2, -2, -2, 0.1]).reshape((1,9))
        expected_DoP = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((1,9))
        expected_DoLP = np.array([0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 0.994169046]).reshape((1,9))
        expected_DoCP = np.array([2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 0.107832773]).reshape((1,9))
        expected_ell = np.array([-0.38196601125010515, 0.38196601125010515, 
            0.38196601125010515, 0.38196601125010515, 0.38196601125010515, 
            -0.38196601125010515, -0.38196601125010515, -0.38196601125010515, 
            0.054074038]).reshape((1,9))
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(expected_DoP, DoP)
        np.testing.assert_array_almost_equal(expected_DoLP, DoLP)
        np.testing.assert_array_almost_equal(expected_DoCP, DoCP)
        np.testing.assert_array_almost_equal(expected_ell, ell)

    def test_3D_array(self):
        '''
        Input is a 3d array
        '''
        S0 = np.array([3, 3, 3, 3, 3, 3, 3, 0.9273618495495703]).reshape((2,2,2))
        S1 = np.array([1, 1, -1, 1, -1, 1, -1, 0.6]).reshape((2,2,2))
        S2 = np.array([2, 2, 2, -2, -2, -2, 2, 0.7]).reshape((2,2,2))
        S3 = np.array([-2, 2, 2, 2, 2, -2, -2, 0.1]).reshape((2,2,2))
        expected_DoP = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape((2,2,2))
        expected_DoLP = np.array([0.7453559924999299, 0.7453559924999299, 
            0.7453559924999299, 0.7453559924999299, 0.7453559924999299, 
            0.7453559924999299, 0.7453559924999299, 0.994169046
            ]).reshape((2,2,2))
        expected_DoCP = np.array([
            2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3, 0.107832773]).reshape((2,2,2))
        expected_ell = np.array([-0.38196601125010515, 0.38196601125010515, 
            0.38196601125010515, 0.38196601125010515, 0.38196601125010515, 
            -0.38196601125010515, -0.38196601125010515, 0.054074038
            ]).reshape((2,2,2))
        DoP, DoLP, DoCP, ell = cl_calcs.degree_of_polarization(S0, S1, S2, S3)
        calculated = np.array([DoP, DoLP, DoCP, ell])
        np.testing.assert_array_almost_equal(expected_DoP, DoP)
        np.testing.assert_array_almost_equal(expected_DoLP, DoLP)
        np.testing.assert_array_almost_equal(expected_DoCP, DoCP)
        np.testing.assert_array_almost_equal(expected_ell, ell)


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
        ('off-axis incident and normal vectors',
            np.array([1, 2, 0.5]),
            np.array([5, -2, 1]),
            1.450987042,
        ),
        ('small angle',
            np.array([1, 2, 0.5]),
            np.array([1, 2, 0.49]),
            4.263210237e-3,
        ),
        ('very large',
            np.array([1, 2, 0.5]),
            np.array([-1, -2, -0.49]),
            4.263210237e-3,
        ),
    )
    def test_angle_of_incidence_single_values(self, incident, normal, expected_angle):
        '''
        check that the angle of incidence is calculated correctly given
        several simple cases
        '''
        angle = cl_calcs.angle_of_incidence(incident, normal)
        np.testing.assert_allclose(angle, expected_angle, atol=1e-7)

    @parameterized.named_parameters(
        ('zero incident',
            np.array([0,0,0]),
            np.array([1,1,1])
        ),
        ('zero normal',
            np.array([1,1,1]),
            np.array([0,0,0]),
        ),
        )
    def test_vector_magnitude_zero(self, incident, normal):
        '''
        check that given a vector of zeros, it fails
        '''
        with self.assertRaises(ValueError):
            cl_calcs.angle_of_incidence(incident, normal)

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
            [np.pi/4],
            [0],
            [np.pi/2],
            [np.pi/4],
            [np.pi/4],
            [np.pi/4],
            [np.pi/4],
            ])
        angles = cl_calcs.angle_of_incidence(incident, normal)
        np.testing.assert_allclose(angles, expected_angles)

    @parameterized.named_parameters(
        ('ones',
            np.array([1,1,1]),
            np.array([1,1,1])
        ),
        ('not ones',
            np.array([40929,40929,40929]),
            np.array([1,1,1]),
        ),
        )
    def test_floating_point_problem_sinusoids(self, incident_vector, normal):
        '''
        Floating point rounding errors can make the value to be fed to an 
        inverse cosine greater than 1: test the round response is appropriate
        '''
        expected_angle = 0.
        calculated_angle = cl_calcs.angle_of_incidence(incident_vector, normal)
        self.assertAlmostEqual(expected_angle, calculated_angle)


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

    def test_complex_refractive_index_surface(self):
        '''
        The surface has a complex refractive index
        '''
        incidence_angles = np.pi/8
        n_surface = 2+1j
        n_environment = 1
        expected_refraction_angles = 0.15321514223203334-0.07736669861166084j
        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)

    def test_complex_refractive_index_environment(self):
        '''
        The environment has a complex refractive index
        '''
        incidence_angles = np.pi/8
        n_surface = 2
        n_environment = 1-1j
        expected_refraction_angles = 0.18893317218363231-0.19359670723263944j
        refraction_angles = cl_calcs.snells_law(incidence_angles, n_surface, n_environment)
        np.testing.assert_allclose(refraction_angles, expected_refraction_angles, atol=1e-7)


class BrewstersAngleTest(parameterized.TestCase):
    '''
    Test the Brewster's angle calculation
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
        calculate Brewster's angle for n1=1, n2=2, using arrays for refractive
            indices
        '''
        n_surface = np.array([2, 3])
        n_environment = np.array([1, 2])
        expected_brewsters = np.array([1.107148718, 0.982793723])
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        np.testing.assert_allclose(brewsters, expected_brewsters)

    def test_same_refractive_index(self):
        '''
        Both n1 and n2 are the same - there is no surface to reflect off!
        '''
        n_surface = 1
        n_environment = 1
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        assert np.isnan(brewsters)

    def test_complex_n_surface(self):
        '''
        calculate Brewster's angle for n1=1+1j, n2=2
        '''
        n_surface = 2+1j
        n_environment = 1
        expected_brewsters = 1.107148718
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        np.testing.assert_allclose(brewsters, expected_brewsters)

    def test_complex_n_environment(self):
        '''
        calculate Brewster's angle for n1=1, n2=2+1j
        '''
        n_surface = 2
        n_environment = 1+1j
        expected_brewsters = 1.107148718
        brewsters = cl_calcs.brewsters_angle(n_surface, n_environment)
        np.testing.assert_allclose(brewsters, expected_brewsters)


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
        np.testing.assert_allclose(r_p, expected_r_p, atol=1e-7, equal_nan=False)

    def test_normal_incidence(self):
        '''
        the reflection coefficients should be equal at normal incidence
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = 0
        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose(r_p, r_s, atol=1e-7)

    @parameterized.named_parameters(
        dict(testcase_name='+x, -y',
             incidence_angle=np.pi/4,
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
             ),
         dict(testcase_name='ones',
              incidence_angle=1,
              expected_r_s=np.array([-0.54108004]),
              expected_r_p=np.array([-0.087243335]),
             ),
        dict(testcase_name='-x, -y',
             incidence_angle=np.pi/4,
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
             ),
        dict(testcase_name='+x, +y',
             incidence_angle=np.pi/4,
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
            ),
        dict(testcase_name='-x, +y',
             incidence_angle=np.pi/4,
             expected_r_s=np.array([-0.45141623]),
             expected_r_p=np.array([-0.20377661]),
            ),
        dict(testcase_name='Brewster angle',
             incidence_angle=1.107148718,
             expected_r_s=np.array([-0.6]),
             expected_r_p=np.array([0]),
             ),
        dict(testcase_name='not nice axis',
            incidence_angle=1.0502663767496385,
             expected_r_s=np.array([-0.5674137875163545]),
             expected_r_p=np.array([-0.049406606222754904]),
             ),
        )
    def test_single_value(self, incidence_angle, expected_r_s, expected_r_p):
        '''
        test the reflection coefficient calculation with a single input angle
        '''
        n_surface = 2
        n_environment = 1
        r_s, r_p = cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose((r_s, r_p), (expected_r_s, expected_r_p), atol=1e-7)
    
    def test_array_of_values(self):
        '''
        test the reflection coefficient calculation with a 1D numpy array input
            for incidence angle
        '''
        n_surface = 2
        n_environment = 1
        incidence_angle = np.array([1., np.pi/4])
        r = np.array(cl_calcs.reflection_coefficients(incidence_angle, n_surface, n_environment))
        expected_r = np.transpose(np.array([[-0.54108004, -0.087243335], [-0.451416229, -0.203776612]]))
        np.testing.assert_allclose(r, expected_r, atol=1e-7)

    def test_complex_refractive_index(self):
        '''
        test the fresnel reflection coefficients for aluminium as the surface,
        at different angles of incidence
        Values obtained from Wolfram demo project at (noting this is using 
        optical sign convention - switch the sign of r_p):
             Tayari Colemanand Anna Petrova-Mayor 
             "Fresnel Coefficients of Metals"
             http://demonstrations.wolfram.com/FresnelCoefficientsOfMetals/
             Wolfram Demonstrations Project
             Published: August 31, 2020 
        '''
        n_environment = 1
        n_surface = 0.965+6.399j
        incidence_angle = np.deg2rad(np.array(
            [0, 10, 20, 30, 40, 50, 60, 70, 80]))
        expected_r_p = np.array([
            -0.912293 - 0.285616j,
            -0.910291 - 0.289706j,
            -0.903869 - 0.302525j,
            -0.891575 - 0.325905j,
            -0.870092 - 0.363598j,
            -0.831861 - 0.422767j,
            -0.757610 - 0.517044j,
            -0.587749 - 0.669632j,
            -0.0914527 - 0.853458j
            ])
        expected_r_s = np.array([
            -0.912293 - 0.285616j,
            -0.914249 - 0.281575j,
            -0.919947 - 0.269512j,
            -0.928898 - 0.249618j,
            -0.940339 - 0.222237j,
            -0.953310 - 0.187908j,
            -0.966749 - 0.147392j,
            -0.979601 - 0.101696j,
            -0.990928 - 0.0520749j
            ])
        calculated_r_s, calculated_r_p = cl_calcs.reflection_coefficients(
            incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose(expected_r_s, calculated_r_s, atol=1e-6)
        np.testing.assert_allclose(expected_r_p, calculated_r_p, atol=1e-6)


    def test_numpy_array_refractive_index(self):
        '''
        test the fresnel reflection coefficients for aluminium as the surface,
        at different angles of incidence
        Values obtained from Wolfram demo project at (noting the demo project 
        is using optical sign convention - switch the sign of r_p):
             Tayari Colemanand Anna Petrova-Mayor 
             "Fresnel Coefficients of Metals"
             http://demonstrations.wolfram.com/FresnelCoefficientsOfMetals/
             Wolfram Demonstrations Project
             Published: August 31, 2020 
        '''
        n_environment = 1
        n_surface = np.array([0.965+6.399j])
        incidence_angle = np.deg2rad(np.array(
            [0, 10, 20, 30, 40, 50, 60, 70, 80]))
        expected_r_p = np.array([
            -0.912293 - 0.285616j,
            -0.910291 - 0.289706j,
            -0.903869 - 0.302525j,
            -0.891575 - 0.325905j,
            -0.870092 - 0.363598j,
            -0.831861 - 0.422767j,
            -0.757610 - 0.517044j,
            -0.587749 - 0.669632j,
            -0.0914527 - 0.853458j
            ])
        expected_r_s = np.array([
            -0.912293 - 0.285616j,
            -0.914249 - 0.281575j,
            -0.919947 - 0.269512j,
            -0.928898 - 0.249618j,
            -0.940339 - 0.222237j,
            -0.953310 - 0.187908j,
            -0.966749 - 0.147392j,
            -0.979601 - 0.101696j,
            -0.990928 - 0.0520749j
            ])
        calculated_r_s, calculated_r_p = cl_calcs.reflection_coefficients(
            incidence_angle, n_surface, n_environment)
        np.testing.assert_allclose(expected_r_s, calculated_r_s, atol=1e-6)
        np.testing.assert_allclose(expected_r_p, calculated_r_p, atol=1e-6)

class ReflectedETest(parameterized.TestCase):
    '''
    Test the calculation of the reflected electric field
    '''
    
    @parameterized.named_parameters(
        ('s-polarized', np.array([[0, 0, 1]]), np.array([[0, 0, -0.451416229]]), 
            np.array([[0, 0, 0]])),
        ('p-polarized', np.array([[1, -1, 0]]), np.array([[0, 0, 0]]), 
            0.203776612*np.array([[-1, 1, 0]])),
        ('mixed-polarized', np.array([[1, -1, 1]]), 
            np.array([[0, 0, -0.451416229]]), 0.203776612*np.array([[-1, 1, 0]])),
        ('3 by 3 array', np.array([[0, 0, 1], [1, -1, 0], [1, -1, 1]]), 
            np.array([[0, 0, -0.451416229], [0, 0, 0], [0, 0, -0.451416229]]), 
            0.203776612 * np.array([[0, 0, 0], [-1, 1, 0], [-1, 1, 0]]))
    )
    def test_e_polarization_state(self, incident_e, expected_e_s, expected_e_p):
        '''
        An input electric field of various polarization states
        '''
        normal = np.array([1, 0, 0])
        incident_direction = np.array([1, 1, 0])
        n_surface = 2
        n_environment = 1
        e_s, e_p = cl_calcs.reflected_e(
            incident_direction,
            incident_e,
            normal,
            n_surface,
            n_environment
            )
        np.testing.assert_allclose(e_s, expected_e_s)
        np.testing.assert_allclose(e_p, expected_e_p)

    @parameterized.named_parameters(
        ('s-polarized', np.array([[0, 0, 1]]), np.array([[0, 0, -0.451416229]]), 
            np.array([[0, 0, 0]])),
        ('p-polarized', np.array([[1, -1, 0]]), np.array([[0, 0, 0]]), 
            0.203776612*np.array([[-1, 1, 0]])),
        ('mixed-polarized', np.array([[1, -1, 1]]), 
            np.array([[0, 0, -0.451416229]]), 0.203776612*np.array([[-1, 1, 0]])),
    )
    def test_e_polarization_state_negative_normal(self, incident_e, expected_e_s, expected_e_p):
        '''
        An input electric field of various polarization states
        '''
        normal = np.array([-1, 0, 0])
        incident_direction = np.array([1, 1, 0])
        n_surface = 2
        n_environment = 1
        e_s, e_p = cl_calcs.reflected_e(
            incident_direction, 
            incident_e, 
            normal, 
            n_surface, 
            n_environment
            )
        np.testing.assert_allclose(e_s, expected_e_s)
        np.testing.assert_allclose(e_p, expected_e_p)

    @parameterized.named_parameters(
        ('s-polarized', np.array([[0, 0, 1]]), 
            np.array([[0, 0, -0.928898-0.249618j]]), np.array([[0, 0, 0]])),
        ('p-polarized', np.array([[1, -1, 0]]), np.array([[0, 0, 0]]), 
            np.array([[-0.891575-0.325905j, 0.891575+0.325905j, 0]])),
        ('mixed-polarized', np.array([[1, -1, 1]]), 
            np.array([[0, 0, -0.928898-0.249618j]]), 
            np.array([[-0.891575-0.325905j, 0.891575+0.325905j, 0]])),
        ('mixed-polarized not ones', np.array([[5, -3, 2]]), 
            np.array([[0, 0, -1.857796-0.499236j]]), 
            np.array([[-4.457875-1.629525j, 2.674725+0.977715j, 0]])),
        ('3 by 3 array', 
            np.array([[0, 0, 1], [1, -1, 0], [1, -1, 1]]), 
            np.array([[0, 0, -0.928898-0.249618j], 
                [0, 0, 0], 
                [0, 0, -0.928898-0.249618j]]), 
            np.array([[0, 0, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0]]))
    )
    def test_e_polarization_state_complex_surface(self, incident_e, expected_e_s, expected_e_p):
        '''
        An input electric field of various polarization states hits a surface
        with a complex refractive index
        '''
        normal = np.array([1, 0, 0])
        incident_direction = np.array([-0.8660254037844387, -0.5, 0])  # 30 deg
        n_surface = 0.965+6.399j  # Aluminium
        n_environment = 1
        e_s, e_p = cl_calcs.reflected_e(
            incident_direction,
            incident_e,
            normal,
            n_surface,
            n_environment
            )
        np.testing.assert_allclose(e_s, expected_e_s, atol=5e-6)
        np.testing.assert_allclose(e_p, expected_e_p, atol=5e-6)

    def test_output_shape_1x3normals(self):
        '''
        Check the output shape of the reflected fields is the same as the input shape
        '''
        incident_e = np.array([[0, 0, 1], [1, -1, 0], [1, -1, 1], [5, -3, 2]])
        expected_e_s = np.array([[0, 0, -0.928898-0.249618j], 
                [0, 0, 0], 
                [0, 0, -0.928898-0.249618j],
                [0, 0, -1.857796-0.499236j]])
        expected_e_p = np.array([[0, 0, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0],
                [-4.457875-1.629525j, 2.674725+0.977715j, 0]])
        normal = np.array([[1, 0, 0]])
        incident_direction = np.array([-0.8660254037844387, -0.5, 0])  # 30 deg
        n_surface = 0.965+6.399j  # Aluminium
        n_environment = 1
        e_s, e_p = cl_calcs.reflected_e(
            incident_direction,
            incident_e,
            normal,
            n_surface,
            n_environment
            )
        expected_shape = np.shape(incident_e)
        np.testing.assert_allclose(e_s, expected_e_s, atol=5e-6)
        np.testing.assert_allclose(e_p, expected_e_p, atol=5e-6)
        np.testing.assert_equal(np.shape(e_s), expected_shape)
        np.testing.assert_equal(np.shape(e_p), expected_shape)

    def test_output_shape_4x3normals(self):
        '''
        Check the output shape of the reflected fields is the same as the input shape
        '''
        incident_e = np.array([[0, 0, 1], [1, -1, 0], [1, -1, 1], [5, -3, 2]])
        expected_e_s = np.array([[0, 0, -0.928898-0.249618j], 
                [0, 0, 0], 
                [0, 0, -0.928898-0.249618j],
                [0, 0, -1.857796-0.499236j]])
        expected_e_p = np.array([[0, 0, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0], 
                [-0.891575-0.325905j, 0.891575+0.325905j, 0],
                [-4.457875-1.629525j, 2.674725+0.977715j, 0]])
        normal = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        incident_direction = np.array([-0.8660254037844387, -0.5, 0])  # 30 deg
        n_surface = 0.965+6.399j  # Aluminium
        n_environment = 1
        e_s, e_p = cl_calcs.reflected_e(
            incident_direction,
            incident_e,
            normal,
            n_surface,
            n_environment
            )
        expected_shape = np.shape(incident_e)
        np.testing.assert_equal(np.shape(e_s), expected_shape)
        np.testing.assert_equal(np.shape(e_p), expected_shape)
        np.testing.assert_allclose(e_s, expected_e_s, atol=5e-6)
        np.testing.assert_allclose(e_p, expected_e_p, atol=5e-6)

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


class NormalizeStokesParametersTest(unittest.TestCase):
    def test_normal(self):
        '''
        Test a normal Stokes vector
        '''
        S0 = np.array([3])
        S1 = np.array([1])
        S2 = np.array([2])
        S3 = np.array([-2])
        s1_expected, s2_expected, s3_expected = np.array([1/3, 2/3, -2/3])
        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
        np.testing.assert_array_almost_equal(
            np.array([s1, s2, s3]),
            np.array([[s1_expected], [s2_expected], [s3_expected]])
            )

    def test_most_zeros(self):
        '''
        Test a Stokes vector where all but S2 are 0
        '''
        S0 = np.array([2])
        S1 = np.array([0])
        S2 = np.array([2])
        S3 = np.array([0])
        s1_expected, s2_expected, s3_expected = np.array([0, 1, 0])
        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
        np.testing.assert_array_almost_equal(
            np.array([s1, s2, s3]),
            np.array([[s1_expected], [s2_expected], [s3_expected]])
            )

    def test_zero_stokes(self):
        '''
        Test a Stokes vector with all 0 components
        '''
        S0 = np.array([0])
        S1 = np.array([0])
        S2 = np.array([0])
        S3 = np.array([0])
        s1_expected, s2_expected, s3_expected = np.array([0, 0, 0])
        s1, s2, s3 = cl_calcs.normalize_stokes_parameters(S0, S1, S2, S3)
        np.testing.assert_array_almost_equal(
            np.array([s1, s2, s3]),
            np.array([[s1_expected], [s2_expected], [s3_expected]])
            )


class DielectricToRefractiveTest(parameterized.TestCase):
    '''
    normal dielectric function
    no imaginary component
    no real component
    positive real and imaginary
    negative real and imaginary
    positive real, negative imaginary
    negative real, positive imaginary
    all zeros
    1 by 2 array
    3 by 2 array
    2 array
    '''
    @parameterized.named_parameters(
        dict(testcase_name='+ real, + imag',
            dielectric = np.array([[1, 2]]),
            expected_n = np.array([1.272019649514069 + 0.7861513777574233j])
            ),
        dict(testcase_name='+ real, - imag',
            dielectric = np.array([[1, -2]]),
            expected_n = np.array([1.272019649514069 + 0.7861513777574233j])
            ),
        dict(testcase_name='- real, + imag',
            dielectric = np.array([[-1, 2]]),
            expected_n = np.array([0.7861513777574233 + 1.272019649514069j])
            ),
        dict(testcase_name='- real, - imag',
            dielectric = np.array([[-1, -2]]),
            expected_n = np.array([0.7861513777574233 + 1.272019649514069j])
            ),
        dict(testcase_name='0 real, - imag',
            dielectric = np.array([[0, -2]]),
            expected_n = np.array([1 + 1j])
            ),
        dict(testcase_name='- real, 0 imag',
            dielectric = np.array([[1, 0]]),
            expected_n = np.array([1 + 0j])
            ),
        dict(testcase_name='0 real, 0 imag',
            dielectric = np.array([[0, 0]]),
            expected_n = np.array([0 + 0j])
            ),
        dict(testcase_name='3 by 2 array',
            dielectric = np.array([[-1, -2], [0, -2], [1, 2]]),
            expected_n = np.array([
                0.7861513777574233 + 1.272019649514069j,
                1 + 1j,
                1.272019649514069 + 0.7861513777574233j
                ])
            ),
        )
    def test_values(self, dielectric, expected_n):
        '''
        Test an assortment of dielectric values that may be encountered
        '''
        calculated_n = cl_calcs.dielectric_to_refractive(dielectric)
        np.testing.assert_allclose(expected_n, calculated_n)

    def test_two_value_array(self):
        '''
        test an array of size 2 - should fail
        '''
        dielectric = np.array([1, 4])
        with self.assertRaises(IndexError):
            calculated_n = cl_calcs.dielectric_to_refractive(dielectric)


class eVToWavelengthTest(parameterized.TestCase):
    '''
    positive number
    real number
    negative number
    wavelength
    very large
    very small
    '''
    @parameterized.named_parameters(
        ('0 eV', np.array([0]), np.array([np.inf])),
        ('0.5 eV', 0.5, 2.479683968664e-6),
        ('3 eV', 3, 4.1328066144400093e-07),
        ('-3 eV', -3, -4.1328066144400093e-07),
        ('1e-6 eV', 1e-6, 1.2398419843320025),
        ('1e6 eV', 1e6, 1.2398419843320027e-12),
            )
    def test_single_values(self, eV, expected_nm):
        calculated_nm = cl_calcs.eV_to_wavelength(eV)
        self.assertAlmostEqual(expected_nm, calculated_nm)


class WavelengthToeVTest(parameterized.TestCase):
    '''
    positive number
    real number
    negative number
    wavelength
    very large
    very small
    '''
    @parameterized.named_parameters(
        ('0 eV', np.array([np.inf]), np.array([0])),
        ('3 eV', 4.1328066144400093e-07, 3),
        ('-3 eV', -4.1328066144400093e-07, -3),
        ('1e-6 eV', 1.2398419843320025, 1e-6),
        ('1e6 eV', 1.2398419843320027e-12, 1e6),
            )
    def test_single_values(self, nm, expected_eV):
        calculated_eV = cl_calcs.wavelength_to_eV(nm)
        np.testing.assert_allclose(expected_eV, calculated_eV)

if __name__ == '__main__':
    if 'unittest.util' in __import__('sys').modules:
    # Show full diff in self.assertEqual.
        __import__('sys').modules['unittest.util']._MAX_LENGTH = 999999999
    unittest.main()
