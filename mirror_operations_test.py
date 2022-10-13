from absl.testing import parameterized
import numpy as np
import  mirror_operations as miop
import unittest

'''
parabola position:
- points to positive x
- position at origin
- (-2.5, 0, 0) on parabola
- (+x, 0, 0) doesn't exist - is NaN
- 

parabola normals:
- at r = 0, unit normal should point along x
- all normals point in the same direction
- all normals should be unit vectors
'''

class ParabolaPositionTest(parameterized.TestCase):
    '''
    Test the parabola_position function in mirror_operations
    '''
    @parameterized.named_parameters(
        ('negative X axis', np.array([[-1, 0, 0]]), np.array([[-2.5, 0, 0]])),
        ('positive Y axis', np.array([[0, 1, 0]]), np.array([[0, 5, 0]])),
        ('negative Y axis', np.array([[0, -1, 0]]), np.array([[0, -5, 0]])),
        ('positive Z axis', np.array([[0, 0, 1]]), np.array([[0, 0, 5]])),
        ('negative Z axis', np.array([[0, 0, -1]]), np.array([[0, 0, -5]])),
        ('off axis', np.array([[-2.3, 1, 1]]), np.array([[-2.3, 1, 1]])),
        ('2x3 input', np.array([[0, 1, 0], [0, 0, 1]]), np.array([[0, 5, 0], [0, 0, 5]])),
#        ('three-vector input', np.array([-1, 0, 0]), np.array([-2.5, 0, 0]))
#        ('2x2x3 input',
#         np.array([[[0, 1, 0], [0, 0, 1]], [[0, -1, 0], [0, 0, -1]]]),
#         np.array([[[0, 5, 0], [0, 0, 5]], [[0, -5, 0], [0, 0, -5]]]),
#         ),
    )
    def test_parabola_positions(self, direction, expected_position):
        '''
        check that parabola positions are calculated correctly
        '''
        position = miop.parabola_position(direction)
        np.testing.assert_allclose(expected_position, position, atol=1e-7)

    def test_parabola_position_positive_X(self):
        '''
        check the parabola positions returns NaN on positive x axis
        '''
        xyz = np.array([[1, 0, 0]])
        position_expected = np.array([[np.nan, np.nan, np.nan]])
        position = miop.parabola_position(xyz);
        self.assertTrue(np.all(np.isnan(position)))

class ParabolaNormalsTest(parameterized.TestCase):
    '''
    Test the parabola_normals function in mirror_operations
    '''
    def test_negative_x_axis(self):
        '''
        Check that the parabola normal on the negative x-axis points along the 
        positive x-axis
        '''
        direction = np.array([[-1, 0, 0]])
        parabola_position = miop.parabola_position(direction)
        calculated_normal = miop.parabola_normals(parabola_position)
        expected_normal = np.array([[1, 0, 0]])
        np.testing.assert_allclose(expected_normal, calculated_normal, atol=1e-7)

    @parameterized.named_parameters(
        dict(testcase_name='negative X axis',
             direction=np.array([[-1, 0, 0]]),
             expected_normal=np.array([[1, 0, 0]])
        ),
        dict(testcase_name='positive Y axis',
             direction=np.array([[0, 1 , 0]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[1, -1/5*5, 0]])
        ),
        dict(testcase_name='negative Y axis',
             direction=np.array([[0, -1, 0]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[1, 1/5*5, 0]])
        ), 
        dict(testcase_name='positive Z axis',
             direction=np.array([[0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, -1/5*5]])
        ),
        dict(testcase_name='negative Z axis',
             direction=np.array([[0, 0, -1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, 1/5*5]])
        ), 
        dict(testcase_name='2 by 3 array input',
             direction=np.array([[0, 0, -1], [0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, 1/5*5], [1, 0, -1/5*5]])
        ),
    )
    def test_parabola_normals_directions(self, direction, expected_normal):
        '''
        check that the parabola normals are calculated correctly in various
        directions
        '''
        position = miop.parabola_position(direction)
        normal = miop.parabola_normals(position)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-7)

    @parameterized.named_parameters(
        dict(testcase_name='negative X axis',
             position=np.array([[-1, 0, 0]]),
             expected_normal=np.array([[1, 0, 0]])
        ),
        dict(testcase_name='positive Y axis',
             position=np.array([[0, 1, 0]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[1, -1/5, 0]])
        ),
        dict(testcase_name='negative Y axis',
             position=np.array([[0, -1, 0]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[1, 1/5, 0]])
        ),
        dict(testcase_name='positive Z axis',
             position=np.array([[0, 0, 1]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[1, 0, -1/5]])
        ),
        dict(testcase_name='negative Z axis',
             position=np.array([[0, 0, -1]]),
             expected_normal=1/np.sqrt(1+1/25)*np.array([[1, 0, 1/5]])
        ),
        dict(testcase_name='positive Y, positive Z',
             position=np.array([[0, 1, 3]]),
             expected_normal=1/np.sqrt(1+10/25)*np.array([[1, -1/5, -3/5]])
        ),
        dict(testcase_name='positive Y, negative Z',
             position=np.array([[0, 1, -3]]),
             expected_normal=1/np.sqrt(1+10/25)*np.array([[1, -1/5, 3/5]])
        ),
        dict(testcase_name='negative Y, positive Z',
             position=np.array([[0, -5, 3]]),
             expected_normal=1/np.sqrt(1+34/25)*np.array([[1, 1, -3/5]])
        ),
        dict(testcase_name='negative Y, negative Z',
             position=np.array([[0, -5, -3]]),
             expected_normal=1/np.sqrt(1+34/25)*np.array([[1, 1, 3/5]])
        ),
    )
    def test_parabola_normals_positions(self, position, expected_normal):
        '''
        check that the parabola normals are calculated correctly given various
        (y,z) positions on the parameterized parabola
        '''
        normal = miop.parabola_normals(position)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-7)

class ParabolaSurfacePolarizationTest(parameterized.TestCase):
    '''
    Test the surface polarization direction for parabolas
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive X axis',
             theta=np.array([np.pi/2]),
             phi=np.array([0]),
             expected_p=np.array([[0, 0, -1]]),
             expected_s=np.array([[0, 1, 0]]),
        ),
        dict(testcase_name='positive Y axis',
             theta=np.array([np.pi/2]),
             phi=np.array([np.pi/2]),
             expected_p=np.array([[0, 0, -1]]),
             expected_s=np.array([[-1, 0, 0]]),
        ),
        dict(testcase_name='negative X axis',
             theta=np.array([np.pi/2]),
             phi=np.array([np.pi]),
             expected_p=np.array([[0, 0, -1]]),
             expected_s=np.array([[0, -1, 0]]),
        ), 
        dict(testcase_name='negative Y axis', 
             theta=np.array([np.pi/2]),
             phi=np.array([3*np.pi/2]),
             expected_p=np.array([[0, 0, -1]]),
             expected_s=np.array([[1, 0, 0]]),
        ),
        dict(testcase_name='negative Y axis backwards',
             theta=np.array([np.pi/2]),
             phi=np.array([-np.pi/2]),
             expected_p=np.array([[0, 0, -1]]),
             expected_s=np.array([[1, 0, 0]]),
        ),
        dict(testcase_name='45 degrees off positive z',
             theta=np.array([np.pi/4]),
             phi=np.array([np.pi/2]),
             expected_p=np.array([[0, 1/np.sqrt(2), -1/np.sqrt(2)]]),
             expected_s=np.array([[-1, 0, 0]]),
        ),
    )
    def test_surface_polarization_directions(self,
                                             theta,
                                             phi,
                                             expected_p,
                                             expected_s):
        p, s = miop.surface_polarization_directions(theta, phi)
        np.testing.assert_allclose(p, expected_p, atol=1e-7)
        np.testing.assert_allclose(s, expected_s, atol=1e-7)


class FresnelReflectionCoefficientsTest(parameterized.TestCase):
    '''
    Test the calculation of Fresnel reflection coefficients
    '''
    @parameterized.named_parameters(
        dict(testcase_name='+x, -y',
             k_vector=np.array([1, -1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=0.203776612,
             expected_r_p=0.041524907,
        ),
        dict(testcase_name='-x, -y',
             k_vector=np.array([-1, -1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=0.203776612,
             expected_r_p=0.041524907,
        ),
        dict(testcase_name='+x, +y',
             k_vector=np.array([1, 1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=0.203776612,
             expected_r_p=0.041524907,
        ),
        dict(testcase_name='-x, +y',
             k_vector=np.array([-1, 1, 0])/np.sqrt(2),
             normal=np.array([1, 0, 0]),
             expected_r_s=0.203776612,
             expected_r_p=0.041524907,
        ),
        dict(testcase_name='Brewster angle',
             k_vector=np.array([2, -1, 0]),
             normal=np.array([0, 1, 0]),
             expected_r_s=0.36,
             expected_r_p=0,
        ),
        dict(testcase_name='not nice axis',
             k_vector=np.array([1, 1, 3]),
             normal=np.array([1, 2.5, 1]),
             expected_r_s=0.21489423472,
             expected_r_p=0.03572184881,
        ),
    )
    def test_incoming_vector_angles(self, k_vector, normal, expected_r_s, expected_r_p):
        '''
        Incoming vector impinges from different quadrants
        '''
        n_mirror = 2
        n_environment = 1
        r_s, r_p = miop.fresnel_reflection_coefficients(normal, k_vector, n_mirror, n_environment)
        np.testing.assert_allclose(r_s, expected_r_s, atol=1e-7)
        np.testing.assert_allclose(r_p, expected_r_p, atol=1e-7)

    def test_mirror_refractive_index_is_zero(self):
        '''
        the refractive index of the mirror is 0
        '''
        n_mirror = 0
        n_environment = 1
        k_vector=np.array([1, -1, 0])/np.sqrt(2)
        normal=np.array([1, 0, 0])
        self.assertRaisesRegex(
            ValueError,
            "Mirror refractive index cannot be 0",
            miop.fresnel_reflection_coefficients,
            normal,
            k_vector,
            n_mirror,
            n_environment)

if __name__ == '__main__':
    unittest.main()
