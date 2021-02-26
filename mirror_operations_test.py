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
        ('negative X axis', 
            np.array([[-1, 0, 0]]), 
            np.array([[1, 0, 0]])
        ),
        ('positive Y axis', 
            np.array([[0, 1 , 0]]), 
            1/np.sqrt(1+1/25*5**2)*np.array([[1, -1/5*5, 0]])
        ),
        ('negative Y axis',
            np.array([[0, -1, 0]]),
            1/np.sqrt(1+1/25*5**2)*np.array([[1, 1/5*5, 0]])
        ), 
        ('positive Z axis', 
            np.array([[0, 0, 1]]), 
            1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, -1/5*5]])
        ),
        ('negative Z axis',
            np.array([[0, 0, -1]]),
            1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, 1/5*5]])
        ), 
        ('2 by 3 array input',
            np.array([[0, 0, -1], [0, 0, 1]]),
            1/np.sqrt(1+1/25*5**2)*np.array([[1, 0, 1/5*5], [1, 0, -1/5*5]])
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
        ('negative X axis', 
            np.array([[-1, 0, 0]]), 
            np.array([[1, 0, 0]])
        ),
        ('positive Y axis', 
            np.array([[0, 1, 0]]), 
            1/np.sqrt(1+1/25)*np.array([[1, -1/5, 0]])
        ),
        ('negative Y axis', 
            np.array([[0, -1, 0]]), 
            1/np.sqrt(1+1/25)*np.array([[1, 1/5, 0]])
        ),
        ('positive Z axis', 
            np.array([[0, 0, 1]]), 
            1/np.sqrt(1+1/25)*np.array([[1, 0, -1/5]])
        ),
        ('negative Z axis', 
            np.array([[0, 0, -1]]), 
            1/np.sqrt(1+1/25)*np.array([[1, 0, 1/5]])
        ),
        ('positive Y, positive Z',
            np.array([[0, 1, 3]]),
            1/np.sqrt(1+10/25)*np.array([[1, -1/5, -3/5]])
        ),
        ('positive Y, negative Z',
            np.array([[0, 1, -3]]),
            1/np.sqrt(1+10/25)*np.array([[1, -1/5, 3/5]])
        ),
        ('negative Y, positive Z',
            np.array([[0, -5, 3]]),
            1/np.sqrt(1+34/25)*np.array([[1, 1, -3/5]])
        ),
        ('negative Y, negative Z',
            np.array([[0, -5, -3]]),
            1/np.sqrt(1+34/25)*np.array([[1, 1, 3/5]])
        ),
    )
    def test_parabola_normals_positions(self, position, expected_normal):
        '''
        check that the parabola normals are calculated correctly given various
        (y,z) positions on the parameterized parabola
        '''
        normal = miop.parabola_normals(position)
        np.testing.assert_allclose(normal, expected_normal, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
