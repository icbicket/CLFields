import coord_transforms
import unittest
import numpy as np
from absl.testing import parameterized

class PolarCartesianTransformTest(parameterized.TestCase):
    '''
    x, y axes
    some intermediate values at non 90 degree angles
    Check zero values
    Single value
    Array of values
    Negative angles
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis',
            r = 1,
            phi = 0,
            expected_x = 1,
            expected_y = 0,
            ),
        dict(testcase_name='positive y-axis',
            r = 1,
            phi = np.pi/2,
            expected_x = 0,
            expected_y = 1,
            ),
        dict(testcase_name='negative x-axis',
            r = 1,
            phi = np.pi,
            expected_x = -1,
            expected_y = 0,
            ),
        dict(testcase_name='negative y-axis',
            r = 1,
            phi = 3*np.pi/2,
            expected_x = 0,
            expected_y = -1,
            ),
        dict(testcase_name='negative y-axis non-integer',
            r = 12.36,
            phi = 3*np.pi/2,
            expected_x = 0,
            expected_y = -12.36,
            ),
        dict(testcase_name='positive x-axis 2 pi',
            r = 1,
            phi = 2*np.pi,
            expected_x = 1,
            expected_y = 0,
            ),
        dict(testcase_name='positive y-axis +2 pi',
            r = 1,
            phi = 2*np.pi+np.pi/2,
            expected_x = 0,
            expected_y = 1,
            ),
        dict(testcase_name='+45 degrees',
            r = 2,
            phi = np.pi/4,
            expected_x = np.sqrt(2),
            expected_y = np.sqrt(2),
            ),
        dict(testcase_name='+135 degrees',
            r = 2,
            phi = np.pi/2 + np.pi/4,
            expected_x = -np.sqrt(2),
            expected_y = np.sqrt(2),
            ),
        dict(testcase_name='+225 degrees',
            r = 2,
            phi = np.pi + np.pi/4,
            expected_x = -np.sqrt(2),
            expected_y = -np.sqrt(2),
            ),
        dict(testcase_name='+315 degrees',
            r = 2,
            phi = 3*np.pi/2 + np.pi/4,
            expected_x = np.sqrt(2),
            expected_y = -np.sqrt(2),
            ),
        dict(testcase_name='-45 degrees',
            r = 2,
            phi = -np.pi/4,
            expected_x = np.sqrt(2),
            expected_y = -np.sqrt(2),
            ),
        dict(testcase_name='-135 degrees',
            r = 2,
            phi = -np.pi/2 - np.pi/4,
            expected_x = -np.sqrt(2),
            expected_y = -np.sqrt(2),
            ),
        dict(testcase_name='-225 degrees',
            r = 2,
            phi = -np.pi - np.pi/4,
            expected_x = -np.sqrt(2),
            expected_y = np.sqrt(2),
            ),
        dict(testcase_name='-315 degrees',
            r = 2,
            phi = -3*np.pi/2 - np.pi/4,
            expected_x = np.sqrt(2),
            expected_y = np.sqrt(2),
            ),
        dict(testcase_name='r=0',
            r = 0,
            phi = np.pi/4,
            expected_x = 0,
            expected_y = 0,
            ),
        )
    def test_single_values(self, r, phi, expected_x, expected_y):
        '''
        values along the x and y axes
        '''
        calculated_x, calculated_y = coord_transforms.polar_to_cartesian(r, phi)
        self.assertAlmostEqual(expected_x, calculated_x)
        self.assertAlmostEqual(expected_y, calculated_y)

    def test_array_of_values(self):
        '''
        a numpy array of values for r and phi
        '''
        r = np.array([1, 2, 3])
        phi = np.array([0, np.pi/4, -np.pi/3])
        expected_x = np.array([1, np.sqrt(2), 1.5])
        expected_y = np.array([0, np.sqrt(2), -np.sqrt(6.75)])
        calculated_x, calculated_y = coord_transforms.polar_to_cartesian(r, phi)
        np.testing.assert_allclose(expected_x, calculated_x)
        np.testing.assert_allclose(expected_y, calculated_y)


class CartesianPolarTransformTest(parameterized.TestCase):
    '''
    x, y axes
    some intermediate values at non 90 degree angles
    Check zero values
    Single value
    Array of values
    Negative angles
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis',
            x = 1,
            y = 0,
            expected_r = 1,
            expected_phi = 0,
            ),
        dict(testcase_name='positive y-axis',
            x = 0,
            y = 1,
            expected_r = 1,
            expected_phi = np.pi/2,
            ),
        dict(testcase_name='negative x-axis',
            x = -1,
            y = 0,
            expected_r = 1,
            expected_phi = np.pi,
            ),
        dict(testcase_name='negative y-axis',
            x = 0,
            y = -1,
            expected_r = 1,
            expected_phi = 3*np.pi/2,
            ),
        dict(testcase_name='negative y-axis non-integer',
            x = 0,
            y = -12.36,
            expected_r = 12.36,
            expected_phi = 3*np.pi/2,
            ),
        dict(testcase_name='+45 degrees',
            x = np.sqrt(2),
            y = np.sqrt(2),
            expected_r = 2,
            expected_phi = np.pi/4,
            ),
        dict(testcase_name='+135 degrees',
            x = -np.sqrt(2),
            y = np.sqrt(2),
            expected_r = 2,
            expected_phi = np.pi/2 + np.pi/4,
            ),
        dict(testcase_name='+225 degrees',
            x = -np.sqrt(2),
            y = -np.sqrt(2),
            expected_r = 2,
            expected_phi = np.pi + np.pi/4,
            ),
        dict(testcase_name='+315 degrees',
            x = np.sqrt(2),
            y = -np.sqrt(2),
            expected_r = 2,
            expected_phi = 3*np.pi/2 + np.pi/4,
            ),
        dict(testcase_name='-45 degrees',
            x = np.sqrt(2),
            y = -np.sqrt(2),
            expected_r = 2,
            expected_phi = -np.pi/4+2*np.pi,
            ),
        dict(testcase_name='-135 degrees',
            x = -np.sqrt(2),
            y = -np.sqrt(2),
            expected_r = 2,
            expected_phi = -np.pi/2 - np.pi/4 + 2*np.pi,
            ),
        dict(testcase_name='-225 degrees',
            x = -np.sqrt(2),
            y = np.sqrt(2),
            expected_r = 2,
            expected_phi = -np.pi - np.pi/4 + 2*np.pi,
            ),
        dict(testcase_name='-315 degrees',
            x = np.sqrt(2),
            y = np.sqrt(2),
            expected_r = 2,
            expected_phi = -3*np.pi/2 - np.pi/4 + 2*np.pi,
            ),
        )
    def test_single_values(self, x, y, expected_r, expected_phi):
        '''
        values along the x and y axes
        '''
        calculated_r, calculated_phi = coord_transforms.cartesian_to_polar(np.array([x]), np.array([y]))
        np.testing.assert_allclose(expected_r, calculated_r)
        np.testing.assert_allclose(expected_phi, calculated_phi)

    def test_r_0(self):
        '''
        r = 0, phi is undefined
        '''
        x = 0
        y = 0
        expected_r = 0
        expected_phi = 0
        calculated_r, calculated_phi = coord_transforms.cartesian_to_polar(np.array([x]), np.array([y]))
        np.testing.assert_allclose(expected_r, calculated_r)
        np.testing.assert_allclose(expected_phi, calculated_phi)

    def test_array_of_values(self):
        '''
        a numpy array of values for x and y
        '''
        x = np.array([1, np.sqrt(2), 1.5])
        y = np.array([0, np.sqrt(2), -np.sqrt(6.75)])
        expected_r = np.array([1, 2, 3])
        expected_phi = np.array([0, np.pi/4, -np.pi/3+2*np.pi])
        calculated_r, calculated_phi = coord_transforms.cartesian_to_polar(x, y)
        np.testing.assert_allclose(expected_r, calculated_r)
        np.testing.assert_allclose(expected_phi, calculated_phi)


class QuadrantSymmetryTest(unittest.TestCase):
    '''
    Check expand_quadrant_symmetry is behaving as expected
    Simple 2x2 array
    3x3 array (odd dimensions)
    3x4 (different length dimensions)
    '''
    def testQuadrant2x2ArrayQ1(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 1
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[0, 1, 0], [2, 3, 2], [0, 1, 0]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 1)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ2(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 2
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[1, 0, 1], [3, 2, 3], [1, 0, 1]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 2)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ3(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 3
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[2, 3, 2], [0, 1, 0], [2, 3, 2]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 3)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ4(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 4
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[3, 2, 3], [1, 0, 1], [3, 2, 3]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 4)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant3x3ArrayQ1(self):
        '''
        3x3 array symmetrizes properly
        input quadrant 1
        '''
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        arrayfull = np.array([
                            [0, 1, 2, 1, 0], 
                            [3, 4, 5, 4, 3], 
                            [6, 7, 8, 7, 6],
                            [3, 4, 5, 4, 3],
                            [0, 1, 2, 1, 0]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 1)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant3x3ArrayQ2(self):
        '''
        3x3 array symmetrizes properly
        input quadrant 2
        '''
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        arrayfull = np.array([
                            [2, 1, 0, 1, 2], 
                            [5, 4, 3, 4, 5], 
                            [8, 7, 6, 7, 8],
                            [5, 4, 3, 4, 5],
                            [2, 1, 0, 1, 2]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 2)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant3x3ArrayQ3(self):
        '''
        3x3 array symmetrizes properly
        input quadrant 3
        '''
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        arrayfull = np.array([
                            [6, 7, 8, 7, 6], 
                            [3, 4, 5, 4, 3], 
                            [0, 1, 2, 1, 0],
                            [3, 4, 5, 4, 3],
                            [6, 7, 8, 7, 6]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 3)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant3x3ArrayQ4(self):
        '''
        3x3 array symmetrizes properly
        input quadrant 4
        '''
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        arrayfull = np.array([
                            [8, 7, 6, 7, 8], 
                            [5, 4, 3, 4, 5], 
                            [2, 1, 0, 1, 2],
                            [5, 4, 3, 4, 5],
                            [8, 7, 6, 7, 8]])
        testarray = coord_transforms.expand_quadrant_symmetry(array, 4)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

class CartesianSphericalCoordinateTransformTest(unittest.TestCase):
    '''
    (x, y, z) = (0, 1, 0)
    '''
    def test010(self):
        '''
        single element xyz vector
        '''
        xyz = np.array([[0, 1, 0]])
        r, theta, phi = coord_transforms.cartesian_to_spherical_coords(xyz)
        rthph = np.array([[1], [np.pi/2], [np.pi/2]])
        np.testing.assert_array_almost_equal(np.array([r, theta, phi]), rthph)

    def testmulti(self):
        '''
        multi-element xyz vectors
        '''
        xyz = np.array([
                    [1, 1, 0], 
                    [1, -1, 0], 
                    [-1, 1, 0], 
                    [0, 1, 1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [0, -1, -1]
                    ])
        r, theta, phi = coord_transforms.cartesian_to_spherical_coords(xyz)
        rthph = np.array([
                    [np.sqrt(2), np.pi/2, np.pi/4],
                    [np.sqrt(2), np.pi/2, 7*np.pi/4],
                    [np.sqrt(2), np.pi/2, 3*np.pi/4],
                    [np.sqrt(2), np.pi/4, np.pi/2],
                    [np.sqrt(2), 3*np.pi/4, np.pi/2],
                    [np.sqrt(2), np.pi/4, 3*np.pi/2],
                    [np.sqrt(2), 3*np.pi/4, 3*np.pi/2]
                        ]);
        np.testing.assert_array_almost_equal(np.transpose(np.array([r, theta, phi])), rthph)

class RotateVectorTest(unittest.TestCase):
    def test010rot001by90(self):
        '''
        take (0,1,0) and rotate around (0,0,1) by pi/2
        '''
        xyz = np.array([0, 1, 0])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([-1, 0, 0]), rotated_vector)

    def test001rot001by90(self):
        '''
        take (0,0,1) and rotate around (0,0,1) by pi/2
        '''
        xyz = np.array([0, 0, 1])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([0, 0, 1]), rotated_vector)

    def test000rotError(self):
        '''
        take (0,0,0) and throw an error
        '''
        xyz = np.array([0, 0, 0])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        self.assertRaises(ValueError, coord_transforms.rotate_vector, xyz, angle, rotation_vector)

class RotateNdVectorTest(unittest.TestCase):
    def testAxisVectorsRotateBy90(self):
        '''
        take (0,1,0), (1, 0, 0) and (0,0,1) and rotate around (0,0,1) by pi/2
        '''
        xyz = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]]), rotated_vector)
    
    def testFloatsVectorsRotateBy90(self):
        '''
        take (0.707, 0.707, 0), (1, 1, 1) and rotate around (0,0,1) by pi/2
        '''
        xyz = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],[1.1, 1.1, 1.1]])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([[-1/np.sqrt(2), 1/np.sqrt(2), 0], [-1.1, 1.1, 1.1]]), rotated_vector)
    
    def testNegativeAngle(self):
        '''
        rotating a vector by -pi/2 vs rotating by pi/2
        '''
        xyz = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],[1.1, 1.1, 1.1]])
        angle = np.pi/2
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        rotated_vector_negative = coord_transforms.rotate_vector_Nd(xyz, -angle, rotation_vector)
        np.testing.assert_array_almost_equal(rotated_vector_negative, np.array([[-1, -1, 1]])*rotated_vector)
    
    def testAnglesGreater2Pi(self):
        '''
        rotating a vector by an angle greater than 2pi
        '''
        xyz = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],[1.1, 1.1, 1.1]])
        angle = np.pi/2+2*np.pi
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([[-1/np.sqrt(2), 1/np.sqrt(2), 0], [-1.1, 1.1, 1.1]]), rotated_vector)
    
    def testAnglesMultiplesOfPi(self):
        '''
        rotating a vector by an angle that is a multiple of pi
        '''
        xyz = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],[1.1, 1.1, 1.1]])
        angle = np.pi
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([-1, -1, 1])*xyz, rotated_vector)
    
    def testRotationAxisHighMagnitude(self):
        '''
        the input rotation axis has a magnitude that is not 1
        '''
        xyz = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],[1.1, 1.1, 1.1]])
        angle = np.pi
        rotation_vector = np.array([0,0,3])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([-1, -1, 1])*xyz, rotated_vector)
    
    def testTiltedRotationAxis(self):
        '''
        the rotation axis is tilted off one of the main axes
        '''
        xyz = np.array([[0, 0, 1]])
        angle = np.pi/2
        rotation_vector = np.array([1, 1, 0])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0]]), rotated_vector)
    
    def testNegativeRotationAxis(self):
        '''
        rotating around the rotation axis vs the negative rotation axis should rotate in opposite directions
        '''
        xyz = np.array([[0, 0, 1]])
        angle = np.pi/2
        rotation_vector = np.array([1, 1, 0])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        rotated_vector_negative = coord_transforms.rotate_vector_Nd(xyz, angle, -rotation_vector)
        np.testing.assert_array_almost_equal(rotated_vector_negative, -rotated_vector)


if __name__ == '__main__':
    unittest.main()
