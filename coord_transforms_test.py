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
        dict(testcase_name='zero vector',
            r = 0,
            phi = 0,
            expected_x = 0,
            expected_y = 0,
            )
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
        dict(testcase_name='zeros',
            x = 0,
            y = 0,
            expected_r = 0,
            expected_phi = 0,
            )
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

class CartesianSphericalCoordinateTransformTest(parameterized.TestCase):
    '''
    x, y, z axes
    one for each octant
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis',
            xyz = np.array([[1, 0, 0]]),
            expected_r_theta_phi = np.array([1, np.pi/2, 0]),
            ),
        dict(testcase_name='positive y-axis',
            xyz = np.array([[0, 1, 0]]),
            expected_r_theta_phi = np.array([1, np.pi/2, np.pi/2]),
            ),
        dict(testcase_name='negative x-axis',
            xyz = np.array([[-1, 0, 0]]),
            expected_r_theta_phi = np.array([1, np.pi/2, np.pi]),
            ),
        dict(testcase_name='negative y-axis',
            xyz = np.array([[0, -1, 0]]),
            expected_r_theta_phi = np.array([1, np.pi/2, 3*np.pi/2]),
            ),
        dict(testcase_name='positive z-axis',
            xyz = np.array([[0, 0, 1]]),
            expected_r_theta_phi = np.array([1, 0, 0]),
            ),
        dict(testcase_name='negative z-axis',
            xyz = np.array([[0, 0, -1]]),
            expected_r_theta_phi = np.array([1, np.pi, 0]),
            ),
        dict(testcase_name='positive xyz',
            xyz = np.array([[1, 1, 0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.2309594173407747, 0.7853981633974483]),
            ),
        dict(testcase_name='positive xy negative z',
            xyz = np.array([[1, 1, -0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.9106332362490184, 0.7853981633974483]),
            ),
        dict(testcase_name='positive xz negative y',
            xyz = np.array([[1, -1, 0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.2309594173407747, 5.497787143782138]),
            ),
        dict(testcase_name='positive yz negative x',
            xyz = np.array([[-1, 1, 0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.2309594173407747, 2.356194490192345]),
            ),
        dict(testcase_name='positive x negative yz',
            xyz = np.array([[1, -1, -0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.9106332362490184, 5.497787143782138]),
            ),
        dict(testcase_name='positive y negative xz',
            xyz = np.array([[-1, 1, -0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.9106332362490184, 2.356194490192345]),
            ),
        dict(testcase_name='positive z negative xy',
            xyz = np.array([[-1, -1, 0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.2309594173407747, 3.9269908169872414]),
            ),
        dict(testcase_name='negative xyz',
            xyz = np.array([[-1, -1, -0.5]]),
            expected_r_theta_phi = np.array([np.sqrt(2.25), 1.9106332362490184, 3.9269908169872414]),
            ),
        dict(testcase_name='zero vector',
            xyz = np.array([[0, 0, 0]]),
            expected_r_theta_phi = np.array([0, np.pi/2, 0])
            )
        )
    def test_axes(self, xyz, expected_r_theta_phi):
        calculated_r, calculated_theta, calculated_phi = coord_transforms.cartesian_to_spherical_coords(xyz)
        np.testing.assert_array_almost_equal(
            expected_r_theta_phi, 
            np.squeeze(np.array([calculated_r, calculated_theta, calculated_phi])))

    def test010(self):
        '''
        single element xyz vector
        (x, y, z) = (0, 1, 0)
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


class SphericalCartesianCoordinateTransformTest(parameterized.TestCase):
    '''
    x, y, z axes
    one for each octant
    Spherical to Cartesian coordinate transforms
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis',
            r_theta_phi = np.array([[1, np.pi/2, 0]]),
            expected_xyz = np.array([1, 0, 0]),
            ),
        dict(testcase_name='positive y-axis',
            r_theta_phi = np.array([[1, np.pi/2, np.pi/2]]),
            expected_xyz = np.array([0, 1, 0]),
            ),
        dict(testcase_name='negative x-axis',
            r_theta_phi = np.array([[1, np.pi/2, np.pi]]),
            expected_xyz = np.array([-1, 0, 0]),
            ),
        dict(testcase_name='negative y-axis',
            r_theta_phi = np.array([[1, np.pi/2, 3*np.pi/2]]),
            expected_xyz = np.array([0, -1, 0]),
            ),
        dict(testcase_name='positive z-axis',
            r_theta_phi = np.array([[1, 0, 0]]),
            expected_xyz = np.array([0, 0, 1]),
            ),
        dict(testcase_name='negative z-axis',
            r_theta_phi = np.array([[1, np.pi, 0]]),
            expected_xyz = np.array([0, 0, -1]),
            ),
        dict(testcase_name='positive xyz',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.2309594173407747, 0.7853981633974483]]),
            expected_xyz = np.array([1, 1, 0.5]),
            ),
        dict(testcase_name='positive xy negative z',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.9106332362490184, 0.7853981633974483]]),
            expected_xyz = np.array([1, 1, -0.5]),
            ),
        dict(testcase_name='positive xz negative y',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.2309594173407747, 5.497787143782138]]),
            expected_xyz = np.array([1, -1, 0.5]),
            ),
        dict(testcase_name='positive yz negative x',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.2309594173407747, 2.356194490192345]]),
            expected_xyz = np.array([-1, 1, 0.5]),
            ),
        dict(testcase_name='positive x negative yz',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.9106332362490184, 5.497787143782138]]),
            expected_xyz = np.array([1, -1, -0.5]),
            ),
        dict(testcase_name='positive y negative xz',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.9106332362490184, 2.356194490192345]]),
            expected_xyz = np.array([-1, 1, -0.5]),
            ),
        dict(testcase_name='positive z negative xy',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.2309594173407747, 3.9269908169872414]]),
            expected_xyz = np.array([-1, -1, 0.5]),
            ),
        dict(testcase_name='negative xyz',
            r_theta_phi = np.array([[np.sqrt(2.25), 1.9106332362490184, 3.9269908169872414]]),
            expected_xyz = np.array([-1, -1, -0.5]),
            ),
        dict(testcase_name='zero vector',
            r_theta_phi = np.array([[0, 0, 0]]),
            expected_xyz = np.array([0, 0, 0]),
            ),
            )
    def test_axes(self, r_theta_phi, expected_xyz):
        calculated_x, calculated_y, calculated_z = coord_transforms.spherical_to_cartesian_coords(r_theta_phi)
        np.testing.assert_array_almost_equal(
            expected_xyz, 
            np.squeeze(np.array([calculated_x, calculated_y, calculated_z])))
    
    def test010(self):
        '''
        single element xyz vector
        (x, y, z) = (0, 1, 0)
        '''
        expected_xyz = np.array([[0], [1], [0]])
        rthph = np.array([[1, np.pi/2, np.pi/2]])
        x, y, z = coord_transforms.spherical_to_cartesian_coords(rthph)
        np.testing.assert_array_almost_equal(np.array([x, y, z]), expected_xyz)

    def testmulti(self):
        '''
        multi-element xyz vectors
        '''
        expected_xyz = np.array([
                    [1, 1, 0], 
                    [1, -1, 0], 
                    [-1, 1, 0], 
                    [0, 1, 1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [0, -1, -1]
                    ])
        rthph = np.array([
                    [np.sqrt(2), np.pi/2, np.pi/4],
                    [np.sqrt(2), np.pi/2, 7*np.pi/4],
                    [np.sqrt(2), np.pi/2, 3*np.pi/4],
                    [np.sqrt(2), np.pi/4, np.pi/2],
                    [np.sqrt(2), 3*np.pi/4, np.pi/2],
                    [np.sqrt(2), np.pi/4, 3*np.pi/2],
                    [np.sqrt(2), 3*np.pi/4, 3*np.pi/2]
                        ])
        x, y, z = coord_transforms.spherical_to_cartesian_coords(rthph)
        np.testing.assert_array_almost_equal(np.transpose(np.array([x, y, z])), expected_xyz)


class CartesianSphericalVectorFieldTest(parameterized.TestCase):
    '''
    Testing the Cartesian to Spherical Vector Field coordinate transform 
    function
    vectors along each of x,y,z axes, positive and negative
    vector pointing into each quadrant
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis positive x-vector',
            theta = np.pi/2,
            phi = 0,
            fx = 1.2,
            fy = 0,
            fz = 0,
            expected_f = np.array([1.2, 0, 0]),
            ),
        dict(testcase_name='positive x-axis negative x-vector',
            theta = np.pi/2,
            phi = 0,
            fx = -1.2,
            fy = 0,
            fz = 0,
            expected_f = np.array([-1.2, 0, 0]),
            ),
        dict(testcase_name='positive x-axis positive y-vector',
            theta = np.pi/2,
            phi = 0,
            fx = 0,
            fy = 0.8,
            fz = 0,
            expected_f = np.array([0, 0, 0.8]),
            ),
        dict(testcase_name='positive x-axis negative y-vector',
            theta = np.pi/2,
            phi = 0,
            fx = 0,
            fy = -0.8,
            fz = 0,
            expected_f = np.array([0, 0, -0.8]),
            ),
        dict(testcase_name='positive x-axis positive z-vector',
            theta = np.pi/2,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = 0.3,
            expected_f = np.array([0, -0.3, 0]),
            ),
        dict(testcase_name='positive x-axis negative z-vector',
            theta = np.pi/2,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = -0.3,
            expected_f = np.array([0, 0.3, 0]),
            ),
        dict(testcase_name='positive x-axis positive xyz vector',
            theta = np.pi/2,
            phi = 0,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([1.2, -0.3, 0.8]),
            ),
        dict(testcase_name='positive x-axis positive xy negative z vector',
            theta = np.pi/2,
            phi = 0,
            fx = 1.2,
            fy = 0.8,
            fz = -0.3,
            expected_f = np.array([1.2, 0.3, 0.8]),
            ),
        dict(testcase_name='positive x-axis positive xz negative y vector',
            theta = np.pi/2,
            phi = 0,
            fx = 1.2,
            fy = -0.8,
            fz = 0.3,
            expected_f = np.array([1.2, -0.3, -0.8]),
            ),
        dict(testcase_name='positive x-axis positive yz negative x vector',
            theta = np.pi/2,
            phi = 0,
            fx = -1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-1.2, -0.3, 0.8]),
            ),
        dict(testcase_name='positive x-axis positive z negative xy vector',
            theta = np.pi/2,
            phi = 0,
            fx = -1.2,
            fy = -0.8,
            fz = 0.3,
            expected_f = np.array([-1.2, -0.3, -0.8]),
            ),
        dict(testcase_name='positive x-axis positive y negative xz vector',
            theta = np.pi/2,
            phi = 0,
            fx = -1.2,
            fy = 0.8,
            fz = -0.3,
            expected_f = np.array([-1.2, 0.3, 0.8]),
            ),
        dict(testcase_name='positive x-axis positive x negative yz vector',
            theta = np.pi/2,
            phi = 0,
            fx = 1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([1.2, 0.3, -0.8]),
            ),
        dict(testcase_name='positive x-axis negative xyz vector',
            theta = np.pi/2,
            phi = 0,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-1.2, 0.3, -0.8]),
            ),
        dict(testcase_name='positive y-axis positive y-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            fx = 0,
            fy = 0.8,
            fz = 0,
            expected_f = np.array([0.8, 0, 0]),
            ),
        dict(testcase_name='positive y-axis negative y-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            fx = 0,
            fy = -0.8,
            fz = 0,
            expected_f = np.array([-0.8, 0, 0]),
            ),
        dict(testcase_name='positive y-axis positive xyz-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.8, -0.3, -1.2]),
            ),
        dict(testcase_name='positive y-axis negative xyz-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.8, 0.3, 1.2]),
            ),
        dict(testcase_name='negative x-axis positive x-vector',
            theta = np.pi/2,
            phi = np.pi,
            fx = 1.2,
            fy = 0,
            fz = 0,
            expected_f = np.array([-1.2, 0, 0]),
            ),
        dict(testcase_name='negative x-axis negative x-vector',
            theta = np.pi/2,
            phi = np.pi,
            fx = -1.2,
            fy = 0,
            fz = 0,
            expected_f = np.array([1.2, 0, 0]),
            ),
        dict(testcase_name='negative x-axis positive xyz-vector',
            theta = np.pi/2,
            phi = np.pi,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-1.2, -0.3, -0.8]),
            ),
        dict(testcase_name='negative x-axis negative xyz-vector',
            theta = np.pi/2,
            phi = np.pi,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([1.2, 0.3, 0.8]),
            ),
        dict(testcase_name='negative y-axis positive y-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            fx = 0,
            fy = 0.8,
            fz = 0,
            expected_f = np.array([-0.8, 0, 0]),
            ),
        dict(testcase_name='negative y-axis negative y-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            fx = 0,
            fy = -0.8,
            fz = 0,
            expected_f = np.array([0.8, 0, 0]),
            ),
        dict(testcase_name='negative y-axis positive xyz-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-0.8, -0.3, 1.2]),
            ),
        dict(testcase_name='negative y-axis negative xyz-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([0.8, 0.3, -1.2]),
            ),
        dict(testcase_name='positive z-axis positive z-vector',
            theta = 0,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = 0.3,
            expected_f = np.array([0.3, 0, 0]),
            ),
        dict(testcase_name='positive z-axis negative z-vector',
            theta = 0,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = -0.3,
            expected_f = np.array([-0.3, 0, 0]),
            ),
        dict(testcase_name='positive z-axis positive xyz-vector',
            theta = 0,
            phi = 0,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.3, 1.2, 0.8]),
            ),
        dict(testcase_name='positive z-axis negative xyz-vector',
            theta = 0,
            phi = 0,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.3, -1.2, -0.8]),
            ),
        dict(testcase_name='negative z-axis positive z-vector',
            theta = np.pi,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = 0.3,
            expected_f = np.array([-0.3, 0, 0]),
            ),
        dict(testcase_name='negative z-axis negative z-vector',
            theta = np.pi,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = -0.3,
            expected_f = np.array([0.3, 0, 0]),
            ),
        dict(testcase_name='negative z-axis positive xyz-vector',
            theta = np.pi,
            phi = 0,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-0.3, -1.2, 0.8]),
            ),
        dict(testcase_name='negative z-axis negative xyz-vector',
            theta = np.pi,
            phi = 0,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([0.3, 1.2, -0.8]),
            ),
        dict(testcase_name='positive xyz positive xyz-vector',
            theta = 0.1,
            phi = 0.2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.4317803940530825, 1.2983958169713976, 0.5456500653189199]),
            ),
        dict(testcase_name='positive xyz negative xyz-vector',
            theta = 0.1,
            phi = 0.2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.4317803940530825, -1.2983958169713976, -0.5456500653189199]),
            ),
        dict(testcase_name='positive xy negative z positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([1.2983958169713974, -0.43178039405308244, 0.5456500653189199]),
            ),
        dict(testcase_name='positive xy negative z negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-1.2983958169713974, 0.43178039405308244, -0.5456500653189199]),
            ),
        dict(testcase_name='positive xz negative y positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + 3*np.pi/2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.244027139269055, -0.5728741127706006, 1.3350153580455388]),
            ),
        dict(testcase_name='positive xz negative y negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + 3*np.pi/2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.244027139269055, 0.5728741127706006, -1.3350153580455388]),
            ),
        dict(testcase_name='positive yz negative x positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.16522210511373298, -1.3582958669594947, -0.5456500653189198]),
            ),
        dict(testcase_name='positive yz negative x negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.16522210511373298, 1.3582958669594947, 0.5456500653189198]),
            ),
        dict(testcase_name='positive x negative yz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + 3*np.pi/2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-0.5728741127706005, -0.24402713926905495, 1.3350153580455388]),
            ),
        dict(testcase_name='positive x negative yz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + 3*np.pi/2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([0.5728741127706005, 0.24402713926905495, -1.3350153580455388]),
            ),
        dict(testcase_name='positive y negative xz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi/2,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.5129740627825038, -0.3529753598977604, -1.3350153580455388]),
            ),
        dict(testcase_name='positive y negative xz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi/2,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.5129740627825038, 0.3529753598977604, 1.3350153580455388]),
            ),
        dict(testcase_name='positive z negative xy positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([0.16522210511373298, -1.3582958669594947, -0.5456500653189198]),
            ),
        dict(testcase_name='positive z negative xy negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([-0.16522210511373298, 1.3582958669594947, 0.5456500653189198]),
            ),
        dict(testcase_name='negative xyz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi,
            fx = 1.2,
            fy = 0.8,
            fz = 0.3,
            expected_f = np.array([-1.3582958669594944, -0.1652221051137329, -0.5456500653189198]),
            ),
        dict(testcase_name='negative xyz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi,
            fx = -1.2,
            fy = -0.8,
            fz = -0.3,
            expected_f = np.array([1.3582958669594944, 0.1652221051137329, 0.5456500653189198]),
            ),
        dict(testcase_name='zero vectors',
            theta = 0,
            phi = 0,
            fx = 0,
            fy = 0,
            fz = 0,
            expected_f = np.array([0, 0, 0]),
            ),
            )
    def test_axes(self, theta, phi, fx, fy, fz, expected_f):
        calculated_fr, calculated_ftheta, calculated_fphi = coord_transforms.cartesian_to_spherical_vector_field(theta, phi, fx, fy, fz)
        np.testing.assert_array_almost_equal(
            expected_f, 
            np.squeeze(np.array([calculated_fr, calculated_ftheta, calculated_fphi])))

    def test_multi_element_array(self):
        theta = np.array([0.1 + np.pi/2, 0.1, 0.1, np.pi/2])
        phi = np.array([0.2 + np.pi, 0.2 + 3*np.pi/2, 0.2, 3*np.pi/2])
        fx = np.array([-1.2, 1.2, 1.2, 0])
        fy = np.array([-0.8, 0.8, 0.8, 0.8])
        fz = np.array([-0.3, 0.3, 0.3, 0])
        calculated_fr, calculated_ftheta, calculated_fphi = coord_transforms.cartesian_to_spherical_vector_field(theta, phi, fx, fy, fz)
        expected_fr = np.array([1.3582958669594944, 0.244027139269055, 0.4317803940530825, -0.8])
        expected_ftheta = np.array([0.1652221051137329, -0.5728741127706006, 1.2983958169713976, 0])
        expected_fphi = np.array([0.5456500653189198, 1.3350153580455388, 0.5456500653189199, 0])
        np.testing.assert_array_almost_equal(expected_fr, calculated_fr)
        np.testing.assert_array_almost_equal(expected_ftheta, calculated_ftheta)
        np.testing.assert_array_almost_equal(expected_fphi, calculated_fphi)


class SphericalCartesianVectorFieldTest(parameterized.TestCase):
    '''
    Testing the Spherical to Cartesian Vector Field coordinate transform 
    function
    vectors along each of x,y,z axes, positive and negative
    vector pointing into each quadrant
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive x-axis positive x-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 1.2,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([1.2, 0, 0])
            ),
        dict(testcase_name='positive x-axis negative x-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = -1.2,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([-1.2, 0, 0])
            ),
        dict(testcase_name='positive x-axis positive y-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 0,
            f_theta = 0,
            f_phi = 0.8,
            expected_f = np.array([0, 0.8, 0])
            ),
        dict(testcase_name='positive x-axis negative y-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 0,
            f_theta = 0,
            f_phi = -0.8,
            expected_f = np.array([0, -0.8, 0])
            ),
        dict(testcase_name='positive x-axis positive z-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 0,
            f_theta = -0.3,
            f_phi = 0,
            expected_f = np.array([0, 0, 0.3])
            ),
        dict(testcase_name='positive x-axis negative z-vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 0,
            f_theta = 0.3,
            f_phi = 0,
            expected_f = np.array([0, 0, -0.3])
            ),
        dict(testcase_name='positive x-axis positive xyz vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 1.2,
            f_theta = -0.3,
            f_phi = 0.8,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive x-axis positive xy negative z vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 1.2,
            f_theta = 0.3,
            f_phi = 0.8,
            expected_f = np.array([1.2, 0.8, -0.3])
            ),
        dict(testcase_name='positive x-axis positive xz negative y vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 1.2,
            f_theta = -0.3,
            f_phi = -0.8,
            expected_f = np.array([1.2, -0.8, 0.3])
            ),
        dict(testcase_name='positive x-axis positive yz negative x vector',
            theta = np.pi/2,
            phi = 0,
            f_r = -1.2,
            f_theta = -0.3,
            f_phi = 0.8,
            expected_f = np.array([-1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive x-axis positive z negative xy vector',
            theta = np.pi/2,
            phi = 0,
            f_r = -1.2,
            f_theta = -0.3,
            f_phi = -0.8,
            expected_f = np.array([-1.2, -0.8, 0.3])
            ),
        dict(testcase_name='positive x-axis positive y negative xz vector',
            theta = np.pi/2,
            phi = 0,
            f_r = -1.2,
            f_theta = 0.3,
            f_phi = 0.8,
            expected_f = np.array([-1.2, 0.8, -0.3])
            ),
        dict(testcase_name='positive x-axis positive x negative yz vector',
            theta = np.pi/2,
            phi = 0,
            f_r = 1.2,
            f_theta = 0.3,
            f_phi = -0.8,
            expected_f = np.array([1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive x-axis negative xyz vector',
            theta = np.pi/2,
            phi = 0,
            f_r = -1.2,
            f_theta = 0.3,
            f_phi = -0.8,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive y-axis positive y-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            f_r = 0.8,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0.8, 0])
            ),
        dict(testcase_name='positive y-axis negative y-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            f_r = -0.8,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, -0.8, 0])
            ),
        dict(testcase_name='positive y-axis positive xyz-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            f_r = 0.8,
            f_theta = -0.3,
            f_phi = -1.2,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive y-axis negative xyz-vector',
            theta = np.pi/2,
            phi = np.pi/2,
            f_r = -0.8,
            f_theta = 0.3,
            f_phi = 1.2,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='negative x-axis positive x-vector',
            theta = np.pi/2,
            phi = np.pi,
            f_r = -1.2,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([1.2, 0, 0])
            ),
        dict(testcase_name='negative x-axis negative x-vector',
            theta = np.pi/2,
            phi = np.pi,
            f_r = 1.2,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([-1.2, 0, 0])
            ),
        dict(testcase_name='negative x-axis positive xyz-vector',
            theta = np.pi/2,
            phi = np.pi,
            f_r = -1.2,
            f_theta = -0.3,
            f_phi = -0.8,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='negative x-axis negative xyz-vector',
            theta = np.pi/2,
            phi = np.pi,
            f_r = 1.2,
            f_theta = 0.3,
            f_phi = 0.8,
            expected_f = np.array([-1.2, -0.8, -0.3,])
            ),
        dict(testcase_name='negative y-axis positive y-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            f_r = -0.8,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0.8, 0])
            ),
        dict(testcase_name='negative y-axis negative y-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            f_r = 0.8,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, -0.8, 0])
            ),
        dict(testcase_name='negative y-axis positive xyz-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            f_r = -0.8,
            f_theta = -0.3,
            f_phi = 1.2,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='negative y-axis negative xyz-vector',
            theta = np.pi/2,
            phi = 3*np.pi/2,
            f_r = 0.8,
            f_theta = 0.3,
            f_phi = -1.2,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive z-axis positive z-vector',
            theta = 0,
            phi = 0,
            f_r = 0.3,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0, 0.3])
            ),
        dict(testcase_name='positive z-axis negative z-vector',
            theta = 0,
            phi = 0,
            f_r = -0.3,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0, -0.3])
            ),
        dict(testcase_name='positive z-axis positive xyz-vector',
            theta = 0,
            phi = 0,
            f_r = 0.3,
            f_theta = 1.2,
            f_phi = 0.8,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive z-axis negative xyz-vector',
            theta = 0,
            phi = 0,
            f_r = -0.3,
            f_theta = -1.2,
            f_phi = -0.8,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='negative z-axis positive z-vector',
            theta = np.pi,
            phi = 0,
            f_r = -0.3,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0, 0.3])
            ),
        dict(testcase_name='negative z-axis negative z-vector',
            theta = np.pi,
            phi = 0,
            f_r = 0.3,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0, -0.3])
            ),
        dict(testcase_name='negative z-axis positive xyz-vector',
            theta = np.pi,
            phi = 0,
            f_r = -0.3,
            f_theta = -1.2,
            f_phi = 0.8,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='negative z-axis negative xyz-vector',
            theta = np.pi,
            phi = 0,
            f_r = 0.3,
            f_theta = 1.2,
            f_phi = -0.8,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive xyz positive xyz-vector',
            theta = 0.1,
            phi = 0.2,
            f_r = 0.4317803940530825,
            f_theta = 1.2983958169713976,
            f_phi = 0.5456500653189199,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive xyz negative xyz-vector',
            theta = 0.1,
            phi = 0.2,
            f_r = -0.4317803940530825,
            f_theta = -1.2983958169713976,
            f_phi = -0.5456500653189199,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive xy negative z positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2,
            f_r = 1.2983958169713974,
            f_theta = -0.43178039405308244,
            f_phi = 0.5456500653189199,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive xy negative z negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2,
            f_r = -1.2983958169713974,
            f_theta = 0.43178039405308244,
            f_phi = -0.5456500653189199,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive xz negative y positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + 3*np.pi/2,
            f_r = 0.244027139269055,
            f_theta = -0.5728741127706006,
            f_phi = 1.3350153580455388,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive xz negative y negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + 3*np.pi/2,
            f_r = -0.244027139269055,
            f_theta = 0.5728741127706006,
            f_phi = -1.3350153580455388,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive yz negative x positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            f_r = 0.16522210511373298,
            f_theta = -1.3582958669594947,
            f_phi = -0.5456500653189198,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive yz negative x negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            f_r = -0.16522210511373298,
            f_theta = 1.3582958669594947,
            f_phi = 0.5456500653189198,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive x negative yz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + 3*np.pi/2,
            f_r = -0.5728741127706005,
            f_theta = -0.24402713926905495,
            f_phi = 1.3350153580455388,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive x negative yz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + 3*np.pi/2,
            f_r = 0.5728741127706005,
            f_theta = 0.24402713926905495,
            f_phi = -1.3350153580455388,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive y negative xz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi/2,
            f_r = 0.5129740627825038,
            f_theta = -0.3529753598977604,
            f_phi = -1.3350153580455388,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive y negative xz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi/2,
            f_r = -0.5129740627825038,
            f_theta = 0.3529753598977604,
            f_phi = 1.3350153580455388,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='positive z negative xy positive xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            f_r = 0.16522210511373298,
            f_theta = -1.3582958669594947,
            f_phi = -0.5456500653189198,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='positive z negative xy negative xyz-vector',
            theta = 0.1,
            phi = 0.2 + np.pi,
            f_r = -0.16522210511373298,
            f_theta = 1.3582958669594947,
            f_phi = 0.5456500653189198,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='negative xyz positive xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi,
            f_r = -1.3582958669594944,
            f_theta = -0.1652221051137329,
            f_phi = -0.5456500653189198,
            expected_f = np.array([1.2, 0.8, 0.3])
            ),
        dict(testcase_name='negative xyz negative xyz-vector',
            theta = 0.1 + np.pi/2,
            phi = 0.2 + np.pi,
            f_r = 1.3582958669594944,
            f_theta = 0.1652221051137329,
            f_phi = 0.5456500653189198,
            expected_f = np.array([-1.2, -0.8, -0.3])
            ),
        dict(testcase_name='zero vectors',
            theta = 0,
            phi = 0,
            f_r = 0,
            f_theta = 0,
            f_phi = 0,
            expected_f = np.array([0, 0, 0])
            ),
            )
    def test_axes(self, theta, phi, f_r, f_theta, f_phi, expected_f):
        calculated_fx, calculated_fy, calculated_fz = coord_transforms.spherical_to_cartesian_vector_field(
            theta, phi, f_r, f_theta, f_phi)
        np.testing.assert_array_almost_equal(
            expected_f, 
            np.squeeze(np.array([calculated_fx, calculated_fy, calculated_fz])))

    def test_multi_element_array(self):
        theta = np.array([0.1 + np.pi/2, 0.1, 0.1, np.pi/2])
        phi = np.array([0.2 + np.pi, 0.2 + 3*np.pi/2, 0.2, 3*np.pi/2])
        f_r = np.array([1.3582958669594944, 0.244027139269055, 0.4317803940530825, -0.8])
        f_theta = np.array([0.1652221051137329, -0.5728741127706006, 1.2983958169713976, 0])
        f_phi = np.array([0.5456500653189198, 1.3350153580455388, 0.5456500653189199, 0])
        expected_fx = np.array([-1.2, 1.2, 1.2, 0])
        expected_fy = np.array([-0.8, 0.8, 0.8, 0.8])
        expected_fz = np.array([-0.3, 0.3, 0.3, 0])
        calculated_fx, calculated_fy, calculated_fz = coord_transforms.spherical_to_cartesian_vector_field(
            theta, phi, f_r, f_theta, f_phi)
        np.testing.assert_array_almost_equal(expected_fx, calculated_fx)
        np.testing.assert_array_almost_equal(expected_fy, calculated_fy)
        np.testing.assert_array_almost_equal(expected_fz, calculated_fz)


class FieldMagnitudeTest(parameterized.TestCase):
    '''
    Testing the field_magnitude function
    - 3 element array
    - Nx3 element array
    - 3xN element array with different axis input
    - real values
    - imaginary values
    - floats
    - 2 element array
    '''
    @parameterized.named_parameters(
        dict(testcase_name='positive real three-vector',
            vector = np.array([1,1,1]),
            expected_magnitude = np.array([1.7320508075688772])
            ),
        dict(testcase_name='negative real three-vector',
            vector = np.array([-1, -1, -1]),
            expected_magnitude = np.array([1.7320508075688772])
            ),
        dict(testcase_name='mixed sign real three-vector',
            vector = np.array([-1, 1, -1]),
            expected_magnitude = np.array([1.7320508075688772])
            ),
        dict(testcase_name='mixed sign real three-vector floats',
            vector = np.array([-1.4, 1.1, -0.1]),
            expected_magnitude = np.array([1.783255450012701])
            ),
        dict(testcase_name='positive imaginary three-vector',
            vector = np.array([2+0.5j, 3+1j, 1+5j]),
            expected_magnitude = np.array([6.34428877022476])
            ),
        dict(testcase_name='negative imaginary three-vector',
            vector = np.array([-2-0.5j, -3-1j, -1-5j]),
            expected_magnitude = np.array([6.34428877022476])
            ),
        dict(testcase_name='mixed sign imaginary three-vector',
            vector = np.array([-2+0.5j, 3-1j, 1+5j]),
            expected_magnitude = np.array([6.34428877022476])
            ),
        dict(testcase_name='mixed sign imaginary two-vector',
            vector = np.array([-2+0.5j, 3-1j]),
            expected_magnitude = np.array([3.774917217635375])
            ),
        dict(testcase_name='mixed sign imaginary one-vector',
            vector = np.array([-2+0.5j]),
            expected_magnitude = np.array([2.0615528128088303])
            ),
        dict(testcase_name='mixed sign imaginary 1x3 vector',
            vector = np.array([[-2+0.5j, 3-1j, 1+5j]]),
            expected_magnitude = np.array([[6.34428877022476]])
            ),
        dict(testcase_name='mixed sign imaginary 3x1 vector',
            vector = np.array([[-2+0.5j], [3-1j], [1+5j]]),
            expected_magnitude = np.array([[2.0615528128088303], [3.1622776601683795], [5.0990195135927845]])
            ),
        dict(testcase_name='0 three-vector',
            vector = np.array([0, 0, 0]),
            expected_magnitude = np.array([0])
            ),
        )
    def test_three_vector(self, vector, expected_magnitude):
        calculated_magnitude = coord_transforms.field_magnitude(vector, keepdims=True)
        np.testing.assert_array_almost_equal(expected_magnitude, calculated_magnitude)

    @parameterized.named_parameters(
        dict(testcase_name='2D array axis 1',
            vector = np.array([[-2+0.5j, 3-1j, 1+5j], [0.5-3j, -10-0.6j, 0]]),
            axis = 1,
            expected_magnitude = np.array([[6.34428877022476], [10.469479452198184]]),
            ),
        dict(testcase_name='2D array axis 0',
            vector = np.array([[-2+0.5j, 3-1j, 1+5j], [0.5-3j, -10-0.6j, 0]]),
            axis = 0,
            expected_magnitude = np.array([[3.6742346141747673, 10.505236789335116, 5.0990195135927845]]),
            ),
        dict(testcase_name='1x3x3 array axis 0',
            vector = np.array([[[-2+0.5j, 3-1j, 1+5j], [0.5-3j, -10-0.6j, 0], [0-2.1j, 5.3, -7-2.4j]]]),
            axis = 0,
            expected_magnitude = np.array([[[2.0615528128088303, 3.1622776601683795, 5.0990195135927845], [3.0413812651491097, 10.017983829094554, 0], [2.1, 5.3, 7.3999999999999995]]]),
            ),
        )
    def test_axis(self, vector, axis, expected_magnitude):
        calculated_magnitude = coord_transforms.field_magnitude(vector, axis=axis, keepdims=True)
        np.testing.assert_array_almost_equal(expected_magnitude, calculated_magnitude)


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

    def testAxisVectorsRotateBy0(self):
        '''
        take (0,1,0), (1, 0, 0) and (0,0,1) and rotate around (0,0,1) by 0
        '''
        xyz = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        angle = 0
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(xyz, rotated_vector)

    def testAxis0VectorsRotateBy90(self):
        '''
        take (0,0,0) and rotate around (0,0,1) by 90
        '''
        xyz = np.array([[0, 0, 0]])
        angle = 0
        rotation_vector = np.array([0,0,1])
        rotated_vector = coord_transforms.rotate_vector_Nd(xyz, angle, rotation_vector)
        np.testing.assert_array_almost_equal(xyz, rotated_vector)

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
