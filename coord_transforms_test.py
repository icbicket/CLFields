import coord_transforms
import unittest
import numpy as np

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

    def testQuadrant2x2ArrayQ2(self):
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

    def testQuadrant3x3ArrayQ2(self):
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

    def test010(self):
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
                    [np.sqrt(2), 3*np.pi/2, np.pi/2],
                    [np.sqrt(2), np.pi/2, np.pi/2],
                    [np.sqrt(2), 3*np.pi/2, np.pi/2]
                        ])
        np.testing.assert_array_almost_equal(np.transpose(np.array([r, theta, phi])), rthph)

if __name__ == '__main__':
    unittest.main()
