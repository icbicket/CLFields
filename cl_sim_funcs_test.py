import cl_sim_funcs
import unittest
import numpy as np


class NormColoursTest(unittest.TestCase):
    '''
    for each of (0,1) and (-1,1):
    all negatives: done
    negative to 0: done
    -something to +something
    -something to +something where |-something| is bigger: done
    -something to +something where |+something| is bigger: done
    0 to +something: done
    +something to +something bigger: done
    '''
    def testSetColourScale_0_to_1(self):
        '''
        Limits are 0-1
        positive numbers starting from 0
        '''
        colours = np.arange(5)
        lims = (0, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 5, 5)/5
        np.testing.assert_array_almost_equal(normed, expected_norm)

    def testSetColourScale_neg1_to_1(self):
        '''
        Limits are -1 to 1
        Positive numbers starting from 0
        '''
        colours = np.arange(5)
        lims = (-1, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(5, 10, 5)/10
        np.testing.assert_array_almost_equal(normed, expected_norm)

    def testSetColourScale_0_1_NegTo0(self):
        '''
        Limits are 0 to 1
        Negative numbers ending in 0
        '''
        colours = - np.arange(5)
        lims = (0, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.flip(np.linspace(0, 5, 5)/5)
        np.testing.assert_array_almost_equal(normed, expected_norm)
    
    def testSetColourScale_neg1_1_NegTo0(self):
        '''
        Limits are -1 to 1
        Negative numbers ending in 0
        '''
        colours = -np.flip(np.arange(5))
        lims = (-1,1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 5, 5)/10
        np.testing.assert_array_almost_equal(normed, expected_norm)
        
    def testSetColourScale_0_1_all_neg(self):
        '''
        Limits are 0 to 1
        Negative numbers not including 0
        '''
        colours = np.arange(-10, -5)
        lims = (0,1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 5, 5)/5
        np.testing.assert_array_almost_equal(normed, expected_norm)

    def testSetColourScale_neg1_1_all_neg(self):
        '''
        Limits are -1 to 1
        Negative numbers not including 0
        '''
        colours = np.arange(-10, -5)
        lims = (-1,1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.arange(-10, -5)/20 + 0.5
        np.testing.assert_array_almost_equal(normed, expected_norm)
    
    def testSetColourScale_0_1_all_pos(self):
        '''
        Limits are 0 to 1
        Positive numbers not including 0
        '''
        colours = np.arange(5, 10)
        lims = (0, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 5, 5)/5
        np.testing.assert_array_almost_equal(normed, expected_norm)
        
    def testSetColourScale_Neg1_1_all_pos(self):
        '''
        Limits are -1 to 1
        Positive numbers not including 0
        '''
        colours = np.arange(5, 10)
        lims = (-1, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.arange(5, 10)/18+0.5
        np.testing.assert_array_almost_equal(normed, expected_norm)
        
    def testSetColourScale_0_1_NegToBigPos(self):
        '''
        Limits are 0 to 1
        Negative number to bigger positive number
        '''
        colours = np.arange(-2, 4)
        lims = (0, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 1, 6)
        np.testing.assert_array_almost_equal(normed, expected_norm)
        
    def testSetColourScale_Neg1_1_NegToBigPos(self):
        '''
        Limits are -1 to 1
        Negative number to bigger positive number
        '''
        colours = np.arange(-2, 4)
        lims = (-1, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 1, 7)[1:]
        np.testing.assert_array_almost_equal(normed, expected_norm)

    def testSetColourScale_0_1_BigNegToPos(self):
        '''
        Limits are 0 to 1
        Bigger negative number to positive number
        '''
        colours = np.arange(-4, 2)
        lims = (0, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 1, 6)
        np.testing.assert_array_almost_equal(normed, expected_norm)
        
    def testSetColourScale_Neg1_1_BigNegToPos(self):
        '''
        Limits are -1 to 1
        Negative number to bigger positive number
        '''
        colours = np.arange(-4, 2)
        lims = (-1, 1)
        normed, _, _ = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.linspace(0, 1, 9)[:-3]
        np.testing.assert_array_almost_equal(normed, expected_norm)

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
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 1)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ2(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 2
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[1, 0, 1], [3, 2, 3], [1, 0, 1]])
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 2)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ3(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 3
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[2, 3, 2], [0, 1, 0], [2, 3, 2]])
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 3)
        np.testing.assert_array_almost_equal(arrayfull, testarray)

    def testQuadrant2x2ArrayQ2(self):
        '''
        2x2 array symmetrizes properly
        input quadrant 4
        '''
        array = np.array([[0, 1], [2, 3]])
        arrayfull = np.array([[3, 2, 3], [1, 0, 1], [3, 2, 3]])
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 4)
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
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 1)
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
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 2)
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
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 3)
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
        testarray = cl_sim_funcs.expand_quadrant_symmetry(array, 4)
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
        r, theta, phi = cl_sim_funcs.cartesian_to_spherical_coords(xyz)
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
        r, theta, phi = cl_sim_funcs.cartesian_to_spherical_coords(xyz)
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
