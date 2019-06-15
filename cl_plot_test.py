import cl_plot
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

if __name__ == '__main__':
    unittest.main()
