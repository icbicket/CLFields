import cl_sim_funcs
import unittest
import numpy as np


class ImageFunctionsTest(unittest.TestCase):
    '''
    for each of (0,1) and (-1,1):
    all negatives
    negative to 0
    -something to +something
    -something to +something where |-something| is bigger
    -something to +something where |+something| is bigger
    0 to +something
    +something to +something bigger
    '''
    def set_colour_scale_0_to_1(self):
        colours = np.arange(5)
        lims = [0, 1]
        normed = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.arange(5)/5
        self.assertAlmostEqual(normed, expected_norm)

    def set_colour_scale_neg1_to_1(self):
        colours = np.arange(5)
        lims = [-1, 1]
        normed = cl_sim_funcs.norm_colours(colours, lims)
        expected_norm = np.arange(5)/10
        self.assertAlmostEqual(normed, expected_norm)

if __name__ == '__main__':
    unittest.main()
