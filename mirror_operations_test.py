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
'''

class ParabolaPositionTest(unittest.TestCase):
    '''
    Test the parabola_position function in mirror_operations
    '''
    def testParabolaPositions(self):
        '''
        check that parabola positions are calculated correctly
        '''
        xyz = np.array([
            [-1, 0, 0], 
            [0, 1, 0], 
            [0, -1, 0], 
            [0, 0, 1], 
            [0, 0, -1], 
            [-2.3, 1, 1],
            ])
        theta = np.array([np.pi/2, np.pi/2, np.pi/2, 0, np.pi])
        phi = np.array([np.pi, np.pi/2, 3*np.pi/2, 0, 0])
        position_expected = np.array([
            [-2.5, 0, 0], 
            [0, 5, 0], 
            [0, -5, 0], 
            [0, 0, 5], 
            [0, 0, -5], 
            [-2.3, 1, 1],
            ])
        position = miop.parabola_position(xyz)
        np.testing.assert_allclose(position_expected, position, atol=1e-7)

    def testParabolaPositionPositiveX(self):
        '''
        check the parabola positions returns NaN on positive x axis
        '''
        xyz = np.array([[1, 0, 0]])
        position_expected = np.array([[np.nan, np.nan, np.nan]])
        position = miop.parabola_position(xyz)
        print(position)
        self.assertTrue(np.all(np.isnan(position)))

if __name__ == '__main__':
    unittest.main()
