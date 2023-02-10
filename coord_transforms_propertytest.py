import coord_transforms
from hypothesis import given
from hypothesis.extra import numpy as npstrat
import hypothesis.strategies as st
import numpy as np
import unittest

sane_floats = st.floats(min_value=-1e16, max_value=1e16).filter(lambda x: np.abs(x)>1e-10)
angle_floats = st.floats(min_value=-1e3, max_value=1e3)
angle_ints = st.integers(min_value=-10, max_value=10)
threeD_vector = npstrat.arrays(dtype=np.float64, shape=3, elements=sane_floats)
nonzero_3d_vector = threeD_vector.filter(lambda vec: np.count_nonzero(np.square(vec))>0)


class VectorRotationTest(unittest.TestCase):
    @given(
        xyz=threeD_vector,
        angle=angle_ints, 
        axis=nonzero_3d_vector
        )
    def testRotate2Pi(self, xyz, angle, axis):
        '''
        A rotation of a vector by an integer multiple of 2pi should return the same vector
        '''
        rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, 2*np.pi*angle, axis)
        xyz_norm = xyz/coord_transforms.field_magnitude(xyz)
        rotated_xyz_norm = rotated_xyz/coord_transforms.field_magnitude(rotated_xyz)
        np.testing.assert_allclose(xyz_norm, rotated_xyz_norm, atol=1e-7)

    @given(
        xyz=threeD_vector,
        angle=angle_ints,
        axis=nonzero_3d_vector
        )
    def testPiNegativePi(self, xyz, angle, axis):
        '''
        A rotation by 2*pi should give the same result as a rotation by negative pi
        '''
        rotated_xyz_positive = coord_transforms.rotate_vector_Nd(xyz, np.pi*angle, axis)
        rotated_xyz_negative = coord_transforms.rotate_vector_Nd(xyz, -np.pi*angle, axis)
        rotated_xyz_positive_norm = rotated_xyz_positive/coord_transforms.field_magnitude(rotated_xyz_positive)
        rotated_xyz_negative_norm = rotated_xyz_negative/coord_transforms.field_magnitude(rotated_xyz_negative)
        np.testing.assert_allclose(rotated_xyz_positive_norm, rotated_xyz_negative_norm, atol=1e-7)


class VectorMagnitudeTest(unittest.TestCase):
    @given(
        xyz=threeD_vector,
        angle=angle_floats,
        axis=nonzero_3d_vector
        )
    def testVectorMagnitude(self, xyz, angle, axis):
        '''
        The magnitude of a vector should be the same before and after it is rotated
        '''
        xyz_mag = np.sqrt(np.sum(np.square(xyz)))
        rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, angle, axis)
        rotated_mag = np.sqrt(np.sum(np.square(rotated_xyz)))
        np.testing.assert_allclose(xyz_mag, rotated_mag)


class DotProductTest(unittest.TestCase):
    @given(
        xyz=threeD_vector,
        angle=angle_floats,
        axis=nonzero_3d_vector
        )
    def testDotProduct(self, xyz, angle, axis):
        '''
        The dot product of the vector onto the rotation axis should be the same 
        before and after rotation
        '''
        xyz_dot = np.dot(xyz, axis)
        rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, angle, axis)
        rotated_dot = np.dot(rotated_xyz, axis)
        np.testing.assert_allclose(xyz_dot, rotated_dot)

## Struggling with floating point errors!
#@given(
#    xyz=threeD_vector,
#    angle=angle_ints,
#    axis=nonzero_3d_vector
#    )
#def testOddMultiplesPi(xyz, angle, axis):
#    '''
#    A rotation by an odd multiple of pi should make the cross product of the 
#    rotated vector and the rotation axis into the negative cross product of the 
#    vector and the rotation axis
#    '''
#    xyz_cross = np.cross(xyz, axis)
#    rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, np.pi*(2*angle-1), axis)
#    rotated_cross = np.cross(rotated_xyz, axis)
#    np.testing.assert_allclose(-xyz_cross, rotated_cross, atol=1e-3)
    
if __name__== '__main__':
    unittest.main()
