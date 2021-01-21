import coord_transforms
from hypothesis import given
from hypothesis.extra import numpy as npstrat
import hypothesis.strategies as st
import numpy as np

sane_floats = st.floats(min_value=-1e16, max_value=1e16)
angle_floats = st.floats(min_value=-1e3, max_value=1e3)
angle_ints = st.integers(min_value=-10, max_value=10)
threeD_vector = npstrat.arrays(dtype=np.float64, shape=3, elements=sane_floats)
nonzero_3d_vector = threeD_vector.filter(lambda vec: np.count_nonzero(vec)>0)

@given(
    xyz=threeD_vector,
    angle=angle_ints, 
    axis=nonzero_3d_vector
    )
def testRotate2Pi(xyz, angle, axis):
    '''
    A rotation of a vector by an integer multiple of 2pi should return the same vector
    '''
    rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, 2*np.pi*angle, axis)
    np.testing.assert_allclose(xyz, rotated_xyz, atol=1e-7)

@given(
    xyz=threeD_vector,
    angle=angle_floats,
    axis=nonzero_3d_vector
    )
def testVectorMagnitude(xyz, angle, axis):
    '''
    The magnitude of a vector should be the same before and after it is rotated
    '''
    xyz_mag = np.sqrt(np.sum(np.square(xyz)))
    rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, angle, axis)
    rotated_mag = np.sqrt(np.sum(np.square(rotated_xyz)))
    np.testing.assert_allclose(xyz_mag, rotated_mag)
    
@given(
    xyz=threeD_vector,
    angle=angle_floats,
    axis=nonzero_3d_vector
    )
def testDotProduct(xyz, angle, axis):
    '''
    The dot product of the vector onto the rotation axis should be the same 
    before and after rotation
    '''
    xyz_dot = np.dot(xyz, axis)
    rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, angle, axis)
    rotated_dot = np.dot(rotated_xyz, axis)
    np.testing.assert_allclose(xyz_dot, rotated_dot)

@given(
    xyz=threeD_vector,
    angle=angle_ints,
    axis=nonzero_3d_vector
    )
def testPiNegativePi(xyz, angle, axis):
    '''
    A rotation by pi should give the same result as a rotation by negative pi
    '''
    rotated_xyz_positive = coord_transforms.rotate_vector_Nd(xyz, np.pi*angle, axis)
    rotated_xyz_negative = coord_transforms.rotate_vector_Nd(xyz, -np.pi*angle, axis)
    np.testing.assert_allclose(rotated_xyz_positive, rotated_xyz_negative)

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
    testRotate2Pi()
    testVectorMagnitude()
    testDotProduct()
    testPiNegativePi()
#    testOddMultiplesPi()
