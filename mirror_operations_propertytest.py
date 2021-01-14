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
def testParabolaNormalsSameDirection(xyz, angle, axis):
    '''
    All the normals of the parabola should point towards +x
    '''
    pass
#    parabola_normals = 
#    rotated_xyz = coord_transforms.rotate_vector_Nd(xyz, 2*np.pi*angle, axis)
#    np.testing.assert_allclose(xyz, rotated_xyz, atol=1e-7)

