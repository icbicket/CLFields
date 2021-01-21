import mirror_operations as miop
from hypothesis import given
from hypothesis.extra import numpy as npstrat
import hypothesis.strategies as st
import numpy as np
import unittest

'''
parabola positions of a given phi quadrant should have the same x and y sign
 (eg phi<pi/2 should be in (+x, +y)
parabola positions with theta<pi/2 should have positive z
'''


sane_floats = st.floats(min_value=-1e16, max_value=1e16)
angle_floats = st.floats(min_value=-1e3, max_value=1e3)
angle_ints = st.integers(min_value=-10, max_value=10)
one_by_three_vector = npstrat.arrays(dtype=np.float64, shape=(1,3), elements=sane_floats)
nonzero_3_vector = one_by_three_vector.filter(lambda vec: np.count_nonzero(vec)>0)
not_positive_x_axis = nonzero_3_vector.filter(lambda vec:
    np.logical_not(
        np.logical_and(vec[:, 0]>0, np.allclose(vec[:, 1:], np.array([0.,0.])))
        )
    )

@given(
    direction=not_positive_x_axis,
    )
def test_parabola_theta_less_than_90(direction):
    '''
    Parabola positions (x,y,z) components should have the same sign as the input
     direction (x,y,z) components
    '''
    position = miop.parabola_position(direction)
    assert np.all(np.sign(position)==np.sign(direction))

@given(
    direction=not_positive_x_axis,
    )
def test_parabola_normals_same_direction(direction):
    '''
    All the normals of the parabola should point towards +x
    '''
    position = miop.parabola_position(direction)
    normals = miop.parabola_normals(position)
    assert normals[:, 0] > 0

@given(
    direction=not_positive_x_axis,
    )
def test_parabola_normals_magnitude(direction):
    '''
    All the normals of the parabola should be unit vectors
    '''
    position = miop.parabola_position(direction)
    normals = miop.parabola_normals(position)
    normal_magnitude = np.sqrt(np.sum(np.square(normals), axis=-1))
    np.testing.assert_almost_equal(normal_magnitude, 1)

if __name__== '__main__':
    test_parabola_normals_same_direction()
    test_parabola_theta_less_than_90()
    test_parabola_normals_magnitude()
    
