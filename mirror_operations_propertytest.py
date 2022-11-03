import mirror_operations as miop
from hypothesis import given
from hypothesis.extra import numpy as npstrat
import hypothesis.strategies as st
import hypothesis
import numpy as np
import unittest
import coord_transforms as ct

'''
parabola positions of a given phi quadrant should have the same x and y sign
 (eg phi<pi/2 should be in (+x, +y)
parabola positions with theta<pi/2 should have positive z
'''

def x_axis_cone(direction):
    r, theta, phi = ct.cartesian_to_spherical_coords(direction)
    cone_condition = np.logical_not(
        np.logical_and(abs(theta-np.pi/2)<5e-8, abs(phi)<5e-8)
        )
    return cone_condition

sane_floats = st.floats(min_value=-1e16, max_value=1e16)
angle_floats = st.floats(min_value=-1e3, max_value=1e3)
angle_ints = st.integers(min_value=-10, max_value=10)
one_by_three_vector = npstrat.arrays(dtype=np.float64, shape=(1,3), elements=sane_floats)
nonzero_3_vector = one_by_three_vector.filter(lambda vec: np.count_nonzero(vec)>0)
not_positive_x_axis = nonzero_3_vector.filter(x_axis_cone)
positive_sane_float = st.floats(min_value=1e-16, max_value=1e16)

@given(
    direction=not_positive_x_axis,
    )
def test_parabola_theta_less_than_90(direction):
    '''
    Parabola positions (x,y,z) components should have the same sign as the 
    input direction (x,y,z) components
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

@given(
    theta=angle_floats,
    phi=angle_floats,
    )
def test_surface_polarization_directions_magnitude(theta, phi):
    '''
    The magnitudes of the surface polarization direction results should be 
    unit vectors
    '''
    p, s = miop.surface_polarization_directions(theta, phi)
    p_magnitude = np.sqrt(np.sum(np.square(p), axis=-1))
    s_magnitude = np.sqrt(np.sum(np.square(s), axis=-1))
    np.testing.assert_almost_equal(p_magnitude, 1)
    np.testing.assert_almost_equal(s_magnitude, 1)

@given(
    normal=nonzero_3_vector,
    e_incident_direction=nonzero_3_vector,
    n_mirror=positive_sane_float,
    n_environment=positive_sane_float,
    )
def test_fresnel_reflection_coefficients_magnitude(
    normal,
    e_incident_direction,
    n_mirror,
    n_environment):
    hypothesis.assume(n_environment < n_mirror)
    r_s, r_p = miop.fresnel_reflection_coefficients(
        normal, e_incident_direction, n_mirror, n_environment)
    assert r_s <= 1 and r_s >= 0
    assert r_p <= 1 and r_p >= 0

if __name__== '__main__':
    test_parabola_normals_same_direction()
    test_parabola_theta_less_than_90()
    test_parabola_normals_magnitude()
    test_surface_polarization_directions_magnitude()
    test_fresnel_reflection_coefficients_magnitude()

class ARMaskCalcTest(unittest.TestCase):
    @given(
        st.floats(min_value=np.pi/2, max_value=3*np.pi/2, allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False))
    def testARMaskCalcNegativeZTrue(self, theta, phi):
        '''
        all inputs in negative Z space should be True
        '''
        calculated = cl_calcs.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=None,
            orientation=0)
        self.assertTrue(calculated)
