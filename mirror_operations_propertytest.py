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

def negative_x_axis_cone(direction):
    r, theta, phi = ct.cartesian_to_spherical_coords(direction)
    cone_condition = np.logical_not(
        np.logical_and(abs(theta-np.pi/2)<5e-8, abs(phi-np.pi)<5e-8)
        )
    return cone_condition

sane_floats = st.floats(min_value=-1e16, max_value=1e16).filter(lambda x: np.abs(x)>1e-10)
angle_floats = st.floats(min_value=-1e3, max_value=1e3)
angle_ints = st.integers(min_value=-10, max_value=10)
one_by_three_vector = npstrat.arrays(dtype=np.float64, shape=(1,3), elements=sane_floats)
nonzero_3_vector = one_by_three_vector.filter(lambda vec: np.count_nonzero(vec)>0)
not_negative_x_axis = nonzero_3_vector.filter(negative_x_axis_cone)
not_negative_x_axis_sane = not_negative_x_axis.filter(lambda vec: np.max(np.abs(vec))/np.min(np.abs(vec))<1e5)
positive_sane_float = st.floats(min_value=1e-16, max_value=1e16)

#electric_vector = npstrat.arrays(dtype=np.float64, shape=(2,3), elements=sane_floats).filter(lambda vec: np.isclose(np.sum(vec[0, :] * vec[1, :], axis=-1), 0, atol=1e-9))
k_vector = npstrat.arrays(dtype=np.float64, shape=(2,3), elements=sane_floats).filter(lambda vec: np.logical_not(np.allclose(vec[0, :], vec[1, :])))

class ParabolaPositionTest(unittest.TestCase):
    @given(
        direction=not_negative_x_axis_sane,
        )
    def test_parabola_theta_less_than_90(self, direction):
        '''
        Parabola positions (x,y,z) components should have the same sign as the 
        input direction (x,y,z) components
        '''
        position = miop.parabola_position(direction)
        assert np.all(np.sign(position)==np.sign(direction))


class ParabolaNormalsTest(unittest.TestCase):
    @given(
        direction=not_negative_x_axis_sane,
        )
    def test_parabola_normals_same_x_direction(self, direction):
        '''
        All the normals of the parabola should point towards +x
        '''
        position = miop.parabola_position(direction)
        normals = miop.parabola_normals(position)
        assert normals[:, 0] < 0

    @given(
        direction=not_negative_x_axis_sane,
        )
    def test_parabola_normals_magnitude(self, direction):
        '''
        All the normals of the parabola should be unit vectors
        '''
        position = miop.parabola_position(direction)
        normals = miop.parabola_normals(position)
        normal_magnitude = np.sqrt(np.sum(np.square(normals), axis=-1))
        np.testing.assert_almost_equal(normal_magnitude, 1)


class SurfacePolarizationDirectionsTest(unittest.TestCase):
    @given(
        theta=angle_floats,
        phi=angle_floats,
        )
    def test_magnitude(self, theta, phi):
        '''
        The magnitudes of the surface polarization direction results should be 
        unit vectors
        '''
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        p_magnitude = np.sqrt(np.sum(np.square(p), axis=-1))
        s_magnitude = np.sqrt(np.sum(np.square(s), axis=-1))
        np.testing.assert_almost_equal(p_magnitude, 1)
        np.testing.assert_almost_equal(s_magnitude, 1)

    @given(
        theta=angle_floats,
        phi=angle_floats,
        )
    def test_s_dot_p_is_zero(self, theta, phi):
        '''
        s-direction dot p-direction should be 0 (within floating point error)
        '''
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        dot_product = np.sum(p * s, axis=-1)
        np.testing.assert_allclose(dot_product, 0, atol=1e-9)

    @given(
        theta=angle_floats,
        phi=angle_floats,
        )
    def test_s_dot_i_is_zero(self, theta, phi):
        '''
        s-direction dot i-direction (emission direction from the origin) should be 0 (within floating point error)
        '''
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        i_x, i_y, i_z = ct.spherical_to_cartesian_coords(np.array([[1, theta, phi]]))
        i = np.hstack((i_x, i_y, i_z))
        dot_product = np.sum(s * i, axis=-1)
        np.testing.assert_allclose(dot_product, 0, atol=1e-9)

    @given(
        theta=angle_floats,
        phi=angle_floats,
        )
    def test_p_dot_i_is_zero(self, theta, phi):
        '''
        p-direction dot i-direction should be 0 (within floating point error)
        '''
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        i_x, i_y, i_z = ct.spherical_to_cartesian_coords(np.array([[1, theta, phi]]))
        i = np.hstack((i_x, i_y, i_z))
        dot_product = np.sum(p * i, axis=-1)
        np.testing.assert_allclose(dot_product, 0, atol=1e-9)

    @given(
        theta=angle_floats,
        phi=angle_floats,
        )
    def test_s_dot_n_is_zero(self, theta, phi):
        '''
        s-direction dot surface normal direction should be 0 (within floating point error)
        '''
        p, s = miop.parabola_surface_polarization_directions(theta, phi)
        i_x, i_y, i_z = ct.spherical_to_cartesian_coords(np.array([[1, theta, phi]]))
        i = np.expand_dims(np.hstack((i_x, i_y, i_z)), axis=0)
        parabola_position = miop.parabola_position(i)
        parabola_normal = miop.parabola_normals(parabola_position)
        dot_product = np.sum(s * parabola_normal, axis=-1)
        np.testing.assert_allclose(dot_product, 0, atol=1e-9)


class FresnelReflectionCoefficientsTest(unittest.TestCase):
    @given(
        normal=nonzero_3_vector,
        e_incident_direction=nonzero_3_vector,
        n_mirror=positive_sane_float,
        n_environment=positive_sane_float,
        )
    def test_fresnel_reflection_coefficients_magnitude(
        self,
        normal,
        e_incident_direction,
        n_mirror,
        n_environment):
        '''
        magnitudes of r_s and r_p should be between 0 and 1, inclusive
        '''
        hypothesis.assume(n_environment < n_mirror)
        wavelength=800e-9
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75,
            thetacutoffhole=4.,
            dielectric=np.array([[1, n_mirror**2, 0],[2, n_mirror**2, 0],[3, n_mirror**2, 0],[4, n_mirror**2, 0]])
            )
        r_s, r_p = miop.get_mirror_reflection_coefficients(
            wavelength, normal, e_incident_direction, mirror=mirror, n_environment=n_environment)
        assert np.abs(r_s) <= 1
        assert np.abs(r_p) <= 1


class ARMaskCalcTest(unittest.TestCase):
    @given(
        st.floats(min_value=np.pi/2, max_value=3*np.pi/2, allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False))
    def testARMaskCalcNegativeZTrue(self, theta, phi):
        '''
        all inputs in negative Z space should be True
        '''
        calculated = miop.ar_mask_calc(
            theta,
            phi,
            holein=True,
            slit=None,
            slit_center=None,
            orientation=0)
        self.assertTrue(calculated)


class GetMirrorReflectedFieldTest(unittest.TestCase):
    '''
    - x-component of reflected_e should be 0
    - reflected e direction should be along -x, with y and z near 0
    '''
#    @given(
#        incident_vector=k_vector,
#        complex_e=nonzero_3_vector,
#        )
#    def test_reflected_e_x_is_0_complex_incident(self, incident_vector, complex_e):
#        '''
#        Check that the x-component of the reflected field is 0, within tolerance
#        '''
#        wavelength = 800e-9
#        mirror = miop.AMOLF_MIRROR
#        n_environment = 1
#        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
#        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
#        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(incident_direction, incident_e, wavelength, n_environment, mirror)
#        print(incident_direction, incident_e, reflected_e_s, reflected_e_p)
#        np.testing.assert_allclose(reflected_e_s[:, 0], 0, atol=1e-8)
#        np.testing.assert_allclose(reflected_e_p[:, 0], 0, atol=1e-8)

    @given(
        incident_vector=k_vector,
        )
    def test_reflected_e_x_is_0_real_incident(self, incident_vector):
        '''
        Check that the x-component of the reflected field is 0, within tolerance
        '''
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        n_environment = 1
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :])
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(incident_direction, incident_e, wavelength, n_environment, mirror)
        print(incident_direction, incident_e, reflected_e_s, reflected_e_p)
        np.testing.assert_allclose(reflected_e_s[:, 0], 0, atol=1e-8)
        np.testing.assert_allclose(reflected_e_p[:, 0], 0, atol=1e-8)

#    @given(
#        incident_vector=k_vector,
#        complex_e=nonzero_3_vector,
#        )
#    def test_reflected_vector_on_x(self, incident_vector, complex_e):
#        '''
#        Check that the reflected field propagates along -x
#        '''
#        wavelength = 800e-9
#        mirror = miop.AMOLF_MIRROR
#        n_environment = 1
#        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
#        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
#        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(incident_direction, incident_e, wavelength, n_environment, mirror)

class MultiFunctionTest(unittest.TestCase):
    '''
    - incident e_s + incident e_p should equal incident_e
    - magnitude of incident_e should be the greater than or equal to magnitude of reflected_e_s + reflected_e_p
    - magnitude of incident_e should be equal to magnitude of reflected_e_s + reflected_e_p, given a real dielectric function
    '''
    @given(
        incident_vector=k_vector,
        complex_e=nonzero_3_vector,
        )
    def test_incident_e_magnitudes(self, incident_vector, complex_e):
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) #+ 1j*complex_e
        print(incident_vector, incident_e, complex_e)
        parabola_positions = miop.parabola_position(incident_direction)
        surface_normal = miop.parabola_normals(parabola_positions)
        n_surface = miop.get_mirror_refractive_index(wavelength, mirror=mirror)
        r_s, r_p = miop.get_mirror_reflection_coefficients(
            wavelength=wavelength, 
            normal=surface_normal, 
            e_incident_direction=incident_direction, 
            mirror=mirror, 
            n_environment=1
            )
        print(r_s, r_p)
        incident_r, incident_theta, incident_phi = ct.cartesian_to_spherical_coords(incident_direction)
        p_direction, s_direction = miop.parabola_surface_polarization_directions(
            incident_theta, 
            incident_phi)
        incident_e_s = np.sum(s_direction * incident_e, axis=-1, keepdims=True) * incident_e
        incident_e_p = np.sum(p_direction * incident_e, axis=-1, keepdims=True) * incident_e
        incident_e_sum = incident_e_s + incident_e_p
        print(incident_e_s, incident_e_p)
        incident_e_magnitude = ct.field_magnitude(incident_e)
        incident_e_sum_magnitude = ct.field_magnitude(incident_e_sum)
        np.testing.assert_allclose(incident_e_sum_magnitude, incident_e_magnitude, atol=1e-7)

if __name__== '__main__':
    unittest.main()
