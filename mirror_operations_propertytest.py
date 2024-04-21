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
    if len(np.shape(direction)) == 1:
        direction = np.expand_dims(direction, axis=0)
    direction = direction/ct.field_magnitude(direction, keepdims=True)
    r, theta, phi = ct.cartesian_to_spherical_coords(direction)
    cone_condition = np.logical_not(
        np.logical_and(abs(theta-np.pi/2)<5e-5, abs(phi-np.pi)<5e-5)
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
sane_k_floats = st.floats(min_value=-1e8, max_value=1e8).filter(lambda x: np.abs(x)>1e-10)
k_vector = npstrat.arrays(dtype=np.float64, shape=(2,3), elements=sane_k_floats).filter(lambda vec: np.logical_not(np.allclose(vec[0, :], vec[1, :]))).filter(lambda vec: negative_x_axis_cone(vec[0, :]))
k_vector_one = npstrat.arrays(dtype=np.float64, shape=(1,3), elements=sane_k_floats).filter(lambda vec: negative_x_axis_cone(vec))

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
        All the normals of the parabola should point towards -x
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
    - x-component of reflected_e should be 0, given real e and complex e
    - reflected e direction should be along -x, with y and z near 0
    - magnitude of reflected_e_s + reflected_e_p should be less than or equal to the magnitude of the incident e field
    '''

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
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        if ct.field_magnitude(reflected_e_s) != 0:
            reflected_e_s_norm = reflected_e_s/ct.field_magnitude(np.real(reflected_e_s))
        else:
            reflected_e_s_norm = reflected_e_s

        if ct.field_magnitude(reflected_e_p) != 0:
            reflected_e_p_norm = reflected_e_p/ct.field_magnitude(np.real(reflected_e_p))
        else:
            reflected_e_p_norm = reflected_e_p
        np.testing.assert_allclose(reflected_e_s_norm[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(reflected_e_p_norm[:, 0], 0, atol=1e-6)

    @given(
        incident_vector=k_vector,
        complex_e=nonzero_3_vector,
        )
    def test_reflected_e_x_is_0_imaginary_incident(self, incident_vector, complex_e):
        '''
        Check that the x-component of the reflected field is 0, within tolerance
        '''
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        n_environment = 1
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        if ct.field_magnitude(reflected_e_s) != 0:
            reflected_e_s_norm = reflected_e_s/ct.field_magnitude(np.real(reflected_e_s))
        else:
            reflected_e_s_norm = reflected_e_s

        if ct.field_magnitude(reflected_e_p) != 0:
            reflected_e_p_norm = reflected_e_p/ct.field_magnitude(np.real(reflected_e_p))
        else:
            reflected_e_p_norm = reflected_e_p
        np.testing.assert_allclose(np.real(reflected_e_s_norm[:, 0]), 0, atol=1e-6)
        np.testing.assert_allclose(np.real(reflected_e_p_norm[:, 0]), 0, atol=1e-6)

    @given(
        incident_vector=k_vector,
        complex_e=nonzero_3_vector,
        )
    def test_reflected_vector_on_x(self, incident_vector, complex_e):
        '''
        Check that the reflected field propagates along -x
        '''
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        n_environment = 1
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        self.assertTrue(reflected_direction[:, 0] <= 0)
        reflected_direction_norm = reflected_direction/ct.field_magnitude(reflected_direction)
        np.testing.assert_allclose(reflected_direction_norm[:, 1], 0, atol=1e-6)
        np.testing.assert_allclose(reflected_direction_norm[:, 2], 0, atol=1e-6)

    @given(
        incident_vector=k_vector,
        complex_e=nonzero_3_vector,
        )
    def test_reflected_e_magnitude_vs_incident_e_magnitude(self, incident_vector, complex_e):
        '''
        Check that the magnitude of the total reflected e-vector is less than or equal to the magnitude of the incoming vector
        '''
        wavelength = 800e-9
        mirror = miop.AMOLF_MIRROR
        n_environment = 1
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        incident_e_magnitude = ct.field_magnitude(incident_e)
        reflected_e_magnitude = ct.field_magnitude(reflected_e_s + reflected_e_p)
        self.assertTrue(reflected_e_magnitude <= incident_e_magnitude)

    @given(
        incident_vector=k_vector,
        complex_e=nonzero_3_vector,
        )
    def test_reflected_e_magnitude_vs_incident_e_magnitude_real_dielectric(self, incident_vector, complex_e):
        '''
        Check that the magnitude of the total reflected e-vector is equal to or less than the magnitude of the incident electric field vector, given a real dielectric function
        '''
        wavelength = 800e-9
        mirror = miop.ParabolicMirror(a=0.1, dfoc=0.5, xcut=-10.75, thetacutoffhole=4., dielectric=np.array([[1, -1, 0],[2, -2, 0],[3,-3, 0],[4, -4, 0]]))
        n_environment = 1
        incident_direction = np.expand_dims(incident_vector[0, :], axis=0)
        incident_e = np.cross(incident_vector[0, :], incident_vector[1, :]) + 1j*complex_e
        reflected_e_s, reflected_e_p, reflected_direction = miop.get_mirror_reflected_field(
            incident_direction, incident_e, wavelength, n_environment, mirror)
        incident_e_magnitude = ct.field_magnitude(incident_e)
        reflected_e_magnitude = ct.field_magnitude(reflected_e_s + reflected_e_p)
        self.assertTrue(np.logical_or(
            np.isclose(reflected_e_magnitude, incident_e_magnitude, atol=1e-6),
            reflected_e_magnitude < incident_e_magnitude)
            )


class MultiFunctionTest(unittest.TestCase):
    '''
    - incident e_s + incident e_p should equal incident_e
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
        incident_r, incident_theta, incident_phi = ct.cartesian_to_spherical_coords(incident_direction)
        p_direction, s_direction = miop.parabola_surface_polarization_directions(
            incident_theta, 
            incident_phi)
        incident_e_s = np.sum(s_direction * incident_e, axis=-1, keepdims=True) * s_direction
        incident_e_p = np.sum(p_direction * incident_e, axis=-1, keepdims=True) * p_direction
        incident_e_sum = incident_e_s + incident_e_p
        incident_e_magnitude = ct.field_magnitude(incident_e)
        incident_e_sum_magnitude = ct.field_magnitude(incident_e_sum)
        np.testing.assert_allclose(incident_e_sum_magnitude, incident_e_magnitude, atol=1e-7)


class GetMuellerMatrixTest(unittest.TestCase):
    '''
    - (0,0) is in [0,1] and all other elements are between [-1,1]
    - (1,1) is symmetrically reflective over the x-axis
    - (2,2) is symmetrically reflective over the x-axis
    - result for phi and -phi should be either the same or negative of each other at any theta
    '''
    
    def positive_and_negative_stokes_vectors(self, theta, phi):
        '''
        Set up the vectors for a positive and negative S1 parameter, taking in theta and phi angles
        '''
        S1_positive_r, S1_positive_theta, S1_positive_phi = ct.spherical_to_cartesian_vector_field(
            theta,
            phi,
            0,
            1,
            0,
            )
        E_S1_positive = np.stack((S1_positive_r, S1_positive_theta, S1_positive_phi), axis=-1)
        S1_negative_r, S1_negative_theta, S1_negative_phi = ct.spherical_to_cartesian_vector_field(
            theta,
            phi,
            0,
            0,
            1,
            )
        E_S1_negative = np.stack((S1_negative_r, S1_negative_theta, S1_negative_phi), axis=-1)
        return E_S1_positive, E_S1_negative
    
    @given(
    incident_vector = k_vector_one
    )
    def test_output_range(self, incident_vector):
        r, theta, phi = ct.cartesian_to_spherical_coords(incident_vector)
        E_S1_positive, E_S1_negative = self.positive_and_negative_stokes_vectors(theta, phi)
        mueller_calc = np.real(miop.get_mirror_mueller_matrix(
            incident_vector,
            E_S1_positive,
            E_S1_negative,
            wavelength=1800e-9,
            output_stokes_indices=(1,2),
        ))
        assert np.all(np.logical_and(mueller_calc[0, :]<= 1, mueller_calc[0, :] >= 0))
        assert np.all(np.logical_and(mueller_calc<=1, mueller_calc>=-1))

    @given(
        incident_vector = k_vector_one
    )
    def test_reflection_symmetry(self, incident_vector):
        '''
        Compare a calculated Mueller matrix to a calculated Mueller matrix on the opposite side of the mirror (eg, with a negative y-value, but same x and z values)
        '''
        r, theta, phi = ct.cartesian_to_spherical_coords(incident_vector)
        incident_vector_flip_y = incident_vector*np.array([1, -1, 1])
        r_flip_y, theta_flip_y, phi_flip_y = ct.cartesian_to_spherical_coords(incident_vector_flip_y)
        E_S1_positive, E_S1_negative = self.positive_and_negative_stokes_vectors(theta, phi)
        E_S1_positive_flip_y, E_S1_negative_flip_y = self.positive_and_negative_stokes_vectors(theta_flip_y, phi_flip_y)
        mueller_calc = np.real(miop.get_mirror_mueller_matrix(
            incident_vector,
            E_S1_positive,
            E_S1_negative,
            wavelength=1800e-9,
            output_stokes_indices=(1,2),
        ))
        mueller_calc_flip_y = np.real(miop.get_mirror_mueller_matrix(
            incident_vector_flip_y,
            E_S1_positive_flip_y,
            E_S1_negative_flip_y,
            wavelength=1800e-9,
            output_stokes_indices=(1,2),
        ))
        symmetric_indices = np.array([0, 1, 4, 5, 10, 11, 14, 15])
        antisymmetric_indices = np.array([2, 6, 7, 8, 9, 13])
        print(incident_vector, theta, mueller_calc[antisymmetric_indices], -mueller_calc_flip_y[antisymmetric_indices])
        assert np.all(np.isclose(mueller_calc[symmetric_indices], mueller_calc_flip_y[symmetric_indices]))
        assert np.all(np.isclose(mueller_calc[antisymmetric_indices], -mueller_calc_flip_y[antisymmetric_indices]))

    @given(
        incident_vector = k_vector_one
    )
    def test_positive_x_greatest_intensity(self, incident_vector):
        '''
        The positive x-axis should have incident light at an angle of 90 degrees. This should reflect the greatest proportion of light (e.g., all other (0,0) Mueller matrix values should be less than this value)
        '''
        r, theta, phi = ct.cartesian_to_spherical_coords(incident_vector)
        
        # set up incident vector to test
        incident_vector_positive_x = np.array([[1, 0, 0]])
        E_S1_positive, E_S1_negative = self.positive_and_negative_stokes_vectors(theta, phi)
        
        # set up vector along positive x-axis
        r_positive_x, theta_positive_x, phi_positive_x = ct.cartesian_to_spherical_coords(incident_vector_positive_x)
        E_S1_positive_positive_x, E_S1_negative_positive_x = self.positive_and_negative_stokes_vectors(theta_positive_x, phi_positive_x)
        
        # calculate Mueller matrix for both vectors
        mueller_calc = np.real(miop.get_mirror_mueller_matrix(
            incident_vector,
            E_S1_positive,
            E_S1_negative,
            wavelength=1800e-9,
            output_stokes_indices=(1,2),
        ))
        mueller_calc_positive_x = np.real(miop.get_mirror_mueller_matrix(
            incident_vector_positive_x,
            E_S1_positive_positive_x,
            E_S1_negative_positive_x,
            wavelength=1800e-9,
            output_stokes_indices=(1,2),
        ))
        # compare the 0th element
        assert mueller_calc[0] <= mueller_calc_positive_x[0]


if __name__== '__main__':
    unittest.main()
