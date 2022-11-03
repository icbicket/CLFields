import cl_calcs
from hypothesis import given
from hypothesis.extra import numpy as npstrat
import hypothesis.strategies as st
import numpy as np
import unittest

sane_floats = st.floats(min_value=-1e16, max_value=1e16)
threeD_vector = npstrat.arrays(dtype=np.float64, shape=3, elements=sane_floats)

class StokesVectorTest(unittest.TestCase):
    @given(
        st.complex_numbers(max_magnitude=1e12, allow_infinity=False, allow_nan=False),
        st.complex_numbers(max_magnitude=1e12, allow_infinity=False, allow_nan=False))
    def test_StokesVectorMagnitudes(self, E1, E2):
        S0, S1, S2, S3 = cl_calcs.stokes_parameters(E1, E2)
        np.testing.assert_allclose(S0, np.sqrt(np.square(S1) + np.square(S2) + np.square(S3)), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
