import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from solver.CL_CD_from_coeff import get_CL_CD_from_coeff

class TestMesher(TestCase):
    def test_get_CL_CD_from_coeff(self):
        AR = 20
        AoA_deg = 10
        CL_expected, CD_ind_expected = get_CL_CD_from_coeff(AR, AoA_deg)

        assert_almost_equal(CL_expected, 0.974775743317)
        assert_almost_equal(CD_ind_expected, 0.018903384655)
