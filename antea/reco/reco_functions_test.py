import numpy  as np
import pandas as pd

from hypothesis.strategies import floats
from hypothesis            import given

from . reco_functions import lower_or_equal, greater_or_equal


f             = floats(min_value=1, max_value=2)
f_lower       = floats(min_value=0, max_value=1)
allowed_error = floats(min_value=1.e-8, max_value=1.e-6)


@given(f, f_lower)
def test_lower_or_equal(f1, f2):
   assert lower_or_equal(f2, f1)

   
@given(f, f_lower)
def test_greater_or_equal(f1, f2):
   assert greater_or_equal(f1, f2)


@given(f, allowed_error)
def test_allowed_error_in_inequality(f1, err):

    f2 = f1 + 2*err
    assert not greater_or_equal(f1, f2, err)
