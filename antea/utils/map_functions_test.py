import os
import pytest
import numpy  as np
import tables as tb

from pytest        import fixture
from pytest        import mark
from collections   import namedtuple
from numpy.testing import assert_allclose

from antea.utils.map_functions  import Map
from antea.utils.map_functions  import load_corrections
from antea.utils.map_functions  import correction_writer

from invisible_cities.io.dst_io import load_dst


@pytest.fixture(scope='session')
def corr_toy_data(ANTEADATADIR):
    x = np.arange(100, 200)
    y = np.arange(-200, -100)
    E = np.arange(1e4, 1e4 + x.size*y.size).reshape(x.size, y.size)
    U = np.arange(1e2, 1e2 + x.size*y.size).reshape(x.size, y.size)
    N = np.ones_like(U)

    corr_filename = os.path.join(ANTEADATADIR, "toy_corr.h5")
    return corr_filename, (x, y, E, U, N)


@mark.parametrize("normalization",
                  ({}, {"norm_strategy": "const",
                        "norm_opts"    : {"value": 1}}))
def test_load_corrections(corr_toy_data, normalization):
  filename, true_data = corr_toy_data
  x, y, E, U, _       = true_data
  corr                = load_corrections(filename,
                                         node = "XYcorrections",
                                         **normalization)
  assert corr == Map((x,y), E, U, **normalization)
