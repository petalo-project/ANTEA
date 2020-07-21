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


def test_correction_writer(config_tmpdir, corr_toy_data):
    output_file = os.path.join(config_tmpdir, "test_corr.h5")

    _, (x, y, F, U, N) = corr_toy_data

    group = "Corrections"
    name  = "XYcorrections"

    with tb.open_file(output_file, 'w') as h5out:
        write = correction_writer(h5out,
                                  group      = group,
                                  table_name = name)
        write(x, y, F, U, N)

    x, y    = np.repeat(x, y.size), np.tile(y, x.size)
    F, U, N = F.flatten(), U.flatten(), N.flatten()

    dst = load_dst(output_file,
                   group = group,
                   node  = name)
    assert_allclose(x, dst.X          .values)
    assert_allclose(y, dst.Y          .values)
    assert_allclose(F, dst.Factor     .values)
    assert_allclose(U, dst.Uncertainty.values)
    assert_allclose(N, dst.NEvt       .values)


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
