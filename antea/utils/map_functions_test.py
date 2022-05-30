import os
import pytest
import numpy  as np
import tables as tb

from pytest        import mark
from pytest        import raises
from numpy.testing import assert_allclose

from antea.utils.map_functions  import Map
from antea.utils.map_functions  import opt_nearest
from antea.utils.map_functions  import correction_writer
from antea.utils.map_functions  import load_corrections
from antea.utils.map_functions  import map_writer
from antea.utils.map_functions  import load_map

from invisible_cities.io.dst_io import load_dst


def test_map_raises_exception_when_data_is_invalid():
    x   = np.arange(  0, 10)
    y   = np.arange(-10,  0)
    z   = np.zeros ((x.size, y.size))
    u_z = np.ones  ((x.size, y.size))
    with raises(AssertionError):
        Map((x, y), z, u_z,
                   norm_strategy =  "index",
                   norm_opts     = {"index": (0, 0)},
                   **opt_nearest)


@pytest.fixture(scope='session')
def corr_toy_data(ANTEADATADIR):
    x = np.arange( 100,  200)
    y = np.arange(-200, -100)
    E = np.arange(1e4, 1e4 + x.size*y.size).reshape(x.size, y.size)
    U = np.arange(1e2, 1e2 + x.size*y.size).reshape(x.size, y.size)
    N = np.ones_like(U)

    corr_filename = os.path.join(ANTEADATADIR, "toy_corr.h5")
    return corr_filename, (x, y, E, U, N)


@pytest.fixture(scope='session')
def map_toy_data(ANTEADATADIR):
    nbins = [100, 200]
    xs    = np.array([np.linspace( 100,  200, n) for n in nbins])
    ys    = np.array([np.linspace(-200, -100, n) for n in nbins])
    us    = np.array([np.linspace( 0.1,  0.2, n) for n in nbins])

    corr_filename = os.path.join(ANTEADATADIR, "toy_map.h5")
    return corr_filename, (xs, ys, us)


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
  filename, (x, y, E, U, _) = corr_toy_data
  corr                      = load_corrections(filename,
                                               node = "XYcorrections",
                                               **normalization)
  assert corr == Map((x,y), E, U, **normalization)


@mark.parametrize("bins, pos",
                 ( (100, 0),
                   (200, 1)))
def test_map_writer(config_tmpdir, map_toy_data, bins, pos):
    output_file = os.path.join(config_tmpdir, "test_map.h5")

    _, (xs, ys, us) = map_toy_data

    group = "Radius"
    name  = f"f{bins}bins"

    with tb.open_file(output_file, 'w') as h5out:
        write = map_writer(h5out,
                           group      = group,
                           table_name = name)
        for x, y, u in zip(xs[pos], ys[pos], us[pos]):
            write(x, y, u)

    dst = load_dst(output_file,
                   group = group,
                   node  = name)

    assert_allclose(xs[pos], dst.PhiRms         .values)
    assert_allclose(ys[pos], dst.Rpos           .values)
    assert_allclose(us[pos], dst.RposUncertainty.values)


@mark.parametrize("bins, pos",
                 ( (100, 0),
                   (200, 1)))
def test_load_map(map_toy_data, bins, pos):
    filename, (sigmas, rs, us) = map_toy_data
    rmap                       = load_map(filename,
                                          group="Radius",
                                          node=f"f2pes{bins}bins")
    assert rmap == Map((sigmas[pos],), rs[pos], us[pos])
