import numpy  as np
import tables as tb

from functools                    import partial
from itertools                    import product
from collections                  import namedtuple
from scipy.interpolate            import griddata

from antea.core.exceptions        import ParameterNotSet
from invisible_cities.io.dst_io   import load_dst
from invisible_cities.io.table_io import make_table


class RPhiRmsDependence(tb.IsDescription):
    PhiRms          = tb.Float32Col(pos=0)
    Rpos            = tb.Float32Col(pos=0)
    RposUncertainty = tb.Float32Col(pos=0)


opt_nearest = {"interp_method": "nearest"}
opt_linear  = {"interp_method": "linear" ,
               "default_f"    :     1    ,
               "default_u"    :     0    }
opt_cubic   = {"interp_method":  "cubic" ,
               "default_f"    :     1    ,
               "default_u"    :     0    }

Measurement = namedtuple('Measurement', 'value uncertainty')

class Map:
    """
    Interface for accessing any kind of correspondence between variables.
    Parameters
    ----------
    xs : np.ndarray
        Array of coordinates corresponding to each correlation.
    fs : np.ndarray
        Array of correlations or the values used for computing them.
    us : np.ndarray
        Array of uncertainties or the values used for computing them.
    norm_strategy : False or string
        Flag to set the normalization option. Accepted values:
        - False:    Do not normalize.
        - "max":    Normalize to maximum energy encountered.
        - "index":  Normalize to the energy placed to index (i,j).
    default_f, default_u : floats
        Default correlation and uncertainty for missing values (where fs = 0).
    """

    def __init__(self,
                 xs, fs, us,
                 norm_strategy = None,
                 norm_opts     = {},
                 interp_method = "nearest",
                 default_f     = 0,
                 default_u     = 0):

        self._xs = [np.array( x, dtype=float) for x in xs]
        self._fs =  np.array(fs, dtype=float)
        self._us =  np.array(us, dtype=float)

        self.norm_strategy   = norm_strategy
        self.norm_opts       = norm_opts
        self.interp_method   = interp_method
        self.default_f       = default_f
        self.default_u       = default_u

        self._normalize        (  norm_strategy,
                                  norm_opts    )
        self._init_interpolator(interp_method  , default_f, default_u)

    def __call__(self, *xs):
        """
        Compute the correlation factor.
        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        # In order for this to work well both for arrays and scalars
        arrays = len(np.shape(xs)) > 1
        if arrays:
            xs = np.stack(xs, axis=1)

        value  = self._get_value      (xs).flatten()
        uncert = self._get_uncertainty(xs).flatten()
        return (Measurement(value   , uncert   ) if arrays else
                Measurement(value[0], uncert[0]))

    def _init_interpolator(self, method, default_f, default_u):
        coordinates           = np.array(list(product(*self._xs)))
        self._get_value       = partial(griddata,
                                        coordinates,
                                        self._fs.flatten(),
                                        method     = method,
                                        fill_value = default_f)

        self._get_uncertainty = partial(griddata,
                                        coordinates,
                                        self._us.flatten(),
                                        method     = method,
                                        fill_value = default_u)

    def _normalize(self, strategy, opts):
        if not strategy            : return

        elif   strategy == "const" :
            if "value" not in opts:
                raise ParameterNotSet("Normalization strategy 'const' requires"
                                      "the normalization option 'value'")
            f_ref = opts["value"]
            u_ref = 0

        elif   strategy == "max"   :
            flat_index = np.argmax(self._fs)
            mult_index = np.unravel_index(flat_index, self._fs.shape)
            f_ref = self._fs[mult_index]
            u_ref = self._us[mult_index]

        elif   strategy == "center":
            index = tuple(i // 2 for i in self._fs.shape)
            f_ref = self._fs[index]
            u_ref = self._us[index]

        elif   strategy == "index" :
            if "index" not in opts:
                raise ParameterNotSet("Normalization strategy 'index' requires"
                                      "the normalization option 'index'")
            index = opts["index"]
            f_ref = self._fs[index]
            u_ref = self._us[index]

        else:
            raise ValueError("Normalization strategy not recognized: {}".format(strategy))

        assert f_ref > 0, "Invalid reference value."

        valid    = (self._fs > 0) & (self._us > 0)
        valid_fs = self._fs[valid].copy()
        valid_us = self._us[valid].copy()

        # Redefine and propagate uncertainties as:
        # u(F) = F sqrt(u(F)**2/F**2 + u(Fref)**2/Fref**2)
        self._fs[ valid]  = f_ref / valid_fs
        self._us[ valid]  = np.sqrt((valid_us / valid_fs)**2 +
                                    (   u_ref / f_ref   )**2 )
        self._us[ valid] *= self._fs[valid]

        # Set invalid to defaults
        self._fs[~valid]  = self.default_f
        self._us[~valid]  = self.default_u

    def __eq__(self, other):
        for i, x in enumerate(self._xs):
            if not np.allclose(x, other._xs[i]):
                return False

        if not np.allclose(self._fs, other._fs):
            return False

        if not np.allclose(self._us, other._us):
            return False

        return True


class ZRfactors(tb.IsDescription):
    z            = tb.Float32Col(pos=0)
    r            = tb.Float32Col(pos=1)
    factor       = tb.Float32Col(pos=2)
    uncertainty  = tb.Float32Col(pos=3)
    nevt         = tb. UInt32Col(pos=4)


def zr_writer(hdf5_file, **kwargs):
    zr_table = make_table(hdf5_file,
                          fformat = ZRfactors,
                          **kwargs)

    def write_zr(zs, rs, fs, us, ns):
        row = zr_table.row
        for i, z in enumerate(zs):
            for j, r in enumerate(rs):
                row["z"]           = z
                row["r"]           = r
                row["factor"]      = fs[i,j]
                row["uncertainty"] = us[i,j]
                row["nevt"]        = ns[i,j]
                row.append()
    return write_zr


def zr_correction_writer(hdf5_file, * ,
                         group       = "Corrections",
                         table_name  = "ZRcorrections",
                         compression = 'ZLIB4'):
    return zr_writer(hdf5_file,
                     group        = group,
                     name         = table_name,
                     description  = "ZR corrections",
                     compression  = compression)


def load_zr_corrections(filename, *,
                        group = "Corrections",
                        node  = "ZRcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    z, r = np.unique(dst.z.values), np.unique(dst.r.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Map((z, r),
               f.reshape(z.size, r.size),
               u.reshape(z.size, r.size),
               **kwargs)


def map_writer(hdf5_file,
               group       = "Radius",
               table_name  = "rphirms_dep",
               data_type   = RPhiRmsDependence,
               description = "RPhiRms Dependence",
               compression = 'ZLIB4',
               xs = 'PhiRms',
               ys = 'Rpos',
               us = 'RposUncertainty'):

    my_table = make_table(hdf5_file,
                          group,
                          table_name,
                          data_type,
                          description,
                          compression)

    def write_map(phi_rms, r, u):
        row = my_table.row
        row[xs] = phi_rms
        row[ys] = r
        row[us] = u
        row.append()


def load_rpos(filename, group = "Radius",
                        node  = "f100bins"):
    dst = load_dst(filename, group, node)
    return Map((dst.RmsPhi     .values,),
                dst.Rpos       .values,
                dst.Uncertainty.values)
