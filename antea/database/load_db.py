import sqlite3
import numpy as np
import pandas as pd
import os
from operator  import itemgetter
from functools import lru_cache


class DetDB:
    petalo = os.environ['ANTEADIR'] + '/database/localdb.PETALODB.sqlite3'

def tmap(*args):
    return tuple(map(*args))

def get_db(db):
    return getattr(DetDB, db, db)


@lru_cache(maxsize=10)
def DataSiPM(db_file, run_number=1e5, conf_label='P7R195Z140mm'):

    conn = sqlite3.connect(get_db(db_file))

    sql = '''select pos.SensorID, map.ElecID "ChannelID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, Z, Centroid "adc_to_pes", Sigma, PhiNumber, ZNumber
from ChannelPosition{0} as pos INNER JOIN ChannelGain{0} as gain
ON pos.SensorID = gain.SensorID INNER JOIN ChannelMapping{0} as map
ON pos.SensorID = map.SensorID INNER JOIN ChannelMatrix{0} as mtrx
ON pos.SensorID = mtrx.SensorID LEFT JOIN
(select * from ChannelMask{0} where MinRun <= {1} and {1} <= MaxRun) as msk
where pos.MinRun <= {1} and {1} <= pos.MaxRun
and gain.MinRun <= {1} and {1} <= gain.MaxRun
and mtrx.MinRun <= {1} and {1} <= mtrx.MaxRun
order by pos.SensorID
'''.format(conf_label, abs(run_number))
    data = pd.read_sql_query(sql, conn)
    conn.close()

    ## Add default value to Sigma for runs without measurement
    if not data.Sigma.values.any():
        data.Sigma = 2.24

    return data


@lru_cache(maxsize=10)
def DataSiPMfull_body(db_file, run_number=1e5, conf_label='P7R410Z1950mm'):

    conn = sqlite3.connect(get_db(db_file))

    sql = '''select pos.SensorID, pos.X, pos.Y, pos.Z, mtrx.PhiNumber, mtrx.ZNumber
from ChannelPosition{0} as pos INNER JOIN ChannelMatrix{0} as mtrx
ON pos.SensorID = mtrx.SensorID
where mtrx.MinRun <= {1} and {1} <= mtrx.MaxRun
order by pos.SensorID
'''.format(conf_label, abs(run_number))
    data = pd.read_sql_query(sql, conn)
    conn.close()

    return data
