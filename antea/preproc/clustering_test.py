import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from antea.preproc.clustering import compute_clusters


def test_compute_clusters():
    '''
    Check if the groups of clusters are done well. For data of same group it gives
    a number different from 0, and if it does not pertain to any group, it gives -1
    '''
    df = pd.DataFrame({'tcoarse_extended': [178, 179, 185, 187, 190, 230, 250, 255, 256, 260]})

    clusters = compute_clusters(df)
    expected = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1])

    np.testing.assert_array_equal(clusters, expected)
