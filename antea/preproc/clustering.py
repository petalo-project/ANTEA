import numpy as np
from sklearn.cluster import DBSCAN

def compute_clusters(df):
    '''
    Assign a cluster id for each real event. Each real event is compound of
    values that do not differ in more than 10 units.
    '''

    values = df.tcoarse_extended.values
    values = values.reshape(values.shape[0],1)

    clusters = DBSCAN(eps=10, min_samples=2).fit(values)
    return clusters.labels_
