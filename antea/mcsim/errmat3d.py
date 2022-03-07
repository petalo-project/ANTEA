import numpy as np

# Class to store the error matrix and relevant information.
# Note that errors are always expressed in (true - reco).
class errmat3d:
    """
    Class to store an error matrix and relevant information. Note that errors
    are always expressed in (true - reco).
    Error matrices are currently stored in a numpy file containing keys:
    'errmat3d': the 3D error matrix containing dimensions [x, y, z] = [coord1, coord2, err],
    where coord1 and coord2 specify a pair of coordinates and err the corresponding error;
    'xmin': the minimum coordinate 1 value;
    'ymin': the minimum coordinate 2 value;
    'zmin': the minimum error value;
    'dx': the coordinate 1 bin width;
    'dy': the coordinate 2 bin width;
    'dz': the error bin width

    The distribution of simulated coordinates is calculated by summing over the
    error dimension.
    """

    def __init__(self,errmat_file):

        # Load the error matrix from file.
        fn = np.load(errmat_file)
        errmat = fn['errmat']
        eff = fn['eff']
        xmin = fn['xmin']
        ymin = fn['ymin']
        zmin = fn['zmin']
        dx = fn['dx']
        dy = fn['dy']
        dz = fn['dz']

        # The coordinate matrix is a 2d array; it's the sum along the error dimension.
        self.coordmat = np.sum(errmat, axis=2)

        # Normalize the error matrix along the error dimension.
        for i in range(len(self.coordmat)):
            for j in range(len(self.coordmat[0])):
                if self.coordmat[i,j] != 0:
                    errmat[i,j,:] = errmat[i,j,:]/self.coordmat[i,j]
                else:
                    errmat[i,j,:] = 0

        # Normalize the coordinate matrix. Sum over both axes.
        self.coordmat /= np.sum(self.coordmat)

        # Save the relevant variables.
        self.errmat = errmat
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.dx = dx
        self.dy = dy
        self.dz = dz


    def get_random_error(self, x, y):
        """
        Select a random error for the specified coordinates.

        :param x: the first coordinate
        :param y: the second coordinate
        :type x: float
        :type y: float
        :returns: a random error corresponding to the specified coordinate
        :rtype: float
        """
        i = int((x - self.xmin)/self.dx)
        j = int((y - self.ymin)/self.dy)
        if i >= len(self.errmat): i = len(self.errmat)-1
        if j >= len(self.errmat[0]): j = len(self.errmat[0])-1
        edist = self.errmat[i,j]
        if all(p == 0 for p in edist):
            return None
        k = np.random.choice(len(edist), p=edist)
        return self.zmin + (k + np.random.uniform())*self.dz
