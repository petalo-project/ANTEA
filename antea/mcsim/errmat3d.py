import numpy as np

# Class to store the error matrix and relevant information.
# Note that errors are always expressed in (true - reco).
class errmat3d:

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


    # Select a random error for the specified coordinate.
    def get_random_error(self, x, y):
        i = int((x - self.xmin)/self.dx)
        j = int((y - self.ymin)/self.dy)
        if i >= len(self.errmat): i = len(self.errmat)-1
        if j >= len(self.errmat[0]): j = len(self.errmat[0])-1
        edist = self.errmat[i,j]
        if all(p == 0 for p in edist):
            return None
        k = np.random.choice(len(edist), p=edist)
        return self.zmin + (k + np.random.uniform())*self.dz
