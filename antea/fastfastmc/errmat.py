import numpy as np

# Class to store the error matrix and relevant information.
# Note that errors are always expressed in (true - reco).
class errmat:

    def __init__(self,errmat_file):

        # Load the error matrix from file.
        fn = np.load(errmat_file)
        errmat = fn['errmat']
        eff = fn['eff']
        xmin = fn['xmin']
        ymin = fn['ymin']
        dx = fn['dx']
        dy = fn['dy']

        # The coordinate matrix is the sum along the error dimension.
        self.coordmat = np.sum(errmat,axis=1)

        # Normalize the error matrix along the error dimension.
        for i in range(len(self.coordmat)):
            errmat[i,:] = errmat[i,:]/self.coordmat[i]

        # Normalize the coordinate matrix.
        self.coordmat /= np.sum(self.coordmat)

        # Save the relevant variables.
        self.errmat = errmat
        self.xmin = xmin
        self.ymin = ymin
        self.dx = dx
        self.dy = dy

    # Select a random coordinate.
    def get_random_coord(self):
        i = np.random.choice(len(self.coordmat),p=self.coordmat)
        return self.xmin + (i + np.random.uniform())*self.dx

    # Select a random error for the specified coordinate.
    def get_random_error(self,x):
        i = int((x - self.xmin)/self.dx)
        if(i >= len(self.errmat)): i = len(self.errmat)-1
        edist = self.errmat[i]
        j = np.random.choice(len(edist),p=edist)
        return self.ymin + (j + np.random.uniform())*self.dy
