import numpy as np


class errmat:
    """
    Class to store an error matrix and relevant information. Note that errors
    are always expressed in (true - reco).
    Error matrices are currently stored in a numpy file containing keys:
    - 'errmat': the 2D error matrix containing dimensions [x,y] = [coord,err],
    where coord specifies a coordinate and err the corresponding error
    - 'xmin': the minimum coordinate value
    - 'ymin': the minimum error value`
    - 'dx': the coordinate bin width
    - 'dy': the error bin width
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

    def get_random_coord(self):
        """
        Select a random coordinate from the coordinate matrix.

        :rtype: float containing the randomly selected coordinate
        """
        i = np.random.choice(len(self.coordmat),p=self.coordmat)
        return self.xmin + (i + np.random.uniform())*self.dx

    def get_random_error(self,x):
        """
        Select a random error for the specified coordinate.
        
        :param x: the coordinate
        :type x: float
        :rtype: float containing a random error corresponding to the specified coordinate
        """
        i = int((x - self.xmin)/self.dx)
        if(i >= len(self.errmat)): i = len(self.errmat)-1
        edist = self.errmat[i]
        j = np.random.choice(len(edist),p=edist)
        return self.ymin + (j + np.random.uniform())*self.dy
