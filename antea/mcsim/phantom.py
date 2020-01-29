import numpy as np

class phantom:
    """
    Stores a simulated phantom distribution.

    The phantom is stored in a numpy file with a single key, 'phantom',
    corresponding to a 3D numpy array containing the distribution.
    """

    def __init__(self,phantom_file=''):

        # Read the phantom from file.
        if(phantom_file == ''):
            npimg = np.ones([10,10,10])
        else:
            fn = np.load(phantom_file)
            npimg = fn['phantom']

        # Normalize the phantom.
        npimg = npimg.astype('float64') / np.sum(npimg)

        # Compute relevant phantom parameters.
        Nx = npimg.shape[0]
        Ny = npimg.shape[1]
        Nz = npimg.shape[2]
        Lx = float(Nx)
        Ly = float(Ny)
        Lz = float(Nz)
        NyNz = Ny*Nz
        cdist = npimg.flatten()  # 1D cumulative distribution

        # Save the parameters as class variables.
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.NyNz = NyNz
        self.cdist = cdist
