import numpy as np

def create_sphere(radius: float) -> np.ndarray:
    """
    Create a spherical volume of specified radius, sampled at 1mm resolution.
    Note: quite inefficient - there is likely a better way to do this.

    :param radius: the radius of the spherical volume, in mm
    """

    # Create a volume large enough to hold the sphere.
    s = np.rint(radius*2).astype('int')
    if(s % 2 == 0): s += 1
    vol = np.zeros([s,s,s])

    # Fill the sphere.
    for ii in range(s**3):

        nnx = int(ii / s**2)
        nny = int(ii/s) % s
        nnz = int(ii) % s

        x = s*(1.0*(nnx)/s - 0.5)
        y = s*(1.0*(nny)/s - 0.5)
        z = s*(1.0*(nnz)/s - 0.5)

        if(x**2 + y**2 + z**2 < radius**2):
            vol[nnx,nny,nnz] += 1

    return vol

def create_cylinder(radius: float, halfheight: float) -> np.ndarray:
    """
    Create a cylindrical volume of specified radius and height.
    Note: quite inefficient - there is likely a better way to do this.

    :param radius: the radius of the cylindrical volume, in mm
    :param halfheight: one half of the height of the cylindrical volume
    """

    # Create a volume large enough to hold the sphere.
    s = np.rint(radius*2).astype('int')
    if(s % 2 == 0): s += 1
    h = np.rint(halfheight*2).astype('int')
    if(h % 2 == 0): h += 1
    vol = np.zeros([s,s,h])

    # Fill the cylinder.
    for ii in range(s**2*h):

        nnx = int(ii / (s*h))
        nny = int(ii/h) % s
        nnz = int(ii) % h

        x = s*(1.0*(nnx)/s - 0.5)
        y = s*(1.0*(nny)/s - 0.5)
        z = h*(1.0*(nnz)/h - 0.5)

        if(x**2 + y**2 < radius**2 and z > -halfheight and z < halfheight):
            vol[nnx,nny,nnz] += 1

    return vol


def circular_profile(img2d: np.ndarray, r: float, bins: int) -> np.ndarray:
    """
    Extract the circular profile with radius r from the specified 2D image.

    :param img2d: the 2D image from which to extract the profile
    :param r: the radius at which to draw the profile
    :param bins: the number of bins in the profile
    """

    # Get the center point.
    cx = int((img2d.shape[0]-1)/2)
    cy = int((img2d.shape[1]-1)/2)

    # Set up the profile histogram.
    phis = np.arange(0,2*np.pi,2*np.pi/bins)
    prof = np.zeros(bins)

    # Extract the profile by picking out values at radius r for different phi.
    for i,phi in enumerate(phis):

        x = cx + int(r*np.cos(phi))
        y = cy + int(r*np.sin(phi))

        prof[i] = img2d[x,y]

    return prof

class phantom:
    """
    Stores a phantom distribution. The phantom can be read in

    The phantom is stored in a numpy file with a single key, 'phantom',
    corresponding to a 3D numpy array containing the distribution.

    :param Nx: length of the x-dimension of the phantom volume (in voxels); irrelevant if a file is specified
    :param Ny: length of the y-dimension of the phantom volume (in voxels); irrelevant if a file is specified
    :param Nz: length of the z-dimension of the phantom volume (in voxels); irrelevant if a file is specified
    :param phantom_file: file containing the phantom volume, if the phantom is to be read from file
    """

    def __init__(self, Nx: int = 10, Ny: int = 10, Nz: int = 10, phantom_file: str = ''):

        # Read the phantom from file.
        if(phantom_file == ''):
            self.vol = np.ones([Nx,Ny,Nz])
        else:
            fn = np.load(phantom_file)
            self.vol = fn['phantom']

        Nx = self.vol.shape[0]
        Ny = self.vol.shape[1]
        Nz = self.vol.shape[2]
        Lx = float(Nx)
        Ly = float(Ny)
        Lz = float(Nz)
        NyNz = Ny*Nz

        # Save the parameters as class variables.
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.NyNz = NyNz

    def get_vol(self) -> np.ndarray:
        """
        Return the 3D phantom volume.
        """
        return self.vol

    def get_cdist(self) -> np.ndarray:
        """
        Return the cumulative distribution.
        """
        nvol = self.vol.astype('float64') / np.sum(self.vol)
        return nvol.flatten()

    def save(self, fname: str):
        """
        Save the phantom volume to a numpy file.

        :param fname: the phantom file name
        """
        np.savez(fname, phantom=vol)

    def add_to_vol(self, vol_add: np.ndarray, x_offset: int, y_offset: int, z_offset: int):
        """
        Adds vol_add to the phantom volume at the specified offset. Note that
        the number of voxels in the x-, y-, and z-dimensions of the volume
        must be

        :param vol_add: the volume to be added
        :param x_offset: the x-offset from the central point
        :param y_offset: the y-offset from the central point
        :param z_offset: the z-offset from the central point
        """

        half_x = vol_add.shape[0] - 1
        if(half_x % 2 != 0):
            print("ERROR: half_x = {} not divisible by 2".format(half_x))
            return None
        half_x = int(half_x/2)

        half_y = vol_add.shape[1] - 1
        if(half_y % 2 != 0):
            print("ERROR: half_y = {} not divisible by 2".format(half_y))
            return None
        half_y = int(half_y/2)

        half_z = vol_add.shape[2] - 1
        if(half_z % 2 != 0):
            print("ERROR: half_z = {} not divisible by 2".format(half_z))
            return None
        half_z = int(half_z/2)

        x0 = int((self.vol.shape[0]-1)/2) + x_offset
        y0 = int((self.vol.shape[1]-1)/2) + y_offset
        z0 = int((self.vol.shape[2]-1)/2) + z_offset

        self.vol[x0-half_x:x0+half_x+1,y0-half_y:y0+half_y+1,z0-half_z:z0+half_z+1] += vol_add
