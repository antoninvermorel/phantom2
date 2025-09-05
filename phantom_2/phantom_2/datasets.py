"""This module manages dataset reading and domain metadata extraction."""
    
class Domain:   
    """
    Represents the spatial domain extracted from a dataset.

    Attributes
    ----------
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    Lx : float
        Total length of the domain in the x-direction.
    Ly : float
        Total length of the domain in the y-direction.
    Nx : int
        Number of grid points in the x-direction.
    Ny : int
        Number of grid points in the y-direction.
    """
    
    
    def __init__(self, ds):
        """
        Initialize Domain parameters based on dataset coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset from which to extract spatial information.
        """
        self.dx = ds.x.values[1] - ds.x.values[0]
        self.dy = ds.y.values[1] - ds.y.values[0]
        self.Lx = ds.x.values[-1] - ds.x.values[0] + self.dx
        self.Ly = ds.y.values[-1] - ds.y.values[0] + self.dy
        self.Nx = len(ds.x)
        self.Ny = len(ds.y)

class Dataset:
    """
    Handles loading and interaction with the dataset file.

    Attributes
    ----------
    ds : xarray.Dataset
        The loaded dataset.
    domain : Domain
        The spatial domain extracted from the dataset.
    times : np.ndarray
        Array of time points available in the dataset.
    x : np.ndarray
        Array of x-coordinates in the dataset.
    y : np.ndarray
        Array of y-coordinates in the dataset.
    """
    
    
    def __init__(self, dataset_name):
        """
        Load the dataset from the specified file and initialize domain metadata.
        The dataset is fully loaded in memory and the file handle is closed.

        Parameters
        ----------
        dataset_name : str
            Path to the dataset file.
        """
        import xarray as xr
        with xr.open_dataset(dataset_name) as ds:
            self.ds = ds.load()
        self.domain = Domain(self.ds)
        self.times = self.ds.t.values
        self.x = self.ds.x.values
        self.y = self.ds.y.values

    def __getattr__(self, name):
        """
        Access dataset variables as attributes if they exist.

        Parameters
        ----------
        name : str
            Attribute name to retrieve from the dataset.

        Returns
        -------
        xarray.DataArray
            Dataset variable corresponding to 'name'.

        Raises
        ------
        AttributeError
            If the attribute is not found in the dataset.
        """
        if name in self.ds:
            return self.ds[name]
        raise AttributeError(f"'Dataset' object has no attribute '{name}'")
        
    def get_frame(self, t_index):
        """
        Retrieve the data frame at a specific time index.

        Parameters
        ----------
        t_index : int
            Index of the time step to retrieve.

        Returns
        -------
        xarray.DataArray
            Data array corresponding to the specified time frame.
        """
        return self.ds.n.isel(t=t_index)
    
    
    # def get_next_dt(self, t_index):
    #     """
    #     Retrieve the next time step from a times list index (used for Kalman computation)

    #     Parameters
    #     ----------
    #     t_index : int
    #         times list index

    #     Returns
    #     -------
    #     float
    #         time step between the time corresponding to the time index and the next time.

    #     """
    #     if len(self.times) == t_index+1:
    #         return None
    #     else:
    #         return self.times[t_index+1] - self.times[t_index]
    
    
    
    
    
    
    
    
    
    

    
    
    
    