"""This module defines a Blob class"""

import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask

class Blob:
    """
    A class representing a detected blob with its contour and related properties.

    Attributes
    ----------
    contour_coords : np.ndarray or None
        Coordinates of the blob contour.
    center_of_mass : np.ndarray or None
        Center of mass of the blob contour.
    contour_features : dict or None
        Computed properties of the contour such as area, length, and convexity deficiency.
    is_truncated : bool or None
        Indicates if the blob touches/truncates the domain boundaries.
    density_threshold_used : float or None
        Density threshold used for blob contours detection.
    max_density : float or None
        Maximum density value inside the blob contour.
    original_contour_data : dict or None
        Original contour data stored for reference.
    local_density_array : xarray.DataArray or None
        Masked local density array inside the blob contour.
    amplitude : Any
        Placeholder attribute (not set in this snippet).
    w_prop : Any
        Placeholder attribute (not set in this snippet).
    w_perp : Any
        Placeholder attribute (not set in this snippet).
    """
    
    
    def __init__(self, time=None):
        """Initialize a Blob instance with all attributes set to None or default."""
        self.time = time
        self.contour_coords = None
        self.center_of_mass = None
        self.contour_features = None
        self.is_truncated = None
        self.density_threshold_used = None
        self.max_density = None
        self.original_contour_data = None
        self.local_density_array = None
        self.amplitude = None
        self.w_prop = None
        self.w_perp = None

    @classmethod
    def empty(cls, time):
        """
        Create an empty Blob instance.

        Returns
        -------
        Blob
            A Blob object with default uninitialized attributes.
        """
        return cls(time=time)
        
    @classmethod
    def from_contour(cls, time, contour_coords, domain, phantom_frame, density_threshold_used, original_contour_data=None):
        """
        Create a Blob instance from contour coordinates and compute derived properties.

        Parameters
        ----------
        contour_coords : np.ndarray
            Coordinates of the contour points.
        domain : Domain
            The spatial domain object containing size and resolution
        phantom_frame : xarray.DataArray
            The data array containing density values for the frame.
        density_threshold_used : float
            Density threshold used for contours detection.
        original_contour_data : dict, optional
            Original contour data to be stored (default is None).

        Returns
        -------
        Blob
            A fully initialized Blob object with computed properties.
        """
        blob = cls(time=time)
        blob.contour_coords = contour_coords
        blob.compute_contour_data()
        blob.compute_is_truncated(domain)
        blob.compute_local_density_array(phantom_frame, domain)
        blob.compute_max_density(blob.local_density_array)    
        blob.density_threshold_used = density_threshold_used
        blob.compute_original_contour_data(original_contour_data=original_contour_data)
        return blob
    
    
    def compute_original_contour_data(self, original_contour_data=None):
        """
        Store or compute the original contour data dictionary.

        Parameters
        ----------
        original_contour_data : dict, optional
            Original contour data to assign. If None, computes from current attributes.
        """
        if original_contour_data is not None:
            self.original_contour_data = original_contour_data
        else:                  
            self.original_contour_data = {
                "coords": self.contour_coords,
                "com": self.center_of_mass,
                "max_density": self.max_density,
                "is_truncated": self.is_truncated,
                "rho_thresh_used": self.density_threshold_used,
                "contour_features": self.contour_features
                }

    def compute_contour_data(self):
        """
        Compute and store geometric properties of the contour such as area, length, and convexity deficiency.
        Sets center_of_mass as the polygon centroid.
        """
        if self.contour_coords is None or len(self.contour_coords) < 4:
            return

        polygon = Polygon(self.contour_coords)
        if not polygon.is_valid or polygon.area == 0:
            return

        convex_hull = ConvexHull(self.contour_coords)
        cd = abs((convex_hull.volume - polygon.area) / convex_hull.volume)

        self.contour_features = {
            "area": polygon.area,
            "contour_length": polygon.length,
            "convexity_deficiency": cd,
        }
        self.center_of_mass = np.array([polygon.centroid.x, polygon.centroid.y])
        

    def compute_is_truncated(self, domain, buffer_factor=1.1):
        """
        Determine if the blob contour is truncated by the domain boundaries.

        Parameters
        ----------
        domain : Domain
            The spatial domain object containing size and resolution.
        buffer_factor : float, optional
            Factor to define the buffer zone for boundary detection (default is 1.1).
        """
        dx, dy = domain.dx, domain.dy
        tol_x = buffer_factor * dx / 2
        tol_y = buffer_factor * dy / 2

        contour = self.contour_coords
        touches_left = np.sum(contour[:, 0] <= tol_x)
        touches_right = np.sum(contour[:, 0] >= domain.Lx - tol_x)
        touches_bottom = np.sum(contour[:, 1] <= tol_y)
        touches_top = np.sum(contour[:, 1] >= domain.Ly - tol_y)

        self.is_truncated = (touches_left + touches_right + touches_bottom + touches_top) >= 2
        

    def compute_local_density_array(self, phantom_frame, domain):
        """
        Compute the local density array masked by the blob's contour within the domain.

        Parameters
        ----------
        phantom_frame : xarray.DataArray
            The data array containing density values for the frame.
        domain : Domain
            The spatial domain object containing size and resolution.
        """
        shape = (domain.Ny, domain.Nx)
        dy = domain.dy
        dx = domain.dx
    
        scaled_coords = np.column_stack((
            self.contour_coords[:, 1] / dy,
            self.contour_coords[:, 0] / dx
        ))
    
        mask = polygon2mask(shape, scaled_coords)
        if mask.sum() == 0:
            self.local_density_array = None
        else:
            self.local_density_array = phantom_frame.where(mask, other=0)
    
    
    def compute_max_density(self, phantom_frame):
        """
        Compute the maximum density value within the blob's local density array.
        
        Parameters
        ----------
        phantom_frame : xarray.DataArray or None
            The data array containing density values for the frame.
        """
        if phantom_frame is None:
            self.max_density = None
        else:
            self.max_density = phantom_frame.values.max()



    def get_observation(self, features, use_original_contour=False):
        """
        Construct the observation vector for Kalman filtering based on the blob's properties.
    
        Parameters
        ----------
        features : list of str
            List of contour feature names to include in the observation vector
            (e.g., 'area', 'length', 'convexity_deficiency').
        use_original_contour : bool, optional
            If True, use the original contour data stored in `original_contour_data`
            instead of the current contour attributes. Default is False.
    
        Returns
        -------
        np.ndarray, shape (n_features+2, 1)
            Observation vector containing:
            - x-coordinate of center of mass
            - y-coordinate of center of mass
            - values of the requested contour features in the order specified by `features`
    
        Raises
        ------
        ValueError
            - If 'self.contour_coords' is None (empty blob)
            - If a feature of 'features' is not valid
        """
        # handle empty blob 
        if self.contour_coords is None:
            return None
        
        # handle invalid features
        allowed_features = {"area", "convexity_deficiency", "contour_length"}
        for feature in features:
           if feature not in allowed_features:
               raise ValueError(
                        f"'features' requires features in {allowed_features}, "
                        f"got {features}.")
        
        # filling in observation vector
        if not use_original_contour:
            obs = [self.center_of_mass[0], self.center_of_mass[1]]
            for feat_name in features:
                val = self.contour_features.get(feat_name, None)
                if val is None:
                    raise ValueError(f"Blob doesn't contain the attribute : '{feat_name}'")
                obs.append(val)
        else:
            obs = [self.original_contour_data["com"][0], self.original_contour_data["com"][1]]
            for feat_name in features:
                val = self.original_contour_data["contour_features"].get(feat_name, None)
                if val is None:
                    raise ValueError(f"Blob doesn't contain the attribute : '{feat_name}'")
                obs.append(val)
            
        return np.array(obs).reshape(-1, 1)
























