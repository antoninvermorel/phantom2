"""This module defines a TrackedBlob class and related functions for manipulating blobs and their tracking."""

import numpy as np
from scipy.spatial.distance import pdist  # -> max_chord_threshold function

class TrackedBlob:
    """
    Class representing a tracked blob over time.

    Attributes
    ----------
        id : int
            Unique identifier of the tracked blob.
        times : list
            List of time points for which data is recorded.
        contours_coords : list
            List of contour coordinates per time.
        centers_of_mass : list
            List of centers of mass per time.
        v_x : list
            Velocity component in x direction per time.
        v_y : list
            Velocity component in y direction per time.
        is_truncated : list
            Flags indicating if the tracked blob contour is truncated per time.
        density_thresholds_used : list
            Density thresholds used for contour detection per time.
        max_densities : list
            Maximum densities in the tracked blob contour per time.
        contours_data : list
            Computed contour data per time.
        original_contours_data : list
            Original contour data per time.
        amplitudes : list
            Amplitude values per time.
        width_prop : list
            Width along the propagation direction per time.
        width_perp : list
            Width perpendicular to the propagation direction per time.
        parents : set
            Set of parent tracked blob ids.
        children : set
            Set of children tracked blob ids.
    """
    

    def __init__(self, blob_id, time_list, first_blob):
        """
        Initialize a TrackedBlob with initial data.

        Parameters
        ----------
            blob_id : int
                Unique ID of the tracked blob.
            time_list : list
                List of initial time points since the beginning of the tracking.
            first_blob : Blob
                The first Blob instance associated with this track.
        """
        self.id = blob_id
        self.times = list(time_list)
        n = len(self.times)
        
        self.contours_coords = [None] * (n - 1) + [first_blob.contour_coords]
        self.centers_of_mass = [None] * (n - 1) + [first_blob.center_of_mass]
        self.v_x = [None] * n
        self.v_y = [None] * n
        self.is_truncated = [None] * (n - 1) + [first_blob.is_truncated]
        self.density_thresholds_used = [None] * (n - 1) + [first_blob.density_threshold_used]
        self.max_densities = [None] * (n - 1) + [first_blob.max_density]
        self.contours_data = [None] * (n - 1) + [first_blob.contour_data]
        self.original_contours_data = [None] * (n - 1) + [first_blob.original_contour_data]
        self.amplitudes = [None] * n
        self.width_prop = [None] * n
        self.width_perp = [None] * n

        self.parents = set()
        self.children = set()
            
    
    def add_data(self, time, blob):
        """
        Add data for a new time point to the tracked blob.

        Parameters
        ----------
            time : float
                The time point at which the blob data is added.
            blob : Blob
                The Blob instance containing the data to add.

        Raises
        ------
            ValueError
                If data for the given time already exists.
        """
        if time in self.times:
            raise ValueError(f"Time {time} already in times list => trackedblob {self.id} has already data for this time.")
            
        v_coords = self.compute_velocity(time, blob)                                
        self.times.append(time)
        self.contours_coords.append(blob.contour_coords)
        self.centers_of_mass.append(blob.center_of_mass)
        self.v_x.append(v_coords[0])
        self.v_y.append(v_coords[1])
        self.is_truncated.append(blob.is_truncated)
        self.density_thresholds_used.append(blob.density_threshold_used)
        self.max_densities.append(blob.max_density)
        self.contours_data.append(blob.contour_data)
        self.original_contours_data.append(blob.original_contour_data)
        self.amplitudes.append(blob.amplitude)
        self.width_prop.append(blob.w_prop)
        self.width_perp.append(blob.w_perp)


    def lifetime(self):
        """
        Compute the lifetime of the tracked blob.

        Returns
        -------
            float
                The duration between the first and last valid center of mass.
                Returns 0.0 if there are less than two valid points.
        """
        active_times = [t for t, com in zip(self.times, self.centers_of_mass) if com is not None]
        if len(active_times) < 2:
            return 0.0  # ou None si tu préfères indiquer une durée non définie
        return active_times[-1] - active_times[0]
    
    
    def is_active(self):
        """
        Check if the tracked blob is active.

        Returns
        -------
            bool
                True if there is at least one valid center of mass, False otherwise.
        """
        return len(self.times) > 0 and self.centers_of_mass[-1] is not None
    
    
    def compute_velocity(self, time, blob):
        """
        Compute the velocity vector of the blob between the last recorded time and the current time.

        Parameters
        ----------
            time : float
                The current time.
            blob : Blob
                The Blob instance corresponding to the tracke blob at the current time.

        Returns
        -------
            list
                Velocity vector [v_x, v_y]. Returns [None, None] if velocity cannot be computed.
        """
        v_coords = [None, None]
        if (len(self.is_truncated) > 0 and 
            self.is_truncated[-1] is not None and
            blob.center_of_mass is not None and
            blob.is_truncated is not None and
            time != self.times[-1]):
            
            if not self.is_truncated[-1] and not blob.is_truncated:
                v_coords = (self.centers_of_mass[-1]-blob.center_of_mass)/(time-self.times[-1])
                
        return v_coords
    
    
    def indiv_dist_threshold(self, tolerance_factor, use_original_contour=False):
        """
        Compute the individual distance threshold for linking blobs, 
        considering the tolerance factor and optionally the original contour data.

        Parameters
        ----------
            tolerance_factor : float
                Multiplicative tolerance factor (>=1).
            use_original_contour : bool
                If True, use original contour data instead of blob contour data for calculation.

        Returns
        -------
            float
                The computed distance threshold corresponding to the tracked blob.

        Raises
        ------
            ValueError
                If the tracked blob is inactive.
        """
        if not self.is_active():
            raise ValueError("TrackedBlob must be active to compute its distance threshold.")
        
        if use_original_contour:
            original_data = self.original_contours_data[-1]
            contour_coords = original_data["coords"]
            density_threshold_used = original_data["rho_thresh_used"]
            max_density = original_data["max_density"]

        else:
            contour_coords = self.contours_coords[-1]
            density_threshold_used = self.density_thresholds_used[-1]
            max_density = self.max_densities[-1]
                
        characteristic_radius_threshold = max_chord_threshold(contour_coords, tolerance_factor, density_threshold_used, max_density)

        v_x = self.v_x[-1]
        v_y = self.v_y[-1]
    
        if v_x is not None and v_y is not None:  # => len(self.times) > 1
            delta_t = self.times[-1] - self.times[-2]
            dyna_dist = np.sqrt(v_x**2 + v_y**2) * delta_t
            return dynamic_threshold(dyna_dist, tolerance_factor, characteristic_radius_threshold)
            
        return characteristic_radius_threshold
                

def max_chord_threshold(contour_coords, tolerance_factor, density_threshold, max_density):
    """
    Calculate a characteristic radius threshold based on the maximum chord length of the contour,
    adjusted by density threshold corrective and tolerance factors.

    Parameters
    ----------
        contour_coords : np.ndarray
            Array of contour coordinates.
        tolerance_factor : float
            Multiplicative tolerance factor (>=1).
        density_threshold : float
            Density threshold used for contour detection.
        max_density : float
            Maximum density into the contour.

    Returns
    -------
        float or None
            The computed characteristic radius threshold, or None if input invalid.

    Raises
    ------
        ValueError
            If tolerance_factor < 1 or densities are invalid.
    """
    if tolerance_factor < 1:
        raise ValueError(f"{tolerance_factor} has to be superior or equal to 1")
    if density_threshold <= 0 or max_density <= 0:
        raise ValueError("Densities must be strictly positive!")
    if max_density <= density_threshold:
        raise ValueError("density_max must be > density_threshold!")
    if contour_coords is None or len(contour_coords) < 2:
        return None
        
    max_chord_length = np.max(pdist(contour_coords))
    if np.isnan(max_chord_length) or np.isinf(max_chord_length):
        return None
    
    indiv_dist_threshold = tolerance_factor * max_chord_length / 2  # characteristic radius distance threshold
    corrective_factor = 1 / np.sqrt(np.log(max_density / density_threshold))  # corrective factor for density_threshold changes
    return indiv_dist_threshold * corrective_factor


def dynamic_threshold(dyna_dist, tolerance_factor, characteristic_radius_threshold):
    """
    Calculate a dynamic distance threshold that adapts to center of mass displacement .

    Parameters
    ----------
        dyna_dist : float
            Distance covered based on velocity and elapsed time.
        tolerance_factor : float
            Multiplicative tolerance factor (>=1).
        characteristic_radius_threshold : float
            Baseline characteristic radius threshold.

    Returns
    -------
        float
            The adjusted dynamic threshold.

    Raises
    ------
        ValueError
            If tolerance_factor < 1.
    """
    if tolerance_factor < 1:
        raise ValueError(f"{tolerance_factor} has to be superior or equal to 1")
        
    dyna_threshold = tolerance_factor * dyna_dist
    min_dyna_threshold = 0.1 * characteristic_radius_threshold
    if dyna_threshold < min_dyna_threshold:
        dyna_threshold = min_dyna_threshold
    return dyna_threshold























