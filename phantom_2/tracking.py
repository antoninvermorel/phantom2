from .tracked_blobs import TrackedBlob
from .blobs import Blob
from .detection_methods import get_detection_method
from .matching_methods import get_matching_method

import numpy as np
from scipy.optimize import linear_sum_assignment

def tracking(ds_read, tracking_config):
    """
    Perform blob tracking over a sequence of frames using the specified detection 
    and matching methods.

    Parameters
    ----------
    ds_read : DatasetReader
        Reader object that provides access to the dataset frames and metadata.
    tracking_config : dict
        Dictionary of tracking parameters. Must be validated by `validate_config`. 
        Specifies density_threshold, detection method, association matching method, and their options.
    gif_fps : int, optional
        Frames per second for GIF output (if generated). Default is 8.

    Returns
    -------
    tracked_blobs : list of TrackedBlob
        List of tracked blobs containing their full temporal evolution across frames.

    Notes
    -----
    - The tracking procedure consists of:
        1. Iterating over all frames in the dataset.
        2. Detecting blobs in the current frame using the given `density_threshold`.
        3. Updating existing tracked blobs and associating them with new detections 
           via the Hungarian algorithm and the configured matching method.
        4. Handling blob interactions (merges/splits/appearances/disappearances).
    - Kalman filters (if configured) are used for prediction and gating.
    """
    validate_config(tracking_config)
    tracked_blobs = []
    domain = ds_read.domain
    all_times_seen = []
    
    for t_idx in range(len(ds_read.times)):
        time = ds_read.times[t_idx]
        all_times_seen.append(time)
        phantom_frame = ds_read.get_frame(t_idx)
        tracked_blobs = update_tracked_blobs(tracked_blobs, phantom_frame, all_times_seen, domain, tracking_config)

    return tracked_blobs


def validate_config(tracking_config: dict):
    """
    Validate only the configuration parts relevant to tracking.

    Raises
    ------
    ValueError
        If required parameters are missing or inconsistent.
    """
    # Density threshold check
    density_threshold = tracking_config.get("density_threshold")
    if density_threshold is None:
        raise ValueError("Tracking requires a positive 'density_threshold' in config.")        
    # Interactions detection method checks
    interac_detection_method = tracking_config.get("interac_detection_method")
    if interac_detection_method not in ("simple_gating", "complex_residual_gating", "subblobs"):
        raise ValueError(f"Unknown detection method: {interac_detection_method}")
    # Associations matching method checks
    allowed_matching_methods = {"euclidean", "mahalanobis"}
    assoc_matching_method = tracking_config.get("assoc_matching_method")
    if assoc_matching_method is None:
        raise ValueError("Blobs tracking requires 'assoc_matching_method' in tracking_config.")
    if assoc_matching_method not in allowed_matching_methods:
        raise ValueError(
            f"Invalid assoc_matching_method '{assoc_matching_method}'."
            f"Must be one of {allowed_matching_methods}."
        )

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
 
def update_tracked_blobs(tracked_blobs, phantom_frame, all_times_seen, domain, tracking_config):
    """
    Update the list of tracked blobs with new frame data, associating newly detected blobs 
    to existing tracked blobs using the method defined in tracking_config.

    Parameters
    ----------
    tracked_blobs : list of TrackedBlob
        List of currently tracked blob objects.
    phantom_frame : xr.DataArray
        2D array of density values for the current frame.
    all_times_seen : list of float
        Time values corresponding to all frames processed so far (including the current one).
    domain : Domain
        Domain object containing spatial extent and resolution attributes.
    tracking_config : dict
        Tracking configuration dictionary. Must be validated by `validate_config`.

    Returns
    -------
    tracked_blobs : list of TrackedBlob
        Updated list of tracked blobs, including merged/split/newly appeared/disappeared blobs.

    Notes
    -----
    - The association step uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) 
      to minimize the total distance between parent blobs and child blobs.
    """       
    # 1. Previous and new frames data extraction
    time_value = all_times_seen[-1]
    child_blobs = find_blobs(phantom_frame, time_value, tracking_config.get(("density_threshold")), domain)      
    parent_trackedblobs = [tb for tb in tracked_blobs if tb.is_active()]
    
    # 2. Tracking
    new_id = max((tb.id for tb in tracked_blobs), default=0) + 1

    if len(parent_trackedblobs) > 0:        
        if len(child_blobs) > 0:                
            # a/ Interactions detection
            interac_detection_method = get_detection_method(tracking_config["interac_detection_method"])
            merged_children_map, splitted_parents_map = interac_detection_method.get_interactions(parent_trackedblobs, child_blobs, tracking_config)
            
            # b/ Blobs associations
            assoc_matching_method = get_matching_method(tracking_config["assoc_matching_method"])
            assoc_matching_method.validate_config(tracking_config)
            cm_kwargs = {"context": "assoc"}            
            cost_matrix = assoc_matching_method.make_cost_matrix(parent_trackedblobs, child_blobs, **cm_kwargs)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            index_to_id = {}
            assigned_children = set()
            assigned_parents = set()
            
            # # DEBUG tests:
            # if np.isclose(parent_trackedblobs[0].times[-1], 8.30):
            #     print(f"update_tracked_blobs cost_matrix at t={time_value:.2f}: {cost_matrix}")
            
            # 1-1 associations
            for parent_index, child_index in zip(row_ind, col_ind):
                parent_tb = parent_trackedblobs[parent_index]
                cost = cost_matrix[parent_index, child_index]
                
                # # DEBUG tests:
                # if (parent_tb.id == 16 or parent_tb.id == 18) and np.isclose(parent_tb.times[-1], 8.30):
                #     print(f"update_tracked_blobs main 1-1 association for id {parent_tb.id} at t={time_value:.2f}")
                #     print(f"parent_com : {parent_tb.centers_of_mass[-1]} ; child_com : {child_blobs[child_index].center_of_mass}")
                
                if assoc_matching_method.gating(cost, tracking_config, trackedblob=parent_tb, context="assoc"):
                    child_blob = child_blobs[child_index]
                    parent_id = parent_tb.id
                    if parent_id not in splitted_parents_map and child_index not in merged_children_map:
                        parent_tb.add_data(child_blob)                        
                        assigned_children.add(child_index)
                        assigned_parents.add(parent_id)
                        index_to_id[child_index] = parent_id
            
            # New appeared blobs (no parent)
            for child_index, blob in enumerate(child_blobs):
                if child_index not in assigned_children:
                    trackedblob = TrackedBlob(new_id, all_times_seen, blob, tracking_config)
                    tracked_blobs.append(trackedblob)
                    index_to_id[child_index] = new_id
                    new_id += 1
            
            id_to_tb = {tb.id: tb for tb in tracked_blobs}
            # Disapeared blobs
            for tb_id, tb in id_to_tb.items():
                if tb_id not in assigned_parents and len(tb.centers_of_mass) < len(all_times_seen):
                    tb.add_data(Blob.empty(time_value))
                    
            # Mergers : N parents -> 1 child
            for child_index, parent_ids in merged_children_map.items():
                child_id = index_to_id[child_index]
                id_to_tb[child_id].parents.update(parent_ids)
                for p_id in parent_ids:
                    id_to_tb[p_id].children.add(child_id)
            
            # Splits : 1 parent -> N children
            for parent_id, child_indices in splitted_parents_map.items():
                for c_idx in child_indices:
                    child_id = index_to_id[c_idx]
                    id_to_tb[child_id].parents.add(parent_id)
                    id_to_tb[parent_id].children.add(child_id)
                    
        else:  # No new detected blob
            for trackedblob in tracked_blobs:
                trackedblob.add_data(Blob.empty(time_value))
                
    else:  # no previous detected blob
        for blob in child_blobs:
            trackedblob = TrackedBlob(new_id, all_times_seen, blob, tracking_config)
            tracked_blobs.append(trackedblob)
            new_id += 1
            
    return tracked_blobs

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

from skimage import measure

def find_blobs(phantom_frame, time, density_threshold, domain):
    """
    Detect all blobs in a frame based on a given density threshold.

    Parameters
    ----------
    phantom_frame : xr.DataArray
        2D array of density values for the current frame.
    time : float
        Time value corresponding to the frame.
    density_threshold : float
        Iso-density threshold for contour extraction.
    domain : Domain
        Domain object containing spatial extent and resolution attributes.

    Returns
    -------
    child_blobs : list of Blob
        List of Blob objects detected in the current frame.

    Notes
    -----
    - Contours are extracted using skimage.measure.find_contours.
    - Each contour is passed to `refining_blob()` for possible subdivision 
      into smaller blobs if convexity deficiency is high.
    """
    child_blobs = []
    contours_list = measure.find_contours(phantom_frame.values, density_threshold)
    for contour in contours_list:
        detected_single_blobs = refining_blob(contour, phantom_frame, time, density_threshold, domain) 
        child_blobs.extend(detected_single_blobs)
    return child_blobs   
    
    
def refining_blob(contour, phantom_frame, time, density_threshold, domain, original_contour_data=None):
    """
    Recursively refine a detected blob's contour into smaller blobs if its convexity 
    deficiency exceeds a threshold.

    Parameters
    ----------
    contour : ndarray
        Nx2 array of (row, column) coordinates for the blob's contour.
    phantom_frame : xr.DataArray
        Local density data for the current frame or sub-region.
    time : float
        Time value corresponding to the frame.
    density_threshold : float
        Iso-density threshold used for the current contour detection.
    domain : Domain
        Domain object containing spatial extent and resolution attributes.
    original_contour_data : dict, optional
        Data associated with the original contour (before refinement), if any.

    Returns
    -------
    detected_single_blobs : list of Blob
        List of final Blob objects resulting from the refinement.

    Notes
    -----
    - If convexity deficiency is low or threshold is close to the blob's max density, 
      the blob is accepted without further subdivision.
    """
    contour_coords = np.column_stack((contour[:, 1] * domain.dx, contour[:, 0] * domain.dy))
    
    detected_single_blobs = []
    
    blob = Blob.from_contour(time, contour_coords, domain, phantom_frame, density_threshold, original_contour_data=original_contour_data)
    
    if blob.contour_features is None or blob.local_density_array is None:
        return detected_single_blobs
    
    # n_min_pixels = max(3, (domain.Nx * domain.Ny) // 575)
    # min_area_thresh = 0.02  # n_min_pixels * domain.dx * domain.dy
    # if blob.contour_features["area"] < min_area_thresh:  # and len(blob.contour_coords) < n_min_pixels
    #     return detected_single_blobs  # too small to be considered as a blob (can be noise)

    if blob.contour_features["convexity_deficiency"] <= 0.05 or density_threshold >= 0.90*blob.max_density:
        detected_single_blobs.append(blob)
        return detected_single_blobs
    
    higher_threshold = min(density_threshold * 1.1, blob.max_density * 0.90)
    sub_contours = measure.find_contours(blob.local_density_array.values, higher_threshold)
    sub_blobs = []
    for sub_contour in sub_contours:
        sub_blobs.extend(refining_blob(sub_contour, blob.local_density_array, time, higher_threshold, domain, original_contour_data=blob.original_contour_data))
    return sub_blobs
    



















