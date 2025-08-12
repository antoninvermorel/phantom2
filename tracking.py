import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
 
from .tracked_blobs import TrackedBlob
from .blobs import Blob
from .detection_methods import get_detection_method

import numpy as np
from scipy.optimize import linear_sum_assignment

def update_tracked_blobs(tracked_blobs, phantom_frame, all_times_seen, density_threshold, tolerance_factor, domain, detection_method, distance_threshold):
    """
   Update the tracked blobs xarray.DataArray with the new frame data by associating new-detected blobs with different maethods including the hungarian algorithm 
   (centers of mass distance critera)

    Parameters
    ----------
    tracked_blobs : list
        List of tracked blobs objects (with TrackedBlob class)
    phantom_frame : xr.DataArray
        List of the frame pixels density
    all_times_seen : list of float
        list of time values for each GIF frame already seen until the current frame
    density_threshold : float
        Iso_values threshold for contours detection
    tolerance_factor : float
        Tolerance factor for blob position estimation
    domain : Domain
        object from the class domain with domain-size attributes 
    detection_method : string
        Name of the traking detection method to use
    distance_threshold : float
        Arbitrary distance threshold for the arbitrary distance threshold method
        
    Returns
    -------
    tracked_blobs : list
        Updated blob tracking list with the new data
        
    """   
    if detection_method == "arbitrary distance threshold" and (distance_threshold is None or distance_threshold <= 0):
        raise ValueError(f"You have to put a strictly positive value of distance_threshold in argument for the {detection_method} method.")
        
    # 1. Previous and new frames data extraction
    child_blobs = find_blobs(phantom_frame, density_threshold, domain)      
    parent_trackedblobs = [tb for tb in tracked_blobs if tb.is_active()]
    
    # 2. Tracking
    new_id = max((tb.id for tb in tracked_blobs), default=0) + 1

    if len(parent_trackedblobs) > 0:
        time_value = all_times_seen[-1]
        
        if len(child_blobs) > 0:
            cost_matrix = np.linalg.norm(
                np.array([tb.centers_of_mass[-1] for tb in parent_trackedblobs])[:, None, :] -
                np.array([blob.center_of_mass for blob in child_blobs])[None, :, :],
                axis=2
            )
            
            interactions_kwargs = {                
                "parent_trackedblobs": parent_trackedblobs,
                "child_blobs": child_blobs,
                "cost_matrix": cost_matrix,
                "tolerance_factor": tolerance_factor,
                "dist_threshold": distance_threshold,
            }
            method_instance = get_detection_method(detection_method)
            merged_children_map, splitted_parents_map = method_instance.get_interactions(**interactions_kwargs)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            index_to_id = {}
            assigned_children = set()
            assigned_parents = set()
            
            for parent_index, child_index in zip(row_ind, col_ind):
                parent_trackedblob = parent_trackedblobs[parent_index]
                indiv_dist_threshold = parent_trackedblob.indiv_dist_threshold(tolerance_factor)
                dist = cost_matrix[parent_index, child_index]
                if dist < indiv_dist_threshold:
                    child_blob = child_blobs[child_index]
                    parent_id = parent_trackedblob.id
                    if parent_id not in splitted_parents_map and child_index not in merged_children_map:  # -> 1-1 association
                        parent_trackedblob.add_data(time_value, child_blob)                        
                        assigned_children.add(child_index)
                        assigned_parents.add(parent_id)
                        index_to_id[child_index] = parent_id
            
            # New appeared blobs (no parent)
            for child_index, blob in enumerate(child_blobs):
                if child_index not in assigned_children:
                    trackedblob = TrackedBlob(new_id, all_times_seen, blob)
                    tracked_blobs.append(trackedblob)
                    index_to_id[child_index] = new_id
                    new_id += 1
            
            # Disapeared blobs
            for trackedblob in tracked_blobs:
                if trackedblob.id not in assigned_parents and len(trackedblob.centers_of_mass) < len(all_times_seen):
                    trackedblob.add_data(time_value, Blob.empty())
                    
            # Mergers : N parents -> 1 child
            for child_index, parent_ids in merged_children_map.items():
                child_id = index_to_id[child_index]
                child_blob = next(tb for tb in tracked_blobs if tb.id == child_id)
                child_blob.parents.update(parent_ids)
                for p_id in parent_ids:
                    parent_blob = next(tb for tb in tracked_blobs if tb.id == p_id)
                    parent_blob.children.add(child_id)
            
            # Splits : 1 parent -> N children
            for parent_id, child_indices in splitted_parents_map.items():
                parent_blob = next(tb for tb in tracked_blobs if tb.id == parent_id)
                for c_idx in child_indices:
                    child_id = index_to_id[c_idx]
                    child_blob = next(tb for tb in tracked_blobs if tb.id == child_id)
                    child_blob.parents.add(parent_id)
                    parent_blob.children.add(child_id)
                    
        else:  # No new detected blob
            for trackedblob in tracked_blobs:
                trackedblob.add_data(time_value, Blob.empty())
                
    else:  # no previous detected blob
        for blob in child_blobs:
            trackedblob = TrackedBlob(new_id, all_times_seen, blob)
            tracked_blobs.append(trackedblob)
            new_id += 1
            
    return tracked_blobs

# -------------------------------------------------------------------------------------------------------------------------------------------------
from skimage import measure

def find_blobs(phantom_frame, density_threshold, domain):
    child_blobs = []
    contours_list = measure.find_contours(phantom_frame.values, density_threshold)
    for contour in contours_list:
        detected_single_blobs = refining_blob(contour, phantom_frame, density_threshold, domain) 
        child_blobs.extend(detected_single_blobs)
    return child_blobs   
    
    
def refining_blob(contour, phantom_frame, density_threshold, domain, original_contour_data=None):
    """
    Refine one blob (given by its contour) into sub-blobs recursively if its convexity deficiency is too high.
    """
    contour_coords = np.column_stack((contour[:, 1] * domain.dx, contour[:, 0] * domain.dy))
    
    detected_single_blobs = []
    blob = Blob.from_contour(contour_coords, domain, phantom_frame, density_threshold, original_contour_data=original_contour_data)
    
    if blob.contour_data is None or blob.local_density_array is None:
        return detected_single_blobs

    if blob.contour_data["convexity_deficiency"] <= 0.05 or density_threshold >= 0.90*blob.max_density:
        detected_single_blobs.append(blob)
        return detected_single_blobs
    
    higher_threshold = min(density_threshold * 1.1, blob.max_density * 0.90)
    sub_contours = measure.find_contours(blob.local_density_array.values, higher_threshold)
    sub_blobs = []
    for sub_contour in sub_contours:
        sub_blobs.extend(refining_blob(sub_contour, blob.local_density_array, higher_threshold, domain, original_contour_data=blob.original_contour_data))
    return sub_blobs
    
































