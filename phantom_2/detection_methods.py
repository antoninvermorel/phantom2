"""In this class are implemented all the interactions detection methods used for blob tracking."""
from .matching_methods import get_matching_method

class DetectionMethod:
    def validate_config(self, tracking_config: dict):
        """
        Validate only the configuration parts relevant to detection methods.
        
        Raises
        ------
        ValueError
            If required parameters are missing or inconsistent.
        """
        # Allowed matching methods
        allowed_matching_methods = {"euclidean", "mahalanobis"}
        detection_method = tracking_config.get("interac_detection_method")
        matching_method = tracking_config.get("interac_matching_method")

        if matching_method is None:
            raise ValueError(f"{detection_method} requires 'interac_matching_method' in tracking_config.")
        if matching_method not in allowed_matching_methods:
            raise ValueError(
                f"Invalid interact_matching_method '{matching_method}' for {detection_method}. "
                f"Must be one of {allowed_matching_methods}."
            )
            
    def get_interactions(self, parent_trackedblobs, child_blobs, tracking_config):
        """
        Must be implemented by subclasses
        """
        raise NotImplementedError("This method should be overridden in subclasses")


class SimpleGating(DetectionMethod):
    def validate_config(self, tracking_config: dict):
        super().validate_config(tracking_config)
        # Compatibility checks
        if tracking_config.get("interact_matching_method") == "mahalanobis":
            raise ValueError("'simple_gating' is not compatible with 'mahalanobis' matching method.")
            
    def get_interactions(self, parent_trackedblobs, child_blobs, tracking_config):
        """
        Detect mergers and splits by applying either:
        - an arbitrary fixed euclidean distance threshold, or
        - an individual euclidean distance threshold per parent blob.

        Parameters
        ----------
        parent_trackedblobs : list
            Tracked blobs from the previous frame.
        child_blobs : list
            Blobs detected at the current frame.
        tracking_config : dict
            Full tracking configuration dictionary.

        Returns
        -------
        merged_children_map : dict[int, set[int]]
            Maps child blob indices to parent ids for mergers.
        splitted_parents_map : dict[int, set[int]]
            Maps parent ids to child indices for splits.
        """     
        self.validate_config(tracking_config)
        
        matching_method = get_matching_method(tracking_config.get("interac_matching_method"))
        matching_method.validate_config(tracking_config)
        
        cost_matrix = matching_method.make_cost_matrix(parent_trackedblobs, child_blobs)

        reverse_map = {j: set() for j in range(len(child_blobs))}
        forward_map = {tb.id: set() for tb in parent_trackedblobs}

        for parent_index, parent_tb in enumerate(parent_trackedblobs):
            for child_index in range(len(child_blobs)):
                dist = cost_matrix[parent_index, child_index] 
                if matching_method.gating(dist, tracking_config, trackedblob=parent_tb, context="interac"):
                    reverse_map[child_index].add(parent_tb.id)
                    forward_map[parent_tb.id].add(child_index)

        merged_children_map = {child_index: parent_ids for child_index, parent_ids in reverse_map.items() if len(parent_ids) > 1}
        splitted_parents_map = {parent_id: child_indices for parent_id, child_indices in forward_map.items() if len(child_indices) > 1}

        return merged_children_map, splitted_parents_map
    
    
    
class ComplexResidualGating(DetectionMethod):
    def get_interactions(self, parent_trackedblobs, child_blobs, tracking_config):
        """
        Detect mergers and splits between parent and child blobs using a two-step process:
        
        1. Hungarian algorithm for 1-1 assignments.
        2. Residual gating for unmatched blobs:
           - For splits (1 parent → N children), associate to a parent only children 
             for which this parent is the minimal-cost parent passing the gating.
           - For mergers (N parents → 1 child), associate to a child only parents for which
             this child is the minimal-cost child passing the gating.

        Parameters
        ----------
        parent_trackedblobs : list
            Tracked blobs from the previous frame.
        child_blobs : list
            Detected blobs in the current frame.
        tracking_config : dict
            Full tracking configuration dictionary.

        Returns
        -------
        merged_children_map : dict[int, set[int]]
            Maps child indices to parent IDs (mergers).
        splitted_parents_map : dict[int, set[int]]
            Maps parent IDs to child indices (splits).
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        
        self.validate_config(tracking_config)
        matching_method = get_matching_method(tracking_config.get("interac_matching_method"))
        matching_method.validate_config(tracking_config)

        # Step 1: Compute cost matrix and Hungarian 1-1 assignment
        cm_kwargs = {"context": "interac"}
        cost_matrix = matching_method.make_cost_matrix(parent_trackedblobs, child_blobs, **cm_kwargs)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Initialize maps
        child_to_parents = {c_idx: set() for c_idx in range(len(child_blobs))}
        parent_to_children = {p_tb.id: set() for p_tb in parent_trackedblobs}

        # Fill maps with Hungarian matches passing gating
        
        # # DEBUG :
        # if np.isclose(parent_trackedblobs[0].times[-1], 14.60):
        #     print(f"cost_matrix : {cost_matrix}")
        #     print("\n")
        #     print("gating 1st pass hungarian")  
        
        for parent_index, child_index in zip(row_ind, col_ind):
            parent_tb = parent_trackedblobs[parent_index]
            cost = cost_matrix[parent_index, child_index]            
            if matching_method.gating(cost, tracking_config, trackedblob=parent_tb, context="interac"):                    
                child_to_parents[child_index].add(parent_tb.id)
                parent_to_children[parent_tb.id].add(child_index)
                
        # # DEBUG :
        # if np.isclose(parent_tb.times[-1], 14.60):
        #     print("Just after complex_residual_gating 1st hungarian pass")
        #     print("child_to_parents :", child_to_parents)
        #     print("parent_to_children :", parent_to_children)

        # Keep track of already matched parents/children
        already_matched_parents = set(row_ind)
        already_matched_children = set(col_ind)

        # Step 2a: Residual gating for splits (1 parent → N children)
        
        # # DEBUG :
        # if np.isclose(parent_trackedblobs[0].times[-1], 14.60):
        #     print("gating splits detection")  
        
        for child_idx, child_blob in enumerate(child_blobs):
            if child_idx in already_matched_children:
                continue
            min_cost_parent_idx = np.argmin(cost_matrix[:, child_idx])
            min_cost = cost_matrix[min_cost_parent_idx, child_idx]
            parent_tb = parent_trackedblobs[min_cost_parent_idx]        
            if matching_method.gating(min_cost, tracking_config, trackedblob=parent_tb, context="interac"):
                parent_to_children[parent_tb.id].add(child_idx)

        # Step 2b: Residual gating for mergers (N parents → 1 child)
        
        # # DEBUG :
        # if np.isclose(parent_trackedblobs[0].times[-1], 14.60):
        #     print("gating mergers detection") 
        
        for parent_idx, parent_tb in enumerate(parent_trackedblobs):
            if parent_idx in already_matched_parents:
                continue
            min_cost_child_idx = np.argmin(cost_matrix[parent_idx, :])
            min_cost = cost_matrix[parent_idx, min_cost_child_idx]                
            if matching_method.gating(min_cost, tracking_config, trackedblob=parent_tb, context="interac"):  
                child_to_parents[min_cost_child_idx].add(parent_tb.id)
            
        # # DEBUG :
        # if np.isclose(parent_tb.times[-1], 14.60):
        #     print("After complex_residual_gating splits/mergers detection")
        #     print("child_to_parents :", child_to_parents)
        #     print("parent_to_children :", parent_to_children)
            
        # Convert to final merged/split maps
        merged_children_map = {
            child_index: parent_ids
            for child_index, parent_ids in child_to_parents.items()
            if len(parent_ids) > 1
        }
        splitted_parents_map = {
            parent_id: child_indices
            for parent_id, child_indices in parent_to_children.items()
            if len(child_indices) > 1
        }
        
        # # DEBUG :
        # if np.isclose(parent_tb.times[-1], 14.60):
        #     print("=>")
        #     print("merged_children_map :", merged_children_map)
        #     print("splitted_parents_map :", splitted_parents_map)
        
        return merged_children_map, splitted_parents_map



class Subblobs(DetectionMethod):
    def get_interactions(self, parent_trackedblobs, child_blobs, tracking_config):
        """
        Detect mergers and splits using original contour information
        when blobs may be divided into sub-blobs or merged from several
        sub-contours.
        
        The method compares the centers of mass (COMs) of original
        contours across frames, applies a Hungarian assignment, and
        uses residual gating to identify 1-to-N splits and N-to-1
        mergers.
        
        Parameters
        ----------
        parent_trackedblobs : list
            Tracked blobs from the previous frame.
        child_blobs : list
            Blobs detected in the current frame.
        tracking_config : dict
            Full tracking configuration.
        
        Returns
        -------
        merged_children_map : dict[int, set[int]]
            Maps child blob indices to parent IDs (mergers).
        splitted_parents_map : dict[int, set[int]]
            Maps parent IDs to child blob indices (splits).
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        
        self.validate_config(tracking_config)
        matching_method = get_matching_method(tracking_config.get("interac_matching_method"))
        matching_method.validate_config(tracking_config)
        
        # 1/ Mapping : original contours → subblobs and original duplications remove
        child_com_to_indices = {}
        trunc_child_blobs = []
        for c_idx, c_blob in enumerate(child_blobs):
            if c_blob.original_contour_data is None or "com" not in c_blob.original_contour_data:
                continue
            com = self.rounded_tuple(c_blob.original_contour_data["com"])
            child_com_to_indices.setdefault(com, []).append(c_idx)
            if len(child_com_to_indices[com]) == 1:
                trunc_child_blobs.append(child_blobs[c_idx])
        trunc_child_blobs = np.array(trunc_child_blobs)

        parent_com_to_ids = {}
        trunc_parent_trackedblobs = []
        for p_idx, p_trackedblob in enumerate(parent_trackedblobs):
            p_id = p_trackedblob.id
            last = p_trackedblob.original_contours_data[-1]
            if last is not None :
                com = self.rounded_tuple(last["com"])
                parent_com_to_ids.setdefault(com, []).append(p_id)
                if len(parent_com_to_ids[com]) == 1:
                    trunc_parent_trackedblobs.append(parent_trackedblobs[p_idx])
        trunc_parent_trackedblobs = np.array(trunc_parent_trackedblobs)            

        # 2/ Compute cost matrix and Hungarian 1-1 assignment for parent/child original contours
        cm_kwargs = {"context": "interac"}
        gating_kwargs = {"use_original_contour": True}
        cost_matrix = matching_method.make_cost_matrix(trunc_parent_trackedblobs, trunc_child_blobs, use_original_contour=True, **cm_kwargs)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        
        # # DEBUG :
        # if np.isclose(trunc_parent_trackedblobs[0].times[-1], 15.50):
        #     print(f"original_cost_matrix in 'subblobs' interactions detection: {cost_matrix}")
        #     print("\n")
            # print("'subblobs' 1st pass hungarian with original contours.")

        # Initialize maps
        child_com_to_parent_coms = {}
        parent_com_to_child_coms = {}

        # Fill maps with Hungarian matches passing gating
        for parent_index, child_index in zip(row_ind, col_ind):
            parent_tb = trunc_parent_trackedblobs[parent_index]
            cost = cost_matrix[parent_index, child_index]
            if matching_method.gating(cost, tracking_config, trackedblob=parent_tb, context="interac", **gating_kwargs):
                child_com = self.rounded_tuple(trunc_child_blobs[child_index].original_contour_data["com"])
                parent_com = self.rounded_tuple(parent_tb.original_contours_data[-1]["com"])
                child_com_to_parent_coms.setdefault(child_com, set()).add(parent_com)
                parent_com_to_child_coms.setdefault(parent_com, set()).add(child_com)
            
        # # DEBUG :
        # if np.isclose(parent_tb.times[-1], 15.50):
        #     print("Just after subblobs gating 1st hungarian pass")
        #     print("child_com_to_parent_coms :", child_com_to_parent_coms)
        #     print("parent_com_to_child_coms :", parent_com_to_child_coms)

        # Keep track of already matched parents/children
        already_matched_parents = set(row_ind)
        already_matched_children = set(col_ind)

        # 3a/ Residual gating for splits (1 original parent → N original children)
        
        # # DEBUG :
        # if np.isclose(trunc_child_blobs[0].time, 15.60):
        #     print(f"'subblobs' gating splits detection on children from the new frame (t={trunc_child_blobs[0].time}).")
        
        for child_idx, child_blob in enumerate(trunc_child_blobs):
            if child_idx in already_matched_children:
                continue
            min_cost_parent_idx = np.argmin(cost_matrix[:, child_idx])
            min_cost = cost_matrix[min_cost_parent_idx, child_idx]
            parent_tb = trunc_parent_trackedblobs[min_cost_parent_idx]            
            if matching_method.gating(min_cost, tracking_config, trackedblob=parent_tb, context="interac", **gating_kwargs):
                child_com = self.rounded_tuple(child_blob.original_contour_data["com"])
                parent_com = self.rounded_tuple(parent_tb.original_contours_data[-1]["com"])
                parent_com_to_child_coms.setdefault(parent_com, set()).add(child_com)

        # 3b/ Residual gating for mergers (N original parents → 1 original child)
        
        # # DEBUG :
        # if np.isclose(trunc_parent_trackedblobs[0].times[-1], 15.50):
        #     print(f"'subblobs' gating mergers detection on parents from the previous frame (t={trunc_parent_trackedblobs[0].times[-1]}).")
        
        for parent_idx, parent_tb in enumerate(trunc_parent_trackedblobs):
            if parent_idx in already_matched_parents:
                continue
            min_cost_child_idx = np.argmin(cost_matrix[parent_idx, :])
            min_cost = cost_matrix[parent_idx, min_cost_child_idx]
            if matching_method.gating(min_cost, tracking_config, trackedblob=parent_tb, context="interac", **gating_kwargs):
                child_com = self.rounded_tuple(trunc_child_blobs[min_cost_child_idx].original_contour_data["com"])
                parent_com = self.rounded_tuple(parent_tb.original_contours_data[-1]["com"])
                child_com_to_parent_coms.setdefault(child_com, set()).add(parent_com)
                
        # # DEBUG :
        # if np.isclose(trunc_parent_trackedblobs[0].times[-1], 15.50):
        #     print("After 'subblobs' splits/mergers detection")
        #     print("child_com_to_parent_coms :", child_com_to_parent_coms)
        #     print("parent_com_to_child_coms :", parent_com_to_child_coms)
                
        # 4/ Interactions detection and mapping between the 2 last frames
        merged_children_map = {}
        splitted_parents_map = {}
        
        # splits : 1 parent blob -> N child blobs
        for child_com, parent_coms in child_com_to_parent_coms.items():
            child_indices = set(child_com_to_indices.get(child_com, []))
            all_parent_ids = set()
            for parent_com in parent_coms:
                parent_ids = parent_com_to_ids.get(parent_com, [])
                all_parent_ids.update(parent_ids)
            
            if not all_parent_ids or not child_indices:
                continue
            
            parents_subcount = len(all_parent_ids)
            child_subcount = len(child_indices)
            if child_subcount == parents_subcount:
                continue
            if child_subcount > 1 and parents_subcount == 1:
                parent_id = next(iter(all_parent_ids))
                splitted_parents_map[parent_id] = child_indices
        
        # mergers : N parent blobs -> 1 child blob
        for parent_com, child_coms in parent_com_to_child_coms.items():
            parent_ids = set(parent_com_to_ids.get(parent_com, []))
            all_child_indices = set()
            for child_com in child_coms:
                child_indices = child_com_to_indices.get(child_com, [])
                all_child_indices.update(child_indices)
            if not all_child_indices or not parent_ids:
                continue

            children_subcount = len(all_child_indices)
            parent_subcount = len(parent_ids)
            if parent_subcount == children_subcount:
                continue
            if parent_subcount > 1 and children_subcount == 1:
                child_index = next(iter(all_child_indices))
                merged_children_map[child_index] = parent_ids
            
        # # DEBUG :
        # if np.isclose(trunc_parent_trackedblobs[0].times[-1], 15.50):
        #     print("=>")
        #     print("merged_children_map :", merged_children_map)
        #     print("splitted_parents_map :", splitted_parents_map)
        #     print("\n")
            
        return merged_children_map, splitted_parents_map

    @staticmethod
    def rounded_tuple(arr, decimals=4):
        import numpy as np
        return tuple(np.round(arr, decimals))



# Factory
def get_detection_method(name: str) -> DetectionMethod:
    """
    Factory function returning a detection method instance by name.

    Parameters
    ----------
    name : str
        The identifier of the detection method. Supported values are:
        - "simple_gating"
        - "complex_residual_gating"
        - "subblobs"

    Raises
    ------
    ValueError
        If the provided name does not match any supported detection method.

    Returns
    -------
    DetectionMethod
        An instance of the requested detection method class.
    """
    mapping = {
        "simple_gating": SimpleGating,
        "complex_residual_gating": ComplexResidualGating,
        "subblobs": Subblobs,
    }
    if name not in mapping:
        raise ValueError(f"Unknown detection method: {name}")
    return mapping[name]()
























