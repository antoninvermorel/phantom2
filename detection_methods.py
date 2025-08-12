"""In this class are implemented all the interactions detection methods used for blob tracking."""

class DetectionMethod:
    def get_interactions(self, *, parent_trackedblobs, child_blobs, cost_matrix=None, tolerance_factor=None, dist_threshold=None, **kwargs):
        """
        Abstract method to be implemented by subclasses to detect interactions between parent tracked blobs and child blobs.

        Parameters
        ----------
        parent_trackedblobs : list
            List of tracked blobs at the previous frame.
        child_blobs : list
            List of blobs detected at the current frame.
        cost_matrix : array-like, optional
            Distance matrix between parent tracked blobs and child blobs. The default is None.
        tolerance_factor : float, optional
            Multiplicative tolerance factor (>=1) for distance threshold computing. The default is None.
        dist_threshold : float, optional
            Fixed arbitrary distance threshold for detection. The default is None.
        **kwargs : dict, optional
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be overridden in subclasses.

        Returns
        -------
        None
        """
        raise NotImplementedError("This method should be overridden in subclasses")


class ArbitraryDistanceThreshold(DetectionMethod):
    def get_interactions(self, *, parent_trackedblobs, child_blobs, cost_matrix, dist_threshold, **kwargs):
        """
        Detect mergers and splits by applying a fixed arbitrary distance threshold.

        Parameters
        ----------
        parent_trackedblobs : list
            List of tracked blobs at the previous frame.
        child_blobs : list
            List of blobs detected at the current frame.
        cost_matrix : array-like
            Distance matrix between parent tracked blobs and child blobs.
        dist_threshold : float
            Distance threshold below which blobs are considered interacting.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        tuple of dict
            merged_children_map : dict
                Maps child blob indices to sets of parent tracked blob ids for mergers.
            splitted_parents_map : dict
                Maps parent tracked blob ids to sets of child blob indices for splits.
        """
        reverse_map = {j: set() for j in range(len(child_blobs))}
        forward_map = {tb.id: set() for tb in parent_trackedblobs}

        for parent_index, tracked_blob in enumerate(parent_trackedblobs):
            for child_index in range(len(child_blobs)):
                dist = cost_matrix[parent_index, child_index]
                if dist < dist_threshold:
                    reverse_map[child_index].add(tracked_blob.id)
                    forward_map[tracked_blob.id].add(child_index)

        merged_children_map = {child_index: parent_ids for child_index, parent_ids in reverse_map.items() if len(parent_ids) > 1}
        splitted_parents_map = {parent_id: child_indices for parent_id, child_indices in forward_map.items() if len(child_indices) > 1}
        return merged_children_map, splitted_parents_map


class IndividualDistanceThreshold(DetectionMethod):
    def get_interactions(self, *, parent_trackedblobs, child_blobs, cost_matrix, tolerance_factor, **kwargs):
        """
        Detect mergers and splits by applying individual distance thresholds per tracked blob.

        Parameters
        ----------
        parent_trackedblobs : list
            List of tracked blobs at the previous frame.
        child_blobs : list
            List of blobs detected at the current frame.
        cost_matrix : array-like
            Distance matrix between parent tracked blobs and child blobs.
        tolerance_factor : float
            Multiplicative tolerance factor (>=1) for distance threshold computing.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        tuple of dict
            merged_children_map : dict
                Maps child blob indices to sets of parent tracked blob ids for mergers.
            splitted_parents_map : dict
                Maps parent tracked blob ids to sets of child blob indices for splits.
        """
        reverse_map = {j: set() for j in range(len(child_blobs))}
        forward_map = {tb.id: set() for tb in parent_trackedblobs}

        for parent_index, tracked_blob in enumerate(parent_trackedblobs):
            indiv_dist_thresh = tracked_blob.indiv_dist_threshold(tolerance_factor)
            for child_index in range(len(child_blobs)):
                dist = cost_matrix[parent_index, child_index]
                if dist < indiv_dist_thresh:
                    reverse_map[child_index].add(tracked_blob.id)
                    forward_map[tracked_blob.id].add(child_index)

        merged_children_map = {child_index: parent_ids for child_index, parent_ids in reverse_map.items() if len(parent_ids) > 1}
        splitted_parents_map = {parent_id: child_indices for parent_id, child_indices in forward_map.items() if len(child_indices) > 1}
        return merged_children_map, splitted_parents_map


class SubblobsDetection(DetectionMethod):
    def get_interactions(self, *, parent_trackedblobs, child_blobs, tolerance_factor, **kwargs):
        """
        Detect interactions by matching original contours centers of mass and identifying mergers/splits.

        Parameters
        ----------
        parent_trackedblobs : list
            List of tracked blobs at the previous frame.
        child_blobs : list
            List of blobs detected at the current frame.
        tolerance_factor : float
            Multiplicative tolerance factor (>=1) for distance threshold computing.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        tuple of dict
            merged_children_map : dict
                Maps child blob indices to sets of parent tracked blob ids for mergers.
            splitted_parents_map : dict
                Maps parent tracked blob ids to sets of child blob indices for splits.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        # 1/ Matching previous/new original contours
        previous_ori_com = np.array([tb.original_contours_data[-1]["com"] for tb in parent_trackedblobs if tb.original_contours_data[-1] is not None])
        new_ori_com = np.array([blob.original_contour_data["com"] for blob in child_blobs if blob.original_contour_data is not None])
                
        cost_matrix = np.linalg.norm(previous_ori_com[:, None, :] - new_ori_com[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        original_matchs = []
        
        for original_parent_index, original_child_index in zip(row_ind, col_ind):
            trackedblob = parent_trackedblobs[original_parent_index]
            dist = cost_matrix[original_parent_index, original_child_index]
            if self.within_thresh(trackedblob, dist, tolerance_factor):
                parent_com = previous_ori_com[original_parent_index]
                child_com = new_ori_com[original_child_index]
                original_matchs.append((self.rounded_tuple(parent_com), self.rounded_tuple(child_com)))
           
        # 2/ Mapping : handle unmatched parents/children original contours to detect N-N assignations
        already_matched_parents = set(row_ind)
        already_matched_children = set(col_ind)
        
        child_to_parents = {}
        parent_to_children = {}

        # original contour split (1 parent → N enfants)
        for ori_cidx in range(len(new_ori_com)):
            if ori_cidx in already_matched_children:
                continue
            parent_dist_min_index = np.argmin(cost_matrix[:, ori_cidx])
            trackedblob = parent_trackedblobs[parent_dist_min_index]
            min_dist = cost_matrix[parent_dist_min_index, ori_cidx]
            if self.within_thresh(trackedblob, min_dist, tolerance_factor):
                parent_com = previous_ori_com[parent_dist_min_index]
                child_com = new_ori_com[ori_cidx]
                parent_to_children.setdefault(self.rounded_tuple(parent_com), set()).add(self.rounded_tuple(child_com))

        for ori_pidx in range(len(previous_ori_com)):
            if ori_pidx in already_matched_parents:
                continue
            child_dist_min_index = np.argmin(cost_matrix[ori_pidx, :])
            trackedblob = parent_trackedblobs[ori_pidx]
            min_dist = cost_matrix[ori_pidx, child_dist_min_index]
            if self.within_thresh(trackedblob, min_dist, tolerance_factor):
                child_com = new_ori_com[child_dist_min_index]
                parent_com = previous_ori_com[ori_pidx]
                child_to_parents.setdefault(self.rounded_tuple(child_com), set()).add(self.rounded_tuple(parent_com))
                
        # adding of the hungarian matched original contours in the maps
        for (parent_com, child_com) in original_matchs:
            child_to_parents.setdefault(child_com, set()).add(parent_com)
            parent_to_children.setdefault(parent_com, set()).add(child_com)
        
        # 3/ Mapping : original contours -> subblobs
        child_com_to_indices = {}
        for child_index, blob in enumerate(child_blobs):
            if blob.original_contour_data is None or "com" not in blob.original_contour_data:
                continue
            com = self.rounded_tuple(blob.original_contour_data["com"])
            child_com_to_indices.setdefault(com, []).append(child_index)

        parent_com_to_ids = {}
        for trackedblob in parent_trackedblobs:
            parent_id = trackedblob.id
            last = trackedblob.original_contours_data[-1]
            if last is not None :
                com = self.rounded_tuple(last["com"])
                parent_com_to_ids.setdefault(com, []).append(parent_id)
                
        # 4/ Interactions detection and mapping between the 2 last frames
        merged_children_map = {}
        splitted_parents_map = {}
        
        # splits : 1 parent blob -> N child blobs
        for child_com, parent_com_set in child_to_parents.items():
            child_indices = set(child_com_to_indices.get(child_com, []))
            all_parent_ids = set()
            for parent_com in parent_com_set:
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
        for parent_com, child_com_set in parent_to_children.items():
            parent_ids = set(parent_com_to_ids.get(parent_com, []))
            all_child_indices = set()
            for child_com in child_com_set:
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
            
        return merged_children_map, splitted_parents_map

    @staticmethod
    def rounded_tuple(arr, decimals=4):
        import numpy as np
        return tuple(np.round(arr, decimals))
    
    @staticmethod
    def within_thresh(trackedblob, dist, tolerance_factor, use_ori=True):
        return dist < trackedblob.indiv_dist_threshold(tolerance_factor, use_original_contour=use_ori)



def get_detection_method(name: str) -> DetectionMethod:
    mapping = {
        "arbitrary distance threshold": ArbitraryDistanceThreshold,
        "personalized distance threshold": IndividualDistanceThreshold,
        "subblobs": SubblobsDetection,
    }
    if name not in mapping:
        raise ValueError(f"Unknown detection method: {name}")
    return mapping[name]()

























