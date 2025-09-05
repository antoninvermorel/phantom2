"""This module contain the MatchingMethod class, in which are implemented different methods to create the cost matrix
the Hungarian algorithm uses, and to do gating test to validate the matchs by the hungarian algorithm"""

import numpy as np

class MatchingMethod:
    def validate_config(self, tracking_config: dict):
        """
        Validate only the configuration parts relevant to matching methods.

        Raises
        ------
        ValueError
            If required parameters are missing or inconsistent.
        """
        # subclasses will enforce detailed rules
    
    def make_cost_matrix(self, parent_trackedblobs, child_blobs, *, use_original_contour=False, **kwargs):
        """
        Compute the cost matrix. The signature is generic.
        
        Parameters
        ----------
        parent_trackedblobs : list
        child_blobs : list
        kwargs : dict
            - 'context' (str, optional) for Mahalanobis
        """
        raise NotImplementedError

    def gating(self, cost, tracking_config, *, trackedblob=None, context=None, **kwargs):
        """
        Return True/False if cost passes gating.
        
        kwargs : dict
            - 'use_original_contour' (Boolean, optional) for Euclidean
        """
        raise NotImplementedError


class EuclideanMethod(MatchingMethod):
    """
    Matching method using euclidean distance as cost for hungarian algorithm
    """
    def validate_config(self, tracking_config: dict):
        super().validate_config(tracking_config)
        # Euclidean distance threshold features 
        if tracking_config.get("assoc_matching_method") == "euclidean":
            assoc_features = tracking_config.get("assoc_euclidean_features")
            if assoc_features is None:
                raise ValueError("Euclidean associations matching requires 'assoc_euclidean_features' in tracking_config.")
            dist_thresh_type = assoc_features.get("dist_thresh_type")
            allowed_dist_thresh_types = {"arbitrary", "individual"}
            if dist_thresh_type is None:
                raise ValueError("Euclidean associations matching requires 'dist_thresh_type' in 'assoc_euclidean_features'.")
            if dist_thresh_type not in allowed_dist_thresh_types:
                raise ValueError(
                    f"'assoc_euclidean_features' requires 'dist_thresh_type' in {allowed_dist_thresh_types}, "
                    f"got {dist_thresh_type}."
                )
            if dist_thresh_type == "arbitrary":
                if assoc_features.get("arbitrary_dist_thresh") is None:
                    raise ValueError("'arbitrary' distance threshold type requires 'arbitrary_dist_thresh' in 'assoc_euclidean_features'.")
            elif dist_thresh_type == "individual":
                if assoc_features.get("tolerance_factor") is None:
                    raise ValueError("individual' distance threshold type requires a positive 'tolerance_factor' in 'assoc_euclidean_features'.")
        
        if tracking_config.get("interac_matching_method") == "euclidean":
            interac_features = tracking_config.get("interac_euclidean_features")
            if interac_features is None:
                raise ValueError("Euclidean interactions matching requires 'interac_euclidean_features' in tracking_config.")
            dist_thresh_type = interac_features.get("dist_thresh_type")
            allowed_dist_thresh_types = {"arbitrary", "individual"}
            if dist_thresh_type is None:
                raise ValueError("Euclidean interactions matching requires 'dist_thresh_type' in 'interac_euclidean_features'.")
            if dist_thresh_type not in allowed_dist_thresh_types:
                raise ValueError(
                    f"'interac_euclidean_features' requires 'dist_thresh_type' in {allowed_dist_thresh_types}, "
                    f"got {dist_thresh_type}."
                )
            if dist_thresh_type == "arbitrary":
                if interac_features.get("arbitrary_dist_thresh") is None:
                    raise ValueError("'arbitrary' distance threshold type requires 'arbitrary_dist_thresh' in 'interac_euclidean_features'.")
            elif dist_thresh_type == "individual":
                if interac_features.get("tolerance_factor") is None:
                    raise ValueError("individual' distance threshold type requires a positive 'tolerance_factor' in 'interac_euclidean_features'.")
                    
                
    def make_cost_matrix(self, parent_trackedblobs, child_blobs, *, use_original_contour=False, **kwargs):
        """
        Compute the Euclidean distance cost matrix between parent tracked blobs 
        and newly detected child blobs.

        Parameters
        ----------
        parent_trackedblobs : list
            List of tracked blob objects from the previous frame.
        child_blobs : list
            List of blob objects detected in the current frame.
        use_original_contour : bool, optional
            If True, use original contour COMs instead of filtered COMs 

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_parents, n_children) where each entry 
            is the Euclidean distance between a parent and a child blob.
        """
        if use_original_contour:
            prev_com = np.array([tb.original_contours_data[-1]["com"] for tb in parent_trackedblobs if tb.original_contours_data[-1] is not None])
            new_com = np.array([blob.original_contour_data["com"] for blob in child_blobs if blob.original_contour_data is not None])
        else:
            prev_com = np.array([tb.centers_of_mass[-1] for tb in parent_trackedblobs])
            new_com = np.array([blob.center_of_mass for blob in child_blobs])

        return np.linalg.norm(prev_com[:, None, :] - new_com[None, :, :], axis=2)
    

    def gating(
            self, cost, tracking_config, *, trackedblob=None, context=None,
            use_original_contour=False, **kwargs
        ):
            """
            Apply Euclidean gating to determine if a given cost (distance) 
            is an admissible match.
        
            Parameters
            ----------
            cost : float
                The Euclidean distance between a parent and a child blob.
            tracking_config : dict
                Tracking configuration containing threshold parameters.
            trackedblob : object, optional
                A tracked blob, required if 'dist_thresh_type' is 'individual'.
            use_original_contour : bool, optional
                If True, uses the blob's original contour COM for 
                threshold computation (default is False).
            context : {"assoc", "interac"}
                Which context is considered (association or interaction).
        
            Raises
            ------
            ValueError
                If required parameters are missing or inconsistent.
        
            Returns
            -------
            bool
                True if the cost is below the computed threshold, 
                False otherwise.
            """
            allowed_contexts = {"assoc", "interac"}
            if context not in allowed_contexts:
                raise ValueError(f"Euclidean gating requires 'context' in {allowed_contexts}, got {context}.")
        
            # Pick the correct config block depending on context
            feature_key = f"{context}_euclidean_features"
            if feature_key not in tracking_config:
                raise ValueError(f"Missing '{feature_key}' in tracking_config.")
        
            cfg = tracking_config[feature_key]
            dist_thresh_type = cfg["dist_thresh_type"]
        
            if dist_thresh_type == "individual":
                if trackedblob is None:
                    raise ValueError("Euclidean gating with 'individual' threshold requires 'trackedblob'.")
                tolerance_factor = cfg["tolerance_factor"]
                dist_thresh = trackedblob.indiv_dist_threshold(
                    tolerance_factor, use_original_contour=use_original_contour
                )
                
                # # DEBUG tests :
                # if np.isclose(trackedblob.times[-1], 8.30) and (trackedblob.id == 16 or trackedblob.id == 18):
                #     print(f"gating test for parent_trackedblob {trackedblob.id}")
                #     print(f"[Euclidean gating] cost={cost:.3f}, threshold={dist_thresh:.3f}, pass={cost < dist_thresh}")
                #     print("\n")
                
            elif dist_thresh_type == "arbitrary":
                dist_thresh = cfg["arbitrary_dist_thresh"]
        
            return cost < dist_thresh



class MahalanobisMethod(MatchingMethod):
    """
    Matching method using squared Mahalanobis distance as cost for hungarian algorithm
    """
    def validate_config(self, tracking_config: dict):
        super().validate_config(tracking_config)
        # Mahalanobis gating checks
        if tracking_config.get("assoc_matching_method") == "mahalanobis":
            assoc_features = tracking_config.get("assoc_mahalanobis_features")
            if assoc_features is None:
                raise ValueError("Mahalanobis associations matching requires 'assoc_mahalanobis_features' in config.")
            elif assoc_features.get("alpha") is None:
                raise ValueError("Mahalanobis associations matching requires 'alpha' in 'assoc_mahalanobis_features'.")
            
        if tracking_config.get("interac_matching_method") == "mahalanobis":
            interac_features = tracking_config.get("interac_mahalanobis_features")
            if interac_features is None:
                raise ValueError("Mahalanobis interactions matching requires 'interac_mahalanobis_features' in config.")
            elif interac_features.get("alpha") is None:
                raise ValueError("Mahalanobis interactions matching requires 'alpha' in 'interac_mahalanobis_features'.")
            

    def make_cost_matrix(self, parent_trackedblobs, child_blobs, *, use_original_contour=False, context=None, **kwargs):
        """
        Compute the Mahalanobis distance cost matrix using a specified Kalman filter
        of the parent tracked blobs. Assumes that Kalman predictions have already been performed.
    
        Parameters
        ----------
        parent_trackedblobs : list of TrackedBlob
            List of tracked blob objects from the previous frame.
        child_blobs : list of Blob
            List of blob objects detected in the current frame.
        context : {"assoc", "interac"}
            Which context is considered (association or interaction).
        use_original_contour : bool, optional
            If True, use original contour COMs for observation (default is False).
    
        Returns
        -------
        np.ndarray
            A 2D array of shape (n_parents, n_children) containing squared 
            Mahalanobis distances between predicted states and observations.
    
        Raises
        ------
        ValueError
            If `kalman_type` is None or not in {"assoc", "interac"}, or
            if a parent TrackedBlob does not have the requested Kalman filter,
            or if a child Blob cannot provide the required observation vector.
        """
        allowed_contexts = {"assoc", "interac"}
        if context not in allowed_contexts:
            raise ValueError(f"Mahalanobis gating requires 'context' in {allowed_contexts}, got {context}.")

        cost_matrix = np.zeros((len(parent_trackedblobs), len(child_blobs)))
    
        for i, tb in enumerate(parent_trackedblobs):
            kalman = tb.kalman_assoc if context == "assoc" else tb.kalman_interac
            if kalman is None:
                raise ValueError(f"The kalman_'{context}' of TrackedBlob {tb.id} is None.")
            # Save kalman state
            state_backup = kalman.x.copy()
            P_backup = kalman.P.copy()
            
            for j, blob in enumerate(child_blobs):
                # Restore kalman state before each repetition
                kalman.x = state_backup.copy()
                kalman.P = P_backup.copy()
                # kalman filter individual prediction
                self.predict_kalman(tb, blob, kalman, use_original_contour=use_original_contour)
                # observation
                z = blob.get_observation(kalman.features_names, use_original_contour=use_original_contour)
                if z is None:
                    raise ValueError(f"Child blob at index {j} cannot provide observation vector for Kalman filter.")
    
                d2 = kalman.mahalanobis2(z)
                cost_matrix[i, j] = d2
    
        return cost_matrix
    
    @staticmethod
    def predict_kalman(parent_tb, child_blob, kalman, use_original_contour=False):
        """
        Predict the next state of a tracked blob's Kalman filter.

        Parameters
        ----------
        parent_tb : TrackedBlob
            The tracked blob considered in the pair with the new detected blob.
        child_blob : Blob
            The new detected blob considered in the pair with the parent tracked blob.
        kalman : Kalman
            The tracked blob kalman filter to predict (kalman_interac or kalman_assoc).
        use_original_contour : bool, optional
            If True, use original contour data instead of blob contour data for calculation. The default is False.

        Returns
        -------
        None.

        """
        # Reset noises matrices
        kalman.Q = kalman.Q0.copy()
        kalman.R = kalman.R0.copy()
        
        # Handle eventual issues
        kalman.handle_truncation(parent_tb, child_blob, use_original_contour=use_original_contour)
        kalman.handle_conv_def(parent_tb, child_blob, use_original_contour=use_original_contour)
        
        # Prediction
        dt = child_blob.time - parent_tb.times[-1]
        kalman.predict(dt)
            
        
    def gating(self, cost, tracking_config, *, trackedblob=None, context=None, **kwargs):
        """
        Apply Mahalanobis gating to decide if a cost is admissible.
    
        Parameters
        ----------
        cost : float
            Squared Mahalanobis distance between prediction and observation.
        tracking_config : dict
            Must contain the relevant '{kalman_type}_mahalanobis_features' entry.
        trackedblob : TrackedBlob, optional
            The tracked object holding Kalman filters.
        context : {"assoc", "interac"}
            Which context is considered (association or interaction).
    
        Returns
        -------
        bool
            True if the cost is below the chi-square threshold, 
            False otherwise.
    
        Raises
        ------
        ValueError
            If required parameters are missing or inconsistent.
        """
        from scipy.stats import chi2
        d2 = cost
    
        allowed_contexts = {"assoc", "interac"}
        if context not in allowed_contexts:
            raise ValueError(f"Mahalanobis gating requires 'context' in {allowed_contexts}, got {context}.")
    
        if trackedblob is None:
            raise ValueError("Mahalanobis gating requires a valid 'trackedblob'.")
    
        # --- Retrieve the Kalman filter
        kalman = getattr(trackedblob, f"kalman_{context}", None)
        if kalman is None:
            raise ValueError(f"TrackedBlob has no active Kalman filter named 'kalman_{context}'.")
    
        df = kalman.H.shape[0]
    
        # --- Retrieve alpha from config
        feature_key = f"{context}_mahalanobis_features"
        cfg = tracking_config.get(feature_key)
        if cfg is None or "alpha" not in cfg:
            raise ValueError(f"Missing 'alpha' in tracking_config['{feature_key}'].")
    
        alpha = cfg["alpha"]
        
        # # DEBUG tests :
        # if np.isclose(trackedblob.times[-1], 12.60) and (trackedblob.id == 11 or trackedblob.id == 17 or trackedblob.id == 19):
        #     print(f"gating test for parent_trackedblob {trackedblob.id}")
        #     print(f"[Mahalanobis gating] cost={cost:.3f}, chi2={chi2.ppf(alpha, df):.3f}, pass={cost < chi2.ppf(alpha, df)}")
        #     print("\n")
    
        return d2 <= chi2.ppf(alpha, df)



# Factory
def get_matching_method(name: str) -> MatchingMethod:
    """
    Factory function to instantiate a matching method by name.

    Parameters
    ----------
    name : str
        Name of the matching method. Accepted values are:
        - 'euclidean'
        - 'mahalanobis'

    Returns
    -------
    MatchingMethod
        An instance of the requested matching method.

    Raises
    ------
    ValueError
        If the provided name does not correspond to a known matching method.
    """
    name = name.lower()
    if name == "euclidean":
        return EuclideanMethod()
    elif name == "mahalanobis":
        return MahalanobisMethod()
    else:
        raise ValueError(f"Unknown matching method '{name}'")




















