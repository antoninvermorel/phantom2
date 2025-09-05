"""In this class are implemented linear Kalman filter and functions to manipulate it"""

import numpy as np
from numpy.linalg import inv

class Kalman:
    """
    Implements a linear Kalman filter for tracking blob features in 2D.
    
    The Kalman filter can model blobs with constant velocity (CV) or constant acceleration (CA),
    optionally including additional features. It provides methods for prediction, update,
    Mahalanobis distance computation, gating, and dynamic adjustment of process/measurement
    noise for truncated blobs.
    
    Attributes
    ----------
    features_names : list of str
        Names of additional blob features included in the state.
    model : str
        Motion model, either "CV" (constant velocity) or "CA" (constant acceleration).
    use_original_contour : bool
        Whether to use the original contour for observation computations.
    F : np.ndarray
        State transition matrix.
    Q : np.ndarray
        Process noise covariance matrix.
    H : np.ndarray
        Observation matrix mapping state to measurement space.
    R : np.ndarray
        Measurement noise covariance matrix.
    x : np.ndarray
        Current state vector.
    P : np.ndarray
        State covariance matrix.
    sigma_proc_pos, sigma_proc_vel, sigma_proc_acc : float
        Standard deviations for process noise in position, velocity, and acceleration.
    sigma_meas_pos : float
        Standard deviation for measurement noise in position.
    sigma_proc_feat, sigma_meas_feat : float
        Standard deviations for process and measurement noise in features.
    """
    
    
    def __init__(self, z0, kalman_features: dict, use_original_contour=False):
        """
        Initialize the Kalman filter with an initial observation and filter parameters.

        Parameters
        ----------
        z0 : array-like, shape (>=2,)
            Initial observation. z0[0] = x0, z0[1] = y0; the remaining entries are optional features.
        kalman_features : dict
            Dictionary containing at least:
                - "features": list of feature names
                - "model": "CV" or "CA"
                - "uncertainties": dictionary of standard deviations for process and measurement noise
        use_original_contour : bool, optional
            If True, use the original contour for the observation. Default is False.

        Raises
        ------
        ValueError
            If z0 has fewer than 2 elements or model is invalid.
        """
        if len(z0) < 2:
            raise ValueError("z0 has to have at least 2 elements (x0, y0)")

        self.features_names = kalman_features.get("features", set())
        self.model = kalman_features.get("model", "CV")
        self.sigma_proc_pos = kalman_features.get("uncertainties").get("sigma_proc_pos")
        self.sigma_proc_vel = kalman_features.get("uncertainties").get("sigma_proc_vel")
        self.sigma_proc_acc = kalman_features.get("uncertainties").get("sigma_proc_acc")
        self.sigma_meas_pos = kalman_features.get("uncertainties").get("sigma_meas_pos")
        self.sigma_proc_feat = kalman_features.get("uncertainties").get("sigma_proc_feat")
        self.sigma_meas_feat = kalman_features.get("uncertainties").get("sigma_meas_feat")
        self.use_original_contour = use_original_contour

        x0, y0 = float(z0[0]), float(z0[1])
        feature_values = z0[2:]  # potential features
        
        # State sizing according to chosen model
        if self.model == 'CV':
            m = len(z0) + 2  # x, y, vx, vy (+ features)
        elif self.model == 'CA':
            m = len(z0) + 4  # x, y, vx, vy, ax, ay (+ features)

        # Transition F (will be filled in later according to dt)
        self.F = np.eye(m)
        
        # Process noise Q
        self.Q = np.zeros((m,m))
        if self.model == "CV":
            self.Q[0,0] = self.sigma_proc_pos**2
            self.Q[1,1] = self.sigma_proc_pos**2
            self.Q[2,2] = self.sigma_proc_vel**2
            self.Q[3,3] = self.sigma_proc_vel**2

        elif self.model == "CA":
            self.Q[0,0] = self.sigma_proc_pos**2
            self.Q[1,1] = self.sigma_proc_pos**2
            self.Q[2,2] = self.sigma_proc_vel**2
            self.Q[3,3] = self.sigma_proc_vel**2
            self.Q[4,4] = self.sigma_proc_acc**2
            self.Q[5,5] = self.sigma_proc_acc**2
        # features
        for k in range(len(self.features_names)):
            self.Q[-len(self.features_names)+k,-len(self.features_names)+k] = self.sigma_proc_feat**2
        
        # Observation H
        H = np.zeros((len(z0), m))
        H[0,0] = 1.0
        H[1,1] = 1.0
        for k in range(len(self.features_names)):
            H[2+k, 4+k if self.model=="CV" else 6+k] = 1.0
        self.H = H

        # Measure noise R
        if len(self.features_names) > 0:
            R_features = [self.sigma_meas_feat**2] * len(self.features_names)
        else:
            R_features = []
        
        self.R = np.diag([self.sigma_meas_pos**2, self.sigma_meas_pos**2] + R_features)

        # Initial state x
        x_init = np.zeros((m,1))
        x_init[0,0] = x0
        x_init[1,0] = y0
        x_init[2,0] = 0.0  # initial vx
        if self.model == 'CA':
            x_init[4,0] = 0.0  # initial ax
            x_init[5,0] = 0.0  # initial ay
        for k, val in enumerate(feature_values):
            x_init[-len(self.features_names)+k,0] = float(val)
        self.x = x_init

        # Initiale covariance P
        if self.model=='CV':
            P0 = np.diag([self.R[0,0], self.R[1,1], 10*self.R[0,0], 10*self.R[1,1]] +
                         [max(1.0, self.R[0,0])]*len(self.features_names))
        else:
            P0 = np.diag([self.R[0,0], self.R[1,1], 10*self.R[0,0], 10*self.R[1,1],
                          100*self.R[0,0], 100*self.R[1,1]] +
                         [max(1.0, self.R[0,0])]*len(self.features_names))
        self.P = P0
        self.Q0 = self.Q.copy()
        self.R0 = self.R.copy()
        
        
    @staticmethod
    def validate_kalman_features(kalman_features):
        """
        Validate and complete Kalman filter configuration.

        Ensures that the provided dictionary has a valid model ("CV" or "CA"),
        a feature set, and a valid uncertainties dict. Missing entries are 
        filled with defaults, with warnings issued. Invalid types or keys 
        raise a ValueError.

        Parameters
        ----------
        kalman_features : dict
            Kalman filter configuration.

        Returns
        -------
        dict
            Validated and completed configuration.
        """
        import warnings
        
        default_kalman_features = {
            "features": set(),
            "model": "CV",
            "uncertainties": {
                "sigma_proc_pos": 0.7,
                "sigma_proc_vel": 4.0,
                "sigma_proc_acc": 1.5,  # only used if model="CA"
                "sigma_meas_pos": 0.2,
                "sigma_proc_feat": 0.2,
                "sigma_meas_feat": 0.2,
            }
        }
        
        allowed_models = {"CV", "CA"}
        allowed_uncs = {
            "sigma_proc_pos", "sigma_proc_vel", "sigma_proc_acc",
            "sigma_meas_pos", "sigma_proc_feat", "sigma_meas_feat"
        }
        
        if kalman_features is None or kalman_features == {}:
            warnings.warn(f"'kalman_features' provided is {kalman_features}, defaulting to {default_kalman_features}")
            kalman_features = default_kalman_features
                    
        completed_kalman_features = kalman_features.copy()
        
        features = kalman_features.get("features")
        if features is None:
            warnings.warn(f"Kalman features set is not provided, defaulting to {default_kalman_features['features']}")
            completed_kalman_features["features"] = default_kalman_features['features']
        elif not isinstance(features, set):
            raise ValueError(f"'kalman_features' requires 'features' as a set. got {type(features).__name__}.")
        
        model = kalman_features.get("model")
        if model is None:
            warnings.warn(f"Kalman model is not provided, defaulting to {default_kalman_features['model']}")
            completed_kalman_features["model"] = default_kalman_features['model']
        elif model not in allowed_models:
            raise ValueError("Provided kalman model not supplied. "
                             f"'kalman_features' requires 'model' in {allowed_models}, "
                             f"got {model}.")
        
        uncs = kalman_features.get("uncertainties")
        if uncs is None:
            warnings.warn(f"Kalman uncertainties dict is not provided, defaulting to {default_kalman_features['uncertainties']}")
            completed_kalman_features["uncertainties"] = default_kalman_features['uncertainties'].copy()
        elif not isinstance (uncs, dict):
            raise ValueError(f"'kalman_features' requires 'uncertainties' as a dict. got {type(uncs).__name__}.")
        for unc in uncs:
            if unc not in allowed_uncs:
                raise ValueError(f"'uncertainties' requires uncertainties in {allowed_uncs}, "
                                 f"got {unc}.")
        for default_unc, value in default_kalman_features['uncertainties'].items():
            if default_unc not in uncs:
                if default_unc == "sigma_proc_acc" and completed_kalman_features["model"] == "CV":
                    continue
                if ((default_unc == "sigma_proc_feat" or default_unc == "sigma_meas_feat") 
                    and completed_kalman_features["features"] == set()):
                    continue
                warnings.warn(f"{default_unc} is not provided in 'kalman_features'['uncertainties'], defaulting to {value}")
                completed_kalman_features["uncertainties"][default_unc] = value
    
        return completed_kalman_features  


    def predict(self, dt):
        """
        Predict the next state and covariance using the Kalman filter equations.

        Parameters
        ----------
        dt : float
            Time step to use for the prediction.

        Returns
        -------
        x : np.ndarray
            Predicted state vector.
        P : np.ndarray
            Predicted state covariance matrix.
        """
        self.F = np.eye(self.F.shape[0])

        # Filling in of transition matrix (dt dependant)
        if self.model == "CV":
            self.F[0,2] = dt
            self.F[1,3] = dt
            
        elif self.model == "CA":
            self.F[0,2] = dt
            self.F[1,3] = dt
            self.F[0,4] = 0.5*dt**2
            self.F[1,5] = 0.5*dt**2
            self.F[2,4] = dt
            self.F[3,5] = dt

        # Standard equations
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P


    def update(self, z):
        """
        Update the state estimate and covariance with a new measurement.

        Parameters
        ----------
        z : array-like, shape (len(H),)
            Measurement vector.

        Raises
        ------
        ValueError
            If the dimension of z does not match the observation matrix H.
        """        
        if len(z) != self.H.shape[0]:
            raise ValueError(f"z dimension ({len(z)}) incompatible with H ({self.H.shape[0]})")
    
        # Innovation
        y = z.reshape(-1,1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
    
        # Kalman gain
        K = self.P @ self.H.T @ inv(S)
    
        # State correction
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P 



    def mahalanobis2(self, z):
        """
        Compute the squared Mahalanobis distance between a measurement and the predicted state.

        Parameters
        ----------
        z : array-like
            Measurement vector.

        Returns
        -------
        float
            Squared Mahalanobis distance.

        Raises
        ------
        ValueError
            If the dimension of z does not match H.
        """
        if len(z) != self.H.shape[0]:
            raise ValueError(f"z dimension ({len(z)}) incompatible with H ({self.H.shape[0]})")
        z_pred = self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d = z - z_pred
        return float(d.T @ inv(S) @ d)


    def get_xy(self, idx_x=0, idx_y=1):
        """
        Return the current x, y position from the state vector.

        Parameters
        ----------
        idx_x : int, optional
            Index of the x-coordinate in the state vector. Default is 0.
        idx_y : int, optional
            Index of the y-coordinate in the state vector. Default is 1.

        Returns
        -------
        tuple of float
            (x, y) position.
        """
        return float(self.x[idx_x]), float(self.x[idx_y])
    

    def get_vxy(self, idx_vx=2, idx_vy=3):
        """
        Return the current velocity components from the state vector.

        Parameters
        ----------
        idx_vx : int, optional
            Index of the x-velocity in the state vector. Default is 2.
        idx_vy : int, optional
            Index of the y-velocity in the state vector. Default is 3.

        Returns
        -------
        tuple of float
            (vx, vy) velocity components.
        """
        return float(self.x[idx_vx]), float(self.x[idx_vy])
    


    def handle_conv_def(self, parent_trackedblob, child_blob,
                        pos_factor=4.0, vel_factor=3.0, acc_factor=2.0,
                        convexity_thresh=0.1, use_original_contour=False):
        """
        Adjust parent Kalman process noise Q based on convexity deficiency
        of the parent or child contour for a single parent-child pair.
    
        Parameters
        ----------
        parent_trackedblob : TrackedBlob
            The tracked blob considered in the pair with the new detected blob.
        child_blob : Blob
            The new detected blob considered in the pair with the parent tracked blob.
        pos_factor : float, optional
            Multiplicative factor for process noise on position when convexity is high.
            Default is 4.0.
        vel_factor : float, optional
            Multiplicative factor for process noise on velocity when convexity is high.
            Default is 3.0.
        acc_factor : float, optional
            Multiplicative factor for process noise on acceleration (CA model only).
            Default is 2.0.
        convexity_thresh : float, optional
            Threshold above which convexity deficiency is considered significant.
            Default is 0.1 (10%).
        use_original_contour : bool, optional
            If True, use original contour data instead of blob contour data for calculation.
    
        Returns
        -------
        None
            The function modifies the process noise matrix Q of the Kalman filter in place.
        """
        
        if not use_original_contour:
            parent_conv_def = parent_trackedblob.contours_features[-1]["convexity_deficiency"]
            child_conv_def = child_blob.contour_features["convexity_deficiency"]
        else:
            parent_conv_def = parent_trackedblob.original_contours_data[-1]["contour_features"]["convexity_deficiency"]
            child_conv_def = child_blob.original_contour_data["contour_features"]["convexity_deficiency"]
            
            
        if parent_conv_def > convexity_thresh or child_conv_def > convexity_thresh:
            # Position
            self.Q[0,0] *= pos_factor
            self.Q[1,1] *= pos_factor
            # Velocity
            self.Q[2,2] *= vel_factor
            self.Q[3,3] *= vel_factor
            # Acceleration (only CA model)
            if self.model == "CA":
                self.Q[4,4] *= acc_factor
                self.Q[5,5] *= acc_factor
                
    
    def handle_truncation(self, parent_trackedblob, child_blob,
                        r_pos_factor=3.0, q_vel_factor=2.0, q_acc_factor=2.0,
                        convexity_thresh=0.1, use_original_contour=False):
        """
        Adjust parent Kalman process noise Q based on convexity deficiency
        of the parent or child contour for a single parent-child pair.
    
        Parameters
        ----------
        parent_trackedblob : TrackedBlob
            The tracked blob considered in the pair with the new detected blob.
        child_blob : Blob
            The new detected blob considered in the pair with the parent tracked blob.
        r_pos_factor : float, optional
            Multiplicative factor for measure noise on position when convexity is high.
            Default is 3.0.
        q_vel_factor : float, optional
            Multiplicative factor for process noise on velocity when convexity is high.
            Default is 2.0.
        q_acc_factor : float, optional
            Multiplicative factor for process noise on acceleration (CA model only).
            Default is 2.0.
        use_original_contour : bool, optional
            If True, use original contour data instead of blob contour data for calculation.
    
        Returns
        -------
        None
            The function modifies the process noise matrix Q of the Kalman filter in place.
        """
        if not use_original_contour :
            parent_trunc_hist = parent_trackedblob.is_truncated
            child_trunc = child_blob.is_truncated
        else:
            parent_trunc_hist = []
            for i in range(len(parent_trackedblob.original_contours_data)):
                parent_trunc_hist.append(parent_trackedblob.original_contours_data[i]["is_truncated"]
                                  if parent_trackedblob.original_contours_data[i] else None)
            child_trunc = child_blob.original_contour_data["is_truncated"]
        
        if (len(parent_trunc_hist) >= 1 and parent_trunc_hist[-1]) or child_trunc:
            # last truncated
            self.R[0,0] *= r_pos_factor
            self.R[1,1] *= r_pos_factor
            self.Q[2,2] *= q_vel_factor
            self.Q[3,3] *= q_vel_factor
            if self.model == "CA":
                self.Q[4,4] *= q_acc_factor
                self.Q[5,5] *= q_acc_factor
    
        elif len(parent_trunc_hist) >= 2 and parent_trunc_hist[-2]:
            # second to last truncated
            self.Q[2,2] *= q_vel_factor
            self.Q[3,3] *= q_vel_factor
            if self.model == "CA":
                self.Q[4,4] *= q_acc_factor
                self.Q[5,5] *= q_acc_factor
    
        elif len(parent_trunc_hist) >= 3 and parent_trunc_hist[-3]:
            # third to last truncated
            if self.model == "CA":
                self.Q[4,4] *= q_acc_factor
                self.Q[5,5] *= q_acc_factor






















