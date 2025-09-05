"""phantom_2 blobs tracking and plots display"""
    
import os
import sys
print(sys.executable)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phantom_2 import Dataset, tracking, gif_plot, scatter_plot, DAG_plot

def main():
    ds_read = Dataset("dataset_phantom_blobmodel_noisy.nc")    
    
    tracking_config = {
        "density_threshold": 0.2,
        "assoc_matching_method": "mahalanobis",  # {"euclidean", "mahalanobis"}
        "interac_matching_method": "euclidean",  # {"euclidean", "mahalanobis"}
        "interac_detection_method": "subblobs",  # {"simple_gating", "complex_residual_gating", "subblobs"}
    
        # Main 1-1 associations subconfig
        # "assoc_euclidean_features": {
        #     "dist_thresh_type": "arbitrary",  # {"arbitrary", "individual"}
        #     "arbitrary_dist_thresh": 2.0,
        # },
        "assoc_mahalanobis_features": {
            "kalman_features": {
                "features": set(),  # {"area", "convexity_deficiency", "contour_length"} but not giving good results for the moment
                "model": "CV",
                "uncertainties": {
                    "sigma_proc_pos": 1.0,
                    "sigma_proc_vel": 4.0,
                    "sigma_proc_acc": 1.5,
                    "sigma_meas_pos": 0.2,
                }
            },
            "alpha": 0.1  # significance level for chi2 gating test : 0.05, 0.01, etc
        },
        
        # Interractions detection subconfig
        "interac_euclidean_features": {
            "dist_thresh_type": "individual",   # {"arbitrary", "individual"}
            "tolerance_factor": 2.0
        },
        # "interac_mahalanobis_features": {
        #     "kalman_features": {
        #         "features": set(),  # {"area", "convexity_deficiency", "contour_length"} but not giving good results for the moment
        #         "model": "CV",
        #         "uncertainties": {
        #             "sigma_proc_pos": 1.0,
        #             "sigma_proc_vel": 4.0,
        #             "sigma_proc_acc": 1.5,
        #             "sigma_meas_pos": 0.2,
        #         }
        #     }, 
        #     "alpha": 0.01  # significance level for chi2 gating test : 0.05, 0.01, etc
        # }
    }  
  
    gif_fps = 5
    
    tracked_blobs = tracking(ds_read, tracking_config)
    
    gif_plot(ds_read, tracked_blobs, tracking_config, gif_fps=gif_fps, save_path="phantom2_blobs_tracking.gif")
    DAG_plot(tracked_blobs, tracking_config, save_path="phantom2_DAG.jpeg")
    scatter_plot(ds_read, tracked_blobs, tracking_config, save_path="phantom2_scatter.jpeg")

main()





















