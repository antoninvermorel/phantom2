"""phantom_2 blobs tracking and plots display"""
    
import os
import sys
print(sys.executable)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phantom_2 import Dataset  # , tracking
from get_valid_intervals import get_valid_intervals

def main():
    ds_read = Dataset("dataset_phantom_blobmodel_noisy.nc")    
    
    tracking_config = {
        "density_threshold": 0.2,
        "assoc_matching_method": "euclidean",
        "interac_matching_method": "euclidean",
        "interac_detection_method": "complex_residual_gating",
            
        # 1-1 associations subconfig
        "assoc_euclidean_features": {
            "dist_thresh_type": "individual",
            "tolerance_factor": 2.6,  # giving ideal result
        },
        
        # Interractions detection subconfig
        "interac_euclidean_features": {  
            "dist_thresh_type": "arbitrary",
            "arbitrary_dist_thresh": None,
        },
    }  
    
    # If a traking config is giving the same results as those human eye could have given :
        
    # tracked_blobs = tracking(ds_read, tracking_config)
    # ideal_result = {tb.id: tb.parents for tb in tracked_blobs}  # when a config is giving the result human eye would give
    # print(ideal_result)
    
    ideal_result = {  # for a 200*200 resolution
        1: set(), 2: set(), 3: set(), 4: set(), 5: {3, 4}, 6: set(), 7: {1, 2}, 8: {5}, 9: {5}, 10: set(),
        11: set(), 12: {10, 6}, 13: {8, 7}, 14: {12}, 15: {12}, 16: {13}, 17: {13}, 18: {13}, 19: set(), 20: set(),
        21: set(), 22: {19, 21}, 23: {11, 20}, 24: {22}, 25: {22}, 26: {23}, 27: {23}
        }
    
    # valid intervals tests for a parameter in the config :
    get_valid_intervals(ds_read, tracking_config, ideal_result, "arbitrary_dist_thresh", "interac", (0.0, 5.0), 0.1)

main()





















