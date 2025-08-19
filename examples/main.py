"""phantom_2 blobs tracking and plots display"""
    
import os
import sys
print(sys.executable)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phantom_2 import Dataset, update_tracked_blobs, gif_plot, scatter_plot, DAG_plot

def main():
    dataset_name = "dataset_phantom_blobmodel_gif.nc"
    density_threshold = 0.2
    tolerance_factor = 2.0
    detection_method = "subblobs"  # {"arbitrary distance threshold, "personalized distance threshold", "subblobs"}
    distance_threshold = 0.925
    gif_fps = 8
        
    tracked_blobs = []
    ds_read = Dataset(dataset_name)
    domain = ds_read.domain
    all_times_seen = []

    for u in range(len(ds_read.times)):
        time = ds_read.times[u]
        all_times_seen.append(time)
        phantom_frame = ds_read.get_frame(u)
        tracked_blobs = update_tracked_blobs(
            tracked_blobs,
            phantom_frame,
            all_times_seen,
            density_threshold,
            tolerance_factor,
            domain,
            detection_method,
            distance_threshold
        )
    
    gif_plot(ds_read, tracked_blobs, gif_fps, density_threshold, tolerance_factor, detection_method, distance_threshold, save_path="phantom_blobs_tracking.gif")
    DAG_plot(tracked_blobs, density_threshold, tolerance_factor, detection_method, distance_threshold)
    scatter_plot(tracked_blobs, density_threshold, tolerance_factor, detection_method, distance_threshold, domain) 
    return tracked_blobs

main()
