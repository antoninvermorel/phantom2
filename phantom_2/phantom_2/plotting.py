import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import imageio_ffmpeg as iio_ffmpeg

# to use ffmpeg
mpl.rcParams['animation.ffmpeg_path'] = iio_ffmpeg.get_ffmpeg_exe()


def gif_plot(ds_read, tracked_blobs, tracking_config, gif_fps=8, save_path="phantom2_tracking.gif"):
    """
    Generate and save an animated GIF visualizing blob tracking over time,
    using ffmpeg for better quality.

    Parameters
    ----------
    ds_read : Dataset
        Dataset object containing simulation or experimental data, including domain size and time series frames.
    tracked_blobs : list of TrackedBlob
        List of tracked blobs with their contour coordinates and metadata.
    tracking_config : dict
        Tracking configuration dictionary (used for personalized title).
    gif_fps : int, optional
        Frames per second for the output GIF animation. Default is 8.
    save_path : str, optional
        Path to save the generated GIF. Default is "phantom2_tracking.gif".

    Returns
    -------
    None
        Displays the animation and saves it as a GIF file.
    """
    assert len(tracked_blobs) > 0
    assert gif_fps > 0

    times = ds_read.times
    domain = ds_read.domain
    Lx, Ly = domain.Lx, domain.Ly

    base_height = 6
    aspect_ratio = Lx / Ly
    fig_width = base_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, base_height))

    # background frame (fixed)
    phantom_frame = ds_read.get_frame(0)
    im = ax.imshow(phantom_frame, origin='lower', extent=[0, Lx, 0, Ly], animated=True)

    # colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    cbar = fig.colorbar(im, cax=cax, label='n', orientation='vertical')
    cbar.ax.set_ylabel('n', rotation=0, va='center', ha='left')

    # prepare lines/texts
    max_blobs = len(tracked_blobs)
    contour_lines_blue, contour_lines_red, text_labels = [], [], []
    for _ in range(max_blobs):
        line_blue, = ax.plot([], [], 'b-', linewidth=1.5)
        line_red, = ax.plot([], [], 'r-', linewidth=1.5)
        contour_lines_blue.append(line_blue)
        contour_lines_red.append(line_red)
        txt = ax.text(0, 0, "", color="red", ha='center', va='center', fontsize=8)
        text_labels.append(txt)

    time_text = ax.text(1.035, 1.0225, "", transform=ax.transAxes, ha='left', va='bottom', fontsize=12)

    # Titles
    ax.set_title("Phantom2 tracking gif", fontweight='bold', pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation='horizontal')
    personalized_text = personalized_config(tracking_config)
    # position under the x-axe
    ax.text(0.5, -0.12, personalized_text, transform=ax.transAxes, ha='center', va='top', fontsize=10)

    # update function
    def update(frame_index):
        time_value = times[frame_index]
        phantom_frame = ds_read.get_frame(frame_index)
        im.set_data(phantom_frame.values)
        cbar.update_normal(im)

        for line_blue, line_red, txt in zip(contour_lines_blue, contour_lines_red, text_labels):
            line_blue.set_data([], [])
            line_red.set_data([], [])
            txt.set_text("")

        count = 0
        for tb in tracked_blobs:
            if time_value not in tb.times:
                continue
            if count >= max_blobs:
                break

            com = tb.centers_of_mass[tb.times.index(time_value)]
            coords = tb.contours_coords[tb.times.index(time_value)]
            original_data = tb.original_contours_data[tb.times.index(time_value)]
            original_coords = original_data["coords"] if original_data is not None else None

            if com is None or coords is None:
                continue

            contour_lines_blue[count].set_data(coords[:, 0], coords[:, 1])
            contour_lines_red[count].set_data(original_coords[:, 0], original_coords[:, 1] if original_coords is not None else [])
            text_labels[count].set_position((com[0], com[1]))
            text_labels[count].set_text(str(tb.id))
            count += 1

        time_text.set_text(f"t={time_value:.2f}")
        return [im, time_text] + contour_lines_blue + contour_lines_red + text_labels

    ani = FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)
    # automatic adjustment of borders
    fig.tight_layout()
    # save using ffmpeg for high quality
    ani.save(save_path, writer="ffmpeg", fps=gif_fps, dpi=100)
    plt.show()
    plt.close(fig)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
import math

def scatter_plot(ds_read, tracked_blobs, tracking_config, save_path=None):
    """
    Plot the spatial trajectories of all tracked blobs as a scatter plot.

    Parameters
    ----------
    ds_read : Dataset
        Dataset object containing simulation or experimental data, including domain size and time series frames.
    tracked_blobs : list of TrackedBlob
        List of tracked blobs containing their centers of mass over time.
    tracking_config : dict
        Dictionary of tracking parameters.
    save_path : str, optional
        File path to save the DAG plot as an image. If None, only displays the plot.

    Returns
    -------
    None
        Displays the scatter plot.
    """
    assert len(tracked_blobs) > 0
    n_blobs = len(tracked_blobs)

    custom_colors = [
        'red', 'blue', 'green', 'gold', 'darkviolet', 'orange', 'deeppink', 'sienna', 'gray', 'lightgreen',
        'darkcyan', 'darkslateblue', 'magenta', 'deepskyblue', 'salmon', 'lightsteelblue', 'limegreen', 'crimson',
        'firebrick', 'khaki', 'darkturquoise', 'mediumaquamarine', 'palevioletred', 'rosybrown', 'yellowgreen',
        'gainsboro', 'wheat', 'lightcoral', 'tan', 'indigo', 'teal', 'slateblue', 'burlywood'
    ]
    if n_blobs > len(custom_colors):
        raise ValueError(f"Not enough colors for {n_blobs} blobs ! Add some in custom_colors.")
    
    domain = ds_read.domain
    Lx = domain.Lx
    Ly = domain.Ly
    base_height = 7
    aspect_ratio = Lx / Ly
    fig_width = base_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, base_height))

    for i, trackedblob in enumerate(tracked_blobs):
        com_list = trackedblob.centers_of_mass
        if com_list is not None:
            x_com_list = [pt[0] for pt in com_list if pt is not None]
            y_com_list = [pt[1] for pt in com_list if pt is not None]
            color = custom_colors[i]
            ax.scatter(x_com_list, y_com_list, marker='.', color=color, label=f"{trackedblob.id}")
    
    ncol = math.ceil(n_blobs/25)  

    ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left', ncol=ncol, markerscale=1.5, handletextpad=0.2, columnspacing=0.7, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation='horizontal')
    ax.set_title("Phantom2 tracked blobs scatter plot", fontweight='bold')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal', adjustable='box')
    personalized_text = personalized_config(tracking_config)
    ax.text(0.5, -0.12, personalized_text, transform=ax.transAxes, ha='center', va='top', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------  
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def DAG_plot(tracked_blobs, tracking_config, save_path=None, ):
    """
    Plot the Directed Acyclic Graph (DAG) of blob lineage relationships.

    Each node represents a tracked blob, and edges represent parent–child relationships
    (splits and merges between blobs across time steps).

    Parameters
    ----------
    tracked_blobs : list of TrackedBlob
        List of tracked blobs containing their IDs and parent relationships.
    tracking_config : dict
        Dictionary of tracking parameters.
    save_path : str, optional
        File path to save the DAG plot as an image. If None, only displays the plot.

    Returns
    -------
    None
        Displays the DAG plot and optionally saves it as an image.
    """
    assert len(tracked_blobs) > 0
    G = nx.DiGraph()
    # Nodes and edges adding
    for trackedblob in tracked_blobs:
        G.add_node(trackedblob.id)
        for parent_id in trackedblob.parents:
            G.add_edge(parent_id, trackedblob.id)

    pos = graphviz_layout(G, prog="dot")  # 'dot' = tree layout

    components = list(nx.weakly_connected_components(G))
    custom_colors = [
        'red', 'darkcyan', 'green', 'gold', 'darkviolet', 'orange', 'deeppink', 'rosybrown', 'gray', 'lightgreen',
        'darkslateblue', 'magenta', 'deepskyblue', 'salmon', 'lightsteelblue', 'limegreen', 'crimson', 'firebrick',
        'khaki', 'darkturquoise', 'mediumaquamarine', 'palevioletred', 'sienna', 'yellowgreen', 'gainsboro',
        'wheat', 'lightcoral', 'tan', 'indigo', 'blue', 'teal'
    ]
    
    if len(components) > len(custom_colors):
        raise ValueError(f"Not enough colors for {len(components)} blob families ! Add some in custom_colors.")
        
        
    color_map = {}
    for i, comp in enumerate(components):
        for node in comp:
            color_map[node] = custom_colors[i]

    node_colors = [color_map[node] for node in G.nodes]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        edge_color='gray',
        arrows=True,
        font_weight='bold',
        ax=ax
    )
    ax.set_title("Phantom_2.0 tracked blobs DAG plot", fontweight='bold')
    personalized_text = personalized_config(tracking_config)
    ax.text(0.5, -0.1, personalized_text, transform=ax.transAxes, ha='center', va='top', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
     
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  
def personalized_config(tracking_config: dict) -> str:
    """
    Create a compact descriptive plot title from the tracking configuration.

    Parameters
    ----------
    tracking_config : dict
        Dictionary of tracking parameters used in tracking.

    Returns
    -------
    str
        A formatted string suitable for plot titles, e.g.:
        density_thresh=0.2 ; interac_detection_method=complex_residual_gating
        matching_methods: assoc=euclidean (arbitrary_dist_thresh=3.0)
                          interac=mahalanobis (alpha=0.05)
    """
    density_threshold = tracking_config["density_threshold"]
    interac_detection_method = tracking_config["interac_detection_method"]

    # Helper to format a matching method with parameters
    def format_matching(context: str) -> str:
        method_name = tracking_config[f"{context}_matching_method"]
        subconfig_euc = tracking_config.get(f"{context}_euclidean_features", {})
        subconfig_mah = tracking_config.get(f"{context}_mahalanobis_features", {})

        if method_name == "euclidean":
            dist_type = subconfig_euc.get("dist_thresh_type", "arbitrary")
            if dist_type == "arbitrary":
                thresh = subconfig_euc.get("arbitrary_dist_thresh", "None")
                return f"{method_name} (arbitrary_dist_thresh={thresh})"
            elif dist_type == "individual":
                tol = subconfig_euc.get("tolerance_factor", "None")
                return f"{method_name} (tol_factor={tol})"
            else:
                return f"{method_name}"
        elif method_name == "mahalanobis":
            alpha = subconfig_mah.get("alpha", "None")
            return f"{method_name} (alpha={alpha})"
        else:
            raise ValueError(f"Unknown matching method: {method_name}")

    assoc_method_str = format_matching("assoc")
    interac_method_str = format_matching("interac")

    title = (
        f"density_thresh={density_threshold} ; interac_detection_method: {interac_detection_method}\n"
        f"assoc_matching: {assoc_method_str}\n"
        f"interac_matching: {interac_method_str}"
    )
    return title



    




















