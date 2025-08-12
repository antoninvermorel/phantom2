import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.animation import FuncAnimation

def gif_plot(ds_read, tracked_blobs, gif_fps, density_threshold, tolerance_factor, detection_method, distance_threshold, save_path="mygif.gif"):
    
    assert len(tracked_blobs) > 0
    assert gif_fps > 0
    
    times = ds_read.times
    domain = ds_read.domain
    Lx = domain.Lx
    Ly = domain.Ly
    
    base_height = 6
    aspect_ratio = Lx / Ly
    fig_width = base_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, base_height))
    
    # background frame (fixed)
    phantom_frame = ds_read.get_frame(0)
    im = ax.imshow(phantom_frame, origin='lower', extent=[0, Lx, 0, Ly], animated=True)
    
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    cbar = fig.colorbar(im, cax=cax, label='n', orientation='vertical')
    cbar.ax.set_ylabel('n', rotation=0, va='center', ha='left')
    
    contour_lines_blue = []
    contour_lines_red = []
    text_labels = []
    max_blobs = len(tracked_blobs)
    
    for _ in range(max_blobs):
        line_blue, = ax.plot([], [], 'b-', linewidth=1.5)
        line_red, = ax.plot([], [], 'r-', linewidth=1.5)
        contour_lines_blue.append(line_blue)
        contour_lines_red.append(line_red)
        txt = ax.text(0, 0, "", color="red", ha='center', va='center', fontsize=8)
        text_labels.append(txt)
    
    time_text = ax.text(1.035, 1.0225, "", transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
    
    personalized_title = personalized_title_with_method(detection_method, density_threshold, tolerance_factor, distance_threshold)
    ax.set_title(f"      Phantom_2.0 tracking, {personalized_title}", pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel('y', rotation='horizontal')

    def update(frame_index):
        time_value = times[frame_index]
        phantom_frame = ds_read.get_frame(frame_index)
        im.set_data(phantom_frame.values)
        cbar.update_normal(im)
        
        # Reset all lines and texts first
        for line_blue, line_red, txt in zip(contour_lines_blue, contour_lines_red, text_labels):
            line_blue.set_data([], [])
            line_red.set_data([], [])
            txt.set_text("")
        
        count = 0
        for trackedblob in tracked_blobs:
            if time_value not in trackedblob.times:
                continue
            
            if count >= max_blobs:
                break
                
            com = trackedblob.centers_of_mass[frame_index]
            coords = trackedblob.contours_coords[frame_index]
            
            original_data = trackedblob.original_contours_data[frame_index]
            original_coords = original_data["coords"] if original_data is not None else None
        
            if com is None or coords is None:
                continue
            
            contour_lines_blue[count].set_data(coords[:, 0], coords[:, 1])
            if original_coords is not None:
                contour_lines_red[count].set_data(original_coords[:, 0], original_coords[:, 1])
            else:
                contour_lines_red[count].set_data([], [])
            text_labels[count].set_position((com[0], com[1]))
            text_labels[count].set_text(str(trackedblob.id))
            
            count += 1

        time_text.set_text(f"t={time_value:.2f}")
        
        return [im, time_text] + contour_lines_blue + contour_lines_red + text_labels

    ani = FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)
    ani.save(save_path, writer="pillow", fps=gif_fps)
    plt.show()
    plt.close(fig)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
import math

def scatter_plot(tracked_blobs, density_threshold, tolerance_factor, detection_method, distance_threshold, domain):
    
    assert len(tracked_blobs) > 0
    n_blobs = len(tracked_blobs)

    custom_colors = [
        'red', 'blue', 'green', 'gold', 'darkviolet', 'orange', 'deeppink', 'sienna', 'gray', 'lightgreen',
        'darkcyan', 'darkslateblue', 'magenta', 'deepskyblue', 'salmon', 'lightsteelblue', 'limegreen', 'crimson',
        'firebrick', 'khaki', 'darkturquoise', 'mediumaquamarine', 'palevioletred', 'rosybrown', 'yellowgreen',
        'gainsboro', 'wheat', 'lightcoral', 'tan', 'indigo'
    ]
    if n_blobs > len(custom_colors):
        raise ValueError(f"Pas assez de couleurs définies pour {n_blobs} blobs ! Ajoutez-en dans custom_colors.")
        
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
    personalized_title = personalized_title_with_method(detection_method, density_threshold, tolerance_factor, distance_threshold)
    ax.set_title(f"Phantom_2.0 tracked blobs scatter plot, {personalized_title}", loc='left')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------  
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def DAG_plot(tracked_blobs, density_threshold, tolerance_factor, detection_method, distance_threshold, save_path=None, ):
    
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
        'wheat', 'lightcoral', 'tan', 'indigo', 'blue'
    ]
    
    if len(components) > len(custom_colors):
        raise ValueError(f"Pas assez de couleurs définies pour {len(components)} blobs ! Ajoutez-en dans custom_colors.")
        
        
    color_map = {}
    for i, comp in enumerate(components):
        for node in comp:
            color_map[node] = custom_colors[i]

    node_colors = [color_map[node] for node in G.nodes]

    # Tracé
    plt.figure(figsize=(8, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        edge_color='gray',
        arrows=True,
        font_weight='bold'
    )

    personalized_title = personalized_title_with_method(detection_method, density_threshold, tolerance_factor, distance_threshold)
    plt.title(f"Phantom_2.0 tracked blobs DAG plot, {personalized_title}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
     
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  
def personalized_title_with_method(detection_method, density_threshold, tolerance_factor, distance_threshold):
    name = detection_method.replace(' ', r'\ ')
    if detection_method == "arbitrary distance threshold":
        return f"(dens_thresh={density_threshold} ; dist_thresh={distance_threshold})\nDetection method: $\\it{{{name}}}$"
    else:
        return f"(dens_thresh={density_threshold} ; tol_factor={tolerance_factor})\nDetection method: $\\it{{{name}}}$"  
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  
    
  
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
    
    
    
    
