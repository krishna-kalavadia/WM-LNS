"""Utility functions for plotting"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from shapely.geometry import MultiPolygon
from matplotlib.patches import Polygon as MplPolygon

def plot_graph_full(G, nodes, obstacle_classes, start_id, goal_id, environment_bounds, path=None, compare_paths=None, generic=False, risk_zones=False):
    """
    Plot the PRM graph along with obstacles, start and goal nodes, and the optimal paths if provided.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    #fig, ax = plt.subplots(figsize=(14, 7))
    x_limit = environment_bounds[0]
    y_limit = environment_bounds[1]
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)


    # Plot obstacles by class with different colors
    if risk_zones:
        color_cycle = ['black', 'red', 'green', 'yellow', 'gray']
    else:
        color_cycle = ['black', 'dimgray', 'silver', 'gainsboro', 'gray']
    for class_idx, obs_class in enumerate(obstacle_classes):
        color = color_cycle[class_idx % len(color_cycle)]
        for obs in obs_class:
            if isinstance(obs, MultiPolygon):
                geoms = obs.geoms
            else:
                geoms = [obs]

            for poly in geoms:
                # Exterior
                if class_idx == 0 or not risk_zones:
                    exterior_coords = list(poly.exterior.coords)
                    patch = MplPolygon(exterior_coords,
                                    closed=True,
                                    facecolor=color,
                                    alpha=1.0,
                                    edgecolor='none')
                    ax.add_patch(patch)
                elif class_idx != 0 and risk_zones:
                    exterior_coords = list(poly.exterior.coords)
                    patch = MplPolygon(exterior_coords,
                                    closed=True,
                                    facecolor=color,
                                    alpha=0.5,
                                    edgecolor='none')
                    ax.add_patch(patch)

                # Interior holes
                for interior in poly.interiors:
                    hole_coords = list(interior.coords)
                    hole = MplPolygon(hole_coords,
                                    closed=True,
                                    facecolor='white',
                                    edgecolor='none')
                    ax.add_patch(hole)

    # Plot our graph edges
    for u, v in G.edges():
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color='lightblue', linewidth=0.5)

    # Plot our graph nodes
    if nodes:
        xs, ys = zip(*nodes)
        ax.scatter(xs, ys, color='blue', s=10)
    ax.scatter(nodes[start_id][0], nodes[start_id][1], color='green', s=100, label='_Start')
    ax.scatter(nodes[goal_id][0], nodes[goal_id][1], color='red', s=100, label='_Goal')

    # Plot a path if provided
    if path:
        path_coords = [nodes[idx] for idx in path]
        path_x, path_y = zip(*path_coords)
        ax.plot(path_x, path_y, color='blue', linewidth=4, label='Path')
    elif compare_paths:
        if len(compare_paths) == 2:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='red', linewidth=4, label='Path 1')
            else:
                ax.plot(path_x, path_y, color='green', linewidth=4, label='WM Path')

            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='green', linewidth=4, label='Path 2')
            else:
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='WS Path')
        if len(compare_paths) == 3:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='red', linewidth=4, label='Path 1')
            else:
                ax.plot(path_x, path_y, color='mediumseagreen', linewidth=4, label='WM')

            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='green', linewidth=4, label='Path 2')
            else:
                ax.plot(path_x, path_y, color='cornflowerblue', linewidth=4, label='WS')

            path_coords = [nodes[idx] for idx in compare_paths[2]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='Path 3')
            else:
                ax.plot(path_x, path_y, color='orange', linewidth=4, label='WM-LNS')
        if len(compare_paths) == 4:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='red', linewidth=4, label='Path 1')
            else:
                ax.plot(path_x, path_y, color='red', linewidth=4, label='WM Path')
            
            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='orange', linewidth=4, label='Path 2')
            else:
                ax.plot(path_x, path_y, color='orange', linewidth=4, label='h-WM Path')

            path_coords = [nodes[idx] for idx in compare_paths[2]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='green', linewidth=4, label='Path 3')
            else:
                ax.plot(path_x, path_y, color='green', linewidth=4, label='WS Path')

            path_coords = [nodes[idx] for idx in compare_paths[3]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='Path 4')
            else:
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='WM-LNS Path')
        if len(compare_paths) == 5:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='red', linewidth=4, label='Path 1')
            else:
                ax.plot(path_x, path_y, color='red', linewidth=4, label='h-WM')
            
            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='orange', linewidth=4, label='Path 2')
            else:
                ax.plot(path_x, path_y, color='orange', linewidth=4, label='h-WM-poly')

            path_coords = [nodes[idx] for idx in compare_paths[2]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='yellow', linewidth=4, label='Path 3')
            else:
                ax.plot(path_x, path_y, color='yellow', linewidth=4, label='h-WM-greedy')

            path_coords = [nodes[idx] for idx in compare_paths[3]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='green', linewidth=4, label='Path 4')
            else:
                ax.plot(path_x, path_y, color='green', linewidth=4, label='WS')

            path_coords = [nodes[idx] for idx in compare_paths[4]]
            path_x, path_y = zip(*path_coords)
            if (generic):
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='Path 5')
            else:
                ax.plot(path_x, path_y, color='blue', linewidth=4, label='WM-LNS')


    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='green', markersize=10),
    #     Line2D([0], [0], marker='o', color='w', label='Goal', markerfacecolor='red', markersize=10),
    # ]

    legend_elements = []

    for idx in range(len(obstacle_classes)):
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                   label=f'Obs. Class {idx+1}',
                                   markerfacecolor=color_cycle[idx % len(color_cycle)],
                                   markersize=10))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=22)

    plt.title("Environment")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.grid(True)
    plt.show()


def plot_graph_sparse(G, nodes, obstacle_classes, start_id, goal_id, environment_bounds, path=None, compare_paths=None, sample_fraction=0.1):
    """
    Plot a sparse representation of our graph if we have super large graphs 
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    x_limit, y_limit = environment_bounds
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)

    # Plot obstacles by class with different colors
    color_cycle = ['black', 'dimgray', 'darkgray', 'lightgray', 'silver', 'gray']
    for class_idx, obs_class in enumerate(obstacle_classes):
        color = color_cycle[class_idx % len(color_cycle)]
        for obs in obs_class:
            if isinstance(obs, MultiPolygon):
                geoms = obs.geoms
            else:
                geoms = [obs]

            for poly in geoms:
                # Exterior
                exterior_coords = list(poly.exterior.coords)
                patch = MplPolygon(exterior_coords,
                                closed=True,
                                facecolor=color,
                                alpha=1.0,
                                edgecolor='none')
                ax.add_patch(patch)

                # Interior holes
                for interior in poly.interiors:
                    hole_coords = list(interior.coords)
                    hole = MplPolygon(hole_coords,
                                    closed=True,
                                    facecolor='white',
                                    edgecolor='none')
                    ax.add_patch(hole)

    # Determine nodes and edges to plot
    nodes_to_plot = set()
    edges_to_plot = set()

    if path:
        path_nodes = set(path)
        nodes_to_plot.update(path_nodes)

    if compare_paths:
        for p in compare_paths:
            path_nodes = set(p)
            nodes_to_plot.update(path_nodes)

    # Sample a subset of the graph
    all_edges = set(G.edges())
    remaining_edges = all_edges - edges_to_plot  
    sample_size = int(len(remaining_edges) * sample_fraction)

    remaining_edges_list = list(remaining_edges)
    actual_sample_size = min(sample_size, len(remaining_edges_list))
    sampled_edges = random.sample(remaining_edges_list, actual_sample_size)

    for edge in sampled_edges:
        edges_to_plot.add(edge)
        nodes_to_plot.add(edge[0])
        nodes_to_plot.add(edge[1])

    # Plot our graph edges
    for u, v in edges_to_plot:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color='lightblue', linewidth=0.5, alpha=0.5)

    # Plot our graph nodes
    sampled_nodes = list(nodes_to_plot)
    if sampled_nodes:
        xs, ys = zip(*[nodes[idx] for idx in sampled_nodes])
        ax.scatter(xs, ys, color='blue', s=10, alpha=0.6)
    ax.scatter(nodes[start_id][0], nodes[start_id][1], color='green', s=100, label='Start')
    ax.scatter(nodes[goal_id][0], nodes[goal_id][1], color='red', s=100, label='Goal')

    # Plot a path if provided
    if path:
        path_coords = [nodes[idx] for idx in path]
        path_x, path_y = zip(*path_coords)
        ax.plot(path_x, path_y, color='orange', linewidth=2, label='Path')
    elif compare_paths:
        if len(compare_paths) == 2:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color='red', linewidth=2, label='WM Path')

            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color='green', linewidth=2, label='WS Path')
        if len(compare_paths) == 3:
            path_coords = [nodes[idx] for idx in compare_paths[0]]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color='red', linewidth=2, label='WM Path')

            path_coords = [nodes[idx] for idx in compare_paths[1]]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color='green', linewidth=2, label='WS Path')

            path_coords = [nodes[idx] for idx in compare_paths[2]]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color='blue', linewidth=2, label='LNS Path')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Goal', markerfacecolor='red', markersize=10),
    ]

    for idx in range(len(obstacle_classes)):
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                   label=f'Obs Class {idx+1}',
                                   markerfacecolor=color_cycle[idx % len(color_cycle)],
                                   markersize=10))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(legend_elements, [h.get_label() for h in legend_elements], loc='upper left', fontsize='medium')

    plt.title("Sparse Environment")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def plot_graph_n_paths(G, nodes, obstacle_classes, start_id, goal_id, environment_bounds, paths=None, path_colors=None):
    """
    Plot the PRM graph along with obstacles, start and goal nodes, and then N provided paths 
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    x_limit = environment_bounds[0]
    y_limit = environment_bounds[1]
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)

    # Plot obstacles by class with different colors
    color_cycle = ['black', 'dimgray', 'darkgray', 'lightgray', 'silver', 'gray']
    for class_idx, obs_class in enumerate(obstacle_classes):
        color = color_cycle[class_idx % len(color_cycle)]
        for obs in obs_class:
            if isinstance(obs, MultiPolygon):
                geoms = obs.geoms
            else:
                geoms = [obs]

            for poly in geoms:
                # Exterior
                exterior_coords = list(poly.exterior.coords)
                patch = MplPolygon(exterior_coords,
                                closed=True,
                                facecolor=color,
                                alpha=1.0,
                                edgecolor='none')
                ax.add_patch(patch)

                # Interior holes
                for interior in poly.interiors:
                    hole_coords = list(interior.coords)
                    hole = MplPolygon(hole_coords,
                                    closed=True,
                                    facecolor='white',
                                    edgecolor='none')
                    ax.add_patch(hole)

    # Plot our graph edges
    for u, v in G.edges():
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color='lightblue', linewidth=0.5)

    # Plot our graph nodes
    if nodes:
        xs, ys = zip(*nodes)
        ax.scatter(xs, ys, color='blue', s=10)
    ax.scatter(nodes[start_id][0], nodes[start_id][1], color='green', s=100, label='Start')
    ax.scatter(nodes[goal_id][0], nodes[goal_id][1], color='red', s=100, label='Goal')

    # Plot a path if provided
    if paths:
        # Check if path colors provided, if so use that
        if path_colors is None or isinstance(path_colors, (str, tuple)):
            colors = [path_colors or 'blue'] * len(paths)
        else:
            colors = path_colors  

        for path, c in zip(paths, colors):
            path_coords = [nodes[idx] for idx in path]
            path_x, path_y = zip(*path_coords)
            ax.plot(path_x, path_y, color=c, linewidth=5)

    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='green', markersize=10),
    #     Line2D([0], [0], marker='o', color='w', label='Goal', markerfacecolor='red', markersize=10),
    # ]

    # for idx in range(len(obstacle_classes)):
    #     legend_elements.append(Line2D([0], [0], marker='s', color='w',
    #                                label=f'Obs Class {idx+1}',
    #                                markerfacecolor=color_cycle[idx % len(color_cycle)],
    #                                markersize=10))

    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(legend_elements, [h.get_label() for h in legend_elements], loc='upper left', fontsize='medium')

    plt.title("Environment")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()
