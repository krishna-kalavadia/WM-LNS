"""Generates the various environments on which the solvers are run"""

import networkx as nx
from shapely.geometry import Point, LineString
import random
import math
from scipy.spatial import KDTree
from wm_lns.utils.plotting_utils import *
import pandas as pd
import random
from shapely.geometry import Polygon
from shapely.geometry import box

def export_nodes_to_csv(nodes, filename=""):
    """
    Exports node coordinates to a CSV file.
    """
    df = pd.DataFrame(nodes, columns=['x', 'y'])
    df.index.name = 'node_id'  
    df.reset_index(inplace=True)
    df.to_csv(filename, index=False)
    print(f"Nodes exported to {filename}")


def import_nodes_from_csv(filename=""):
    """
    Imports node coordinates from a CSV file.
    """
    try:
        df = pd.read_csv(filename)
        if not {'node_id', 'x', 'y'}.issubset(df.columns):
            raise ValueError("CSV file in wrong format.")
        
        df_sorted = df.sort_values(by='node_id').reset_index(drop=True)
        
        nodes = list(zip(df_sorted['x'], df_sorted['y']))
        
        print(f"Imported {len(nodes)} nodes from {filename}.")
        return nodes
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []
    except Exception as e:
        print(f"An error occurred while importing nodes: {e}")
        return []
    

def generate_indoor_instance_1(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, buffer_distance=2.5):
    """
    Map used for Instance 1: Maze
    """

    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 7.5
    walls = [
        box(0, -half_thick, width, half_thick), 
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick, height), 
        box(width - half_thick, 0, width + half_thick, height) 
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Obstacles inside room
    # Central walls
    add_block(0.55, 0.5, 0.4, 0.02)

    add_block(0.45, 0.59, 0.2, 0.02)
    add_block(0.4, 0.68, 0.02, 0.19)

    # Vertical block hanging from top
    add_block(0.75, 0.9, 0.02, 0.255)

    # L central shape 
    add_block(0.63, 0.29, 0.02, 0.2)
    add_block(0.6, 0.4, 0.25, 0.02)

    # Bottom right vertical block & lower horizontal block sticking out
    add_block(0.3, 0.1, 0.02, 0.2)

    nodes = generate_nodes(obstacle_list,
                           min_node_distance,
                           max_nodes,
                           environment_bounds,
                           start_node,
                           goal_node)

    obstacle_classes = [obstacle_list]
    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)
    
    return G, nodes, start_id, goal_id, obstacle_classes


def generate_indoor_instance_2(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, buffer_distance=2.5):
    """
    Build a environment simulating an indoor map
    """
    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 7.5
    walls = [
        box(0, -half_thick, width, half_thick), 
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick, height), 
        box(width - half_thick, 0, width + half_thick, height) 
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )
    
    # Obstacles inside room
    add_block(0.38, 0.4, 0.1, 0.1)
    add_block(0.42, 0.65, 0.1, 0.1)
    add_block(0.65, 0.63, 0.1, 0.1)
    add_block(0.65, 0.36, 0.1, 0.1)

    # obstacle_classes = [walls, [obstacle_list[4], obstacle_list[7]], [obstacle_list[5], obstacle_list[6]]]
    
    risk_zones = []
    def add_risk_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        risk_zones.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Low Risk Zone
    add_risk_block(0.49, 0.5, 0.3, 0.4)
    
    # High Risk Zones
    add_risk_block(0.2, 0.7, 0.35, 0.2)
    add_risk_block(0.84, 0.6, 0.27, 0.2)

    obstacle_classes = [walls + obstacle_list]

    nodes = generate_nodes(obstacle_list,
                           min_node_distance,
                           max_nodes,
                           environment_bounds,
                           start_node,
                           goal_node)
    if len(nodes) < 2:
        raise RuntimeError("Could not generate enough indoor nodes.")
    
    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)

    # Now assign risk values
    risk_map = [
      (risk_zones[0], 1),
      (risk_zones[1], 2),
      (risk_zones[2], 2)
    ]

    node_risk = []
    for (x, y) in nodes:
        p = Point(x, y)
        r = 0.0
        for geom, severity in risk_map:
            if geom.covers(p):            
                r = max(r, severity)
        node_risk.append(r)

    for u, v, data in G.edges(data=True):
        length = float(data['weight'])
        obs_dist = float(data['objectives'][1])
        risk_cost = max(node_risk[u], node_risk[v])
        data['objectives'] = [length, obs_dist, risk_cost]

    obstacle_classes = [walls + obstacle_list, [risk_zones[1], risk_zones[2]], [risk_zones[0]]]

    return G, nodes, start_id, goal_id, obstacle_classes


def is_node_valid(node, obstacle_list, min_distance, existing_nodes):
    """
    Check if the provided node is inside any of the obstacles
    or too close to existing nodes.
    """
    point = Point(node)

    # Check if inside any obstacle
    for obstacle in obstacle_list:
        if obstacle.contains(point):
            return False
    
    # Check minimum distance from existing points
    for existing_node in existing_nodes:
        if math.hypot(node[0] - existing_node[0], node[1] - existing_node[1]) < min_distance:
            return False
    
    return True


def generate_nodes(obstacle_list, min_distance, max_points, environment_bounds, start_node, goal_node):
    """
    Generate a uniform sample of nodes within our environment that are obstacle free
    """
    nodes = [start_node, goal_node]
    x_limit = environment_bounds[0]
    y_limit = environment_bounds[1]
    max_attempts = 5000
    # max_attempts = 10000

    attempts = 0
    while len(nodes) < max_points and attempts < max_attempts:
        x = random.uniform(0, x_limit)
        y = random.uniform(0, y_limit)
        node = (x, y)

        if is_node_valid(node, obstacle_list, min_distance, nodes):
            nodes.append(node)
        attempts += 1

    if attempts == max_attempts:
        print(f"Generated {len(nodes)} points out of {max_points} desired")
    
    return nodes


def is_edge_valid(node1, node2, obstacle_list):
    """
    Check if the segment between two nodes is collision-free.
    """
    line = LineString([node1, node2])
    for obstacle in obstacle_list:
        if line.crosses(obstacle) or line.within(obstacle) or line.touches(obstacle):
            return False
    return True


def build_prm(nodes, connection_radius, obstacle_list, environment_bounds):
    """
    Build the PRM graph by connecting nodes within a certain radius.
    """
    G = nx.Graph()
    # G.add_nodes_from(range(len(nodes))) 

    for idx, xy in enumerate(nodes):
        G.add_node(idx, pos=xy)   

    kdtree = KDTree(nodes)
    for idx, node in enumerate(nodes):
        indices = kdtree.query_ball_point(node, connection_radius)
        
        # Connect our current node to all its neighbors
        for neighbor_idx in indices:
            if neighbor_idx == idx:
                continue 
            if G.has_edge(idx, neighbor_idx):
                continue
            
            neighbor_node = nodes[neighbor_idx]
            if is_edge_valid(node, neighbor_node, obstacle_list):
                distance = math.hypot(neighbor_node[0] - node[0], neighbor_node[1] - node[1])
                G.add_edge(idx, neighbor_idx, weight=distance)
    
    print(f"PRM generated with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    return G


def compute_distances_to_obstacles(nodes, obstacle_classes):
    """
    Compute distances from each node to each class of obstacles.
    """
    node_obstacle_dists = []

    for node in nodes:
        point = Point(node)
        per_class_distances = []

        for obs_class in obstacle_classes:
            min_dist = float('inf')
            for obs in obs_class:
                dist = obs.exterior.distance(point)
                if dist < min_dist:
                    min_dist = dist
            per_class_distances.append(min_dist)

        node_obstacle_dists.append(per_class_distances)

    return node_obstacle_dists


def assign_edge_objectives(G, obstacle_distances):
    """
    Assign objectives to each node in our graph
    """
    for u, v, data in G.edges(data=True):
        obj_vector = []

        # Objective 0 is path length
        obj_vector.append(data['weight'])

        # Objectives 1...n are obs dist or will overwritten with risk later
        u_dists = obstacle_distances[u]
        v_dists = obstacle_distances[v]

        for class_idx in range(len(u_dists)):
            min_dist = min(u_dists[class_idx], v_dists[class_idx])
            if min_dist == 0:
                obj_vector.append(float('inf')) 
            else:
                obj_vector.append(1.0 / min_dist)

        data['objectives'] = obj_vector

    return G


def generate_indoor_pf_map(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, buffer_distance=2.5):
    """
    Build a environment simulating an indoor map for checking sol diversity
    """

    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 5
    walls = [
        box(0, -half_thick, width, half_thick * 1.4),
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick * 1.95, height), 
        box(width - half_thick, 0, width + half_thick, height) 
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Obstacles inside room
    add_block(0.05, 0.1, 0.075, 0.35)
    add_block(0.2, 0.95, 0.35, 0.05)

    add_block(0.95, 0.95, 0.4, 0.1)
    add_block(0.95, 0.95, 0.1, 0.55)

    add_block(0.75, 0.25, 0.45, 0.45)
    add_block(0.587, 0.5, 0.125, 0.05)

    add_block(0.395, 0.39, 0.1, 0.15)
    add_block(0.375, 0.62, 0.15, 0.075)
    add_block(0.675, 0.65, 0.05, 0.1)

    nodes = generate_nodes(obstacle_list,
                           min_node_distance,
                           max_nodes,
                           environment_bounds,
                           start_node,
                           goal_node)

    obstacle_classes = [obstacle_list]

    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)
    
    return G, nodes, start_id, goal_id, obstacle_classes


def generate_indoor_pf_map_from_nodes(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, nodes, buffer_distance=2.5):
    """
    Build a environment simulating an indoor map for checking sol diversity
    Import Existing Nodes
    """

    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 5
    walls = [
        box(0, -half_thick, width, half_thick * 1.4),
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick * 1.95, height), 
        box(width - half_thick, 0, width + half_thick, height)
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Obstacles inside room
    add_block(0.05, 0.1, 0.075, 0.35)
    add_block(0.2, 0.95, 0.35, 0.05)

    add_block(0.95, 0.95, 0.4, 0.1)
    add_block(0.95, 0.95, 0.1, 0.55)

    add_block(0.75, 0.25, 0.45, 0.45)
    add_block(0.587, 0.5, 0.125, 0.05)

    add_block(0.395, 0.39, 0.1, 0.15)
    add_block(0.375, 0.62, 0.15, 0.075)
    add_block(0.675, 0.65, 0.05, 0.1)

    obstacle_classes = [obstacle_list]

    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)
    
    return G, nodes, start_id, goal_id, obstacle_classes


def generate_indoor_pf_map_with_risk(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, buffer_distance=2.5):
    """
    Build a environment simulating an indoor map for checking sol diversity, this time with risk so three objectives
    """

    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 5
    walls = [
        box(0, -half_thick, width, half_thick * 1.4), 
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick * 1.95, height), 
        box(width - half_thick, 0, width + half_thick, height) 
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Obstacles inside room
    add_block(0.05, 0.1, 0.075, 0.35)
    add_block(0.2, 0.95, 0.35, 0.05)

    add_block(0.95, 0.95, 0.4, 0.1)
    add_block(0.95, 0.95, 0.1, 0.55)

    add_block(0.75, 0.25, 0.45, 0.45)
    add_block(0.587, 0.5, 0.125, 0.05)

    add_block(0.395, 0.39, 0.1, 0.15)
    add_block(0.375, 0.62, 0.15, 0.075)
    add_block(0.675, 0.65, 0.05, 0.1)

    risk_zones = []
    def add_risk_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        risk_zones.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # High Risk Zone
    add_risk_block(0.47, 0.79, 0.36, 0.25)

    # Low Risk Zone
    add_risk_block(0.41, 0.52, 0.22, 0.11)
    
    # Medium Risk Zone
    add_risk_block(0.65, 0.56, 0.1, 0.07)

    nodes = generate_nodes(obstacle_list,
                           min_node_distance,
                           max_nodes,
                           environment_bounds,
                           start_node,
                           goal_node)

    obstacle_classes = [obstacle_list]

    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)

    # Now assign risk values
    risk_map = [
      (risk_zones[0], 15),
      (risk_zones[1], 1),
      (risk_zones[2], 5)
    ]

    node_risk = []
    for (x, y) in nodes:
        p = Point(x, y)
        r = 0.0
        for geom, severity in risk_map:
            if geom.covers(p):        
                r = max(r, severity)
        node_risk.append(r)

    for u, v, data in G.edges(data=True):
        length = float(data['weight'])
        obs_dist = float(data['objectives'][1])
        risk_cost = max(node_risk[u], node_risk[v])

        data['objectives'] = [length, obs_dist, risk_cost]

    obstacle_classes = [obstacle_list, [risk_zones[0]], [risk_zones[1]], [risk_zones[2]]]
    
    return G, nodes, start_id, goal_id, obstacle_classes


def generate_indoor_pf_map_with_risk_from_nodes(environment_bounds, min_node_distance, max_nodes, connection_radius, start_node, goal_node, nodes, buffer_distance=2.5):
    """
    Build a environment simulating an indoor map for checking sol diversity, this time with risk so three objectives
    Import existing nodes
    """

    W, H = environment_bounds
    width, height = environment_bounds
    obstacle_list = []
    
    # Room walls
    half_thick = buffer_distance * 5
    walls = [
        box(0, -half_thick, width, half_thick * 1.4), 
        box(0, height - half_thick, width, height + half_thick), 
        box(-half_thick, 0, half_thick * 1.95, height), 
        box(width - half_thick, 0, width + half_thick, height)
    ]
    obstacle_list.extend(walls)

    def add_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        obstacle_list.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # Obstacles inside room
    add_block(0.05, 0.1, 0.075, 0.35)
    add_block(0.2, 0.95, 0.35, 0.05)

    add_block(0.95, 0.95, 0.4, 0.1)
    add_block(0.95, 0.95, 0.1, 0.55)

    add_block(0.75, 0.25, 0.45, 0.45)
    add_block(0.587, 0.5, 0.125, 0.05)

    # Generate Obstacles
    add_block(0.395, 0.39, 0.1, 0.15)
    add_block(0.375, 0.62, 0.15, 0.075)
    add_block(0.675, 0.65, 0.05, 0.1)

    risk_zones = []
    def add_risk_block(cx_frac, cy_frac, w_frac, h_frac):
        cx, cy = cx_frac*W, cy_frac*H
        hw, hh = (w_frac*W)/2, (h_frac*H)/2
        risk_zones.append(
            box(cx-hw, cy-hh, cx+hw, cy+hh).buffer(buffer_distance)
        )

    # High Risk Zone
    add_risk_block(0.47, 0.79, 0.36, 0.25)

    # Low Risk Zone
    add_risk_block(0.41, 0.52, 0.22, 0.11)
    
    # Medium Risk Zone
    add_risk_block(0.65, 0.56, 0.1, 0.07)

    obstacle_classes = [obstacle_list]

    G = build_prm(nodes, connection_radius, obstacle_list, environment_bounds)
    start_id, goal_id = 0, 1
    obstacle_distances = compute_distances_to_obstacles(nodes, obstacle_classes)
    G = assign_edge_objectives(G, obstacle_distances)

    # Now assign risk values
    risk_map = [
      (risk_zones[0], 10),
      (risk_zones[1], 1),
      (risk_zones[2], 5)
    ]

    node_risk = []
    for (x, y) in nodes:
        p = Point(x, y)
        r = 0.0
        for geom, severity in risk_map:
            if geom.covers(p):         
                r = max(r, severity)
        node_risk.append(r)

    for u, v, data in G.edges(data=True):
        length = float(data['weight'])
        obs_dist = float(data['objectives'][1])

        risk_cost = max(node_risk[u], node_risk[v])

        data['objectives'] = [length, obs_dist, risk_cost]

    obstacle_classes = [obstacle_list, [risk_zones[0]], [risk_zones[1]], [risk_zones[2]]]
    
    return G, nodes, start_id, goal_id, obstacle_classes


def sample_edge_reflected_points(bounds, margin, setting=0, seed=None):
    """
    Sample points on opposite ends of the map
    """
    rng = random.Random(seed)
    w, h = bounds
    minx, maxx = margin, w - margin
    miny, maxy = margin, h - margin

    # side = rng.choice(['top','bottom','left','right'])
    # side = rng.choice(['left','right'])

    # if setting == 0:
    #     side = rng.choice(['left','right'])
    # else:
    #     side = rng.choice(['top','bottom'])

    if setting == 0:
        side = rng.choice(['left'])
    else:
        side = rng.choice(['bottom'])
    
    if side == 'top':
        sx, sy = rng.uniform(minx, maxx), maxy
    elif side == 'bottom':
        sx, sy = rng.uniform(minx, maxx), miny
    elif side == 'left':
        sx, sy = minx, rng.uniform(miny, maxy)
    elif side == 'right':
        sx, sy = maxx, rng.uniform(miny, maxy)

    cx, cy = w/2.0, h/2.0
    gx, gy = 2*cx - sx, 2*cy - sy
    gx = min(max(gx, minx), maxx)
    gy = min(max(gy, miny), maxy)

    return (sx, sy), (gx, gy)


if __name__ == "__main__":
    # environment_bounds = (500, 500)
    # start_node = (25, 25)
    # goal_node = (475, 475)
    # G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_1(environment_bounds, 12.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
    # plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)

    # environment_bounds = (500, 500)
    # start_node = (25, 25)
    # goal_node = (475, 475)
    # G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_2(environment_bounds, 12.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
    # plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, risk_zones=True)

    environment_bounds = (500, 500)
    start_node = (200, 25)
    goal_node = (475, 300)
    G, nodes, start_id, goal_id, obstacle_list = generate_indoor_pf_map_with_risk(environment_bounds, 11.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
    plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, risk_zones=True)
