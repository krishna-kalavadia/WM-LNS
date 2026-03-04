"""Solves the discrete multi-objective-optimization path planning via Beam Search Weighted Max"""
import math
import heapq
from itertools import count
from wm_lns.environments.generate_environments import *
from wm_lns.utils.plotting_utils import plot_graph_full, plot_graph_sparse
from wm_lns.utils.common_utils import compute_path_objectives


def compute_path_objectives(path, G, weights):
    """
    Compute the weighted objectives for a given path
    """
    num_objectives = len(weights)
    cumulative_weighted_objectives = [0.0] * len(weights)

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_objectives = G.edges[u, v]['objectives']

        for j in range(num_objectives):
            cumulative_weighted_objectives[j] += weights[j] * edge_objectives[j]
    
    return cumulative_weighted_objectives


def dominates(path1_objectives, path2_objectives):
    """
    Check if path1's objectives dominates path2's objectives.
    
    Note: path1 dominates path2 if all are path1's objectives are better than or equal to path2
          and there is at least one objective in path one that is better
    """
    better_in_at_least_one = False
    for a, b in zip(path1_objectives, path2_objectives):
        if a > b:
            return False
        elif a < b:
            better_in_at_least_one = True
    return better_in_at_least_one


def beam_weighted_max_solver(G, start_id, goal_id, weights, budget=None, three_dim=False):
    """
    WM-beam
    """
    if three_dim:
        num_obj = len(weights)
        gx, gy, gz = G.nodes[goal_id]["pos"]
        h = {
            n: [
                math.sqrt(
                    (G.nodes[n]["pos"][0] - gx) ** 2
                + (G.nodes[n]["pos"][1] - gy) ** 2
                + (G.nodes[n]["pos"][2] - gz) ** 2
                )
            ] + [0.0] * (num_obj - 1)
            for n in G.nodes()
        }
    else:
        num_obj = len(weights)
        gx, gy = G.nodes[goal_id]['pos']
        h = {
            n: [math.hypot(G.nodes[n]['pos'][0] - gx,
                        G.nodes[n]['pos'][1] - gy)] + [0.0] * (num_obj - 1)
            for n in G.nodes()
        }

    # Constants
    rho = 1e-6  # Tie breaker constant
    num_objectives = len(weights)

    # Keep a counter for each node since sometimes paths costs are the same and then it'll try to compare objectives
    counter = count()

    # Initial objectives
    initial_objectives = [0.0] * num_objectives
    initial_raw_objectives = [0.0] * num_objectives
    initial_cost = max(initial_objectives[i] + weights[i]*h[start_id][i] for i in range(num_obj))

    # Priority queue structure: (current_cost, unique_counter, current_node, cumulative_objectives, path)
    heap = []
    heapq.heappush(heap, (initial_cost, next(counter), start_id, initial_objectives, initial_raw_objectives, [start_id]))

    # Keep track of best current cost vectors for each node
    open_list = {start_id: [(initial_raw_objectives, initial_objectives, [start_id])]}

    while heap:
        current_cost, _, current_node, cumulative_objectives, raw_cumulative_objectives, path = heapq.heappop(heap)
        
        # Skip states that were pruned from this node's open list
        if (raw_cumulative_objectives, cumulative_objectives, path) not in open_list.get(current_node, []):
            continue

        # If goal is reached, return current path
        if current_node == goal_id:
            return path

        for neighbor in G.neighbors(current_node):
            # Avoid Cycles
            if neighbor in path:
                continue

            # Avoid Duplicates
            new_prefix = path + [neighbor]
            if new_prefix in [p for (_ ,_ , p) in open_list.get(neighbor, [])]:
                continue

            # Create our "tentative path" by computing our new objectives
            new_edge_objectives = G.edges[current_node, neighbor]['objectives']
            new_cumulative_objectives = list(cumulative_objectives)
            new_raw_cumulative_objectives = list(raw_cumulative_objectives)

            # Compute new objectives
            for i in range(num_objectives):
                new_cumulative_objectives[i] += weights[i] * new_edge_objectives[i]
                new_raw_cumulative_objectives[i] += new_edge_objectives[i]

            # Add our heuristic
            f_vals = [
                new_cumulative_objectives[i] + weights[i] * h[neighbor][i]
                for i in range(num_obj)
            ]

            # Check any existing path dominate our "tentative path" to current neighbor or if 
            # our "tentative path" dominates any existing paths
            dominated = False
            paths_to_remove = []
            for (existing_path_raw_objectives, existing_path_weighted_objectives, path_to_neighbor) in open_list.get(neighbor, []):
                if dominates(existing_path_raw_objectives, new_raw_cumulative_objectives):
                    dominated = True
                    break
                if dominates(new_raw_cumulative_objectives, existing_path_raw_objectives):
                    paths_to_remove.append((existing_path_raw_objectives, existing_path_weighted_objectives, path_to_neighbor))

            # Current path is dominated by an existing path, skip it
            if dominated:
                continue 

            # Remove any existing paths that are dominated by the new path
            for dominated_path in paths_to_remove:
                open_list[neighbor].remove(dominated_path)

            open_list.setdefault(neighbor, []).append((new_raw_cumulative_objectives, new_cumulative_objectives, path + [neighbor]))
            
            # Beam Search part
            if budget is not None and len(open_list[neighbor]) > budget:
                # Find the worst cost and remove it from our open list for that node
                worst = max(open_list[neighbor], key=lambda tup: max(tup[1]))
                open_list[neighbor].remove(worst)

                # If we just kicked out the very one we tried to add, skip pushing it
                if worst == (new_raw_cumulative_objectives, new_cumulative_objectives, path + [neighbor]):
                    continue

            new_path_cost = max(f_vals) + rho * sum(new_raw_cumulative_objectives)
            heapq.heappush(heap, (new_path_cost, next(counter), neighbor, new_cumulative_objectives, new_raw_cumulative_objectives, path + [neighbor]))

    print("No path found.")
    return None


def main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, weights, budget=None, three_dim=False):
    """
    Run our WM-beam
    """    
    
    # Find the optimal path using WM
    if budget == None:
        optimal_path = beam_weighted_max_solver(G, start_id, goal_id, weights, three_dim=three_dim)
    else:
        optimal_path = beam_weighted_max_solver(G, start_id, goal_id, weights, budget=budget, three_dim=three_dim)
    
    cumulative_weighted_objectives = None
    if optimal_path:
        cumulative_weighted_objectives = compute_path_objectives(optimal_path, G, weights)
        print(f"Cumulative Objectives for Optimal Path: {cumulative_weighted_objectives}")
        print(f"Cumulative Weighted Max Cost: {max(cumulative_weighted_objectives)}")
    else:
        print("No optimal path could be found.")
    
    return G, nodes, obstacle_list, start_id, goal_id, optimal_path, cumulative_weighted_objectives
