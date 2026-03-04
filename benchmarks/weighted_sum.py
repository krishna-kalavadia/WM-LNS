"""Solves the discrete multi-objective-optimization path planning via a weighted sum (WS)"""
import networkx as nx
import matplotlib.pyplot as plt
import math
import heapq
from wm_lns.environments.generate_environments import *
from wm_lns.utils.plotting_utils import plot_graph_full
from wm_lns.utils.common_utils import compute_path_objectives


def reconstruct_path(parent, current):
    """
    Reconstruct and return path, just reverses it
    """
    path = []
    u = current
    while u != -1:
        path.append(u)
        u = parent[u]
    return list(reversed(path))


def weighted_sum_solver(G, start, goal, weights, h_list):
    """
    Solve the WS problem with A* on a graph
    """
    N = G.number_of_nodes()
    g_score = [math.inf] * N
    f_score = [math.inf] * N
    closed   = [False] * N
    parent   = [-1] * N

    g_score[start] = 0.0
    f_score[start] = h_list[start]

    open_heap = [(f_score[start], start)]

    while open_heap:
        current_f, u = heapq.heappop(open_heap)
        if closed[u]:
            continue
        if u == goal:
            return reconstruct_path(parent, u)

        closed[u] = True

        for v in G.adj[u]:
            if closed[v]:
                continue

            # Compute weighted‐sum edge cost
            edge_objectives = G.edges[u, v]['objectives']
            edge_cost = 0.0
            for w, o in zip(weights, edge_objectives):
                edge_cost += w * o

            tentative = g_score[u] + edge_cost
            if tentative < g_score[v]:
                parent[v] = u
                g_score[v] = tentative
                f_score[v] = tentative + h_list[v]
                heapq.heappush(open_heap, (f_score[v], v))

    return None


def main_ws(G, start_id, goal_id, obstacle_list, weights, three_dim=False):
    
    if (three_dim):
        N = G.number_of_nodes()
        gx, gy, gz = G.nodes[goal_id]['pos']
        h_list = [0.0] * N 
        for n in G.nodes:
            x, y, z = G.nodes[n]['pos']
            h_list[n] = weights[0] * math.sqrt(
                    (x - gx)**2 + (y - gy)**2 + (z - gz)**2
                )
    else: 
        N = G.number_of_nodes()
        gx, gy = G.nodes[goal_id]['pos']
        h_list = [0.0] * N
        for n in range(N):
            x, y = G.nodes[n]['pos']
            h_list[n] = weights[0] * math.hypot(x - gx, y - gy)

    optimal_path = weighted_sum_solver(G, start_id, goal_id, weights, h_list)
    cumulative_weighted_objectives = None
    if optimal_path is not None:
        cumulative_weighted_objectives = compute_path_objectives(optimal_path, G, weights)
        print(f"Cumulative Objectives for Optimal Path: {cumulative_weighted_objectives}")
        print(f"Cumulative Weighted Sum Cost: {sum(cumulative_weighted_objectives)}")
    else:
        print("No optimal path could be found.")

    return G, obstacle_list, start_id, goal_id, optimal_path, cumulative_weighted_objectives
