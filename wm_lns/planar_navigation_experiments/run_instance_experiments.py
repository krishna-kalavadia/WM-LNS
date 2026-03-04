"""Runtime and Solution Quality Analysis"""

import random
import math
import time
from wm_lns.environments.generate_environments import *
from wm_lns.utils.plotting_utils import plot_graph_full
from wm_lns.utils.common_utils import sample_log_scale
from wm_lns.wm_lns import lns, compute_wm_cost, compute_optimality_error, normalize_weights, normalize_solutions
from benchmarks.weighted_sum import main_ws
from benchmarks.heuristic_weighted_max import main_heuristic_wm
from benchmarks.beam_search_weighted_max import main_beam_search_wm


def instance_1():
    for _ in range(50):
        # Instance 1
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=0, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_1(environment_bounds, 6.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
        
        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.0017, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[0] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="random", iterations=400, non_improving_limit=50)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Budgets are tuned so sub-optimal WM solvers have a similar runtime than WM-LNS
        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=18)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=35)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-1-Small"
        with open("wm_lns/planar_navigation_experiments/data/instance1_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row 
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")

    # =================================

    for _ in range(50):
        # Instance 1
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=0, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_1(environment_bounds, 4.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
        
        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.0017, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[0] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="random", iterations=400, non_improving_limit=50)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Budgets are tuned so sub-optimal WM solvers have a similar runtime than WM-LNS
        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=18)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=35)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-1-Medium"
        with open("wm_lns/planar_navigation_experiments/data/instance1_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row 
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")


    # =================================

    for _ in range(50):
        # Instance 1
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=0, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_1(environment_bounds, 3, 5000, 0.075 * min(environment_bounds), start_node, goal_node)

        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.0017, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[0] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="random", iterations=400, non_improving_limit=50)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Budgets are tuned so sub-optimal WM solvers have a similar runtime than WM-LNS
        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=18)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=30)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-1-Large"
        with open("wm_lns/planar_navigation_experiments/data/instance1_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row 
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")


def instance_2():
    for _ in range(50):
        # Instance 2
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=1, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_2(environment_bounds, 10, 5000, 0.075 * min(environment_bounds), start_node, goal_node)

        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.012, 11.5, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[1] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="guided", iterations=75, non_improving_limit=25)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Budgets are tuned so sub-optimal WM solvers have a similar runtime than WM-LNS
        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=50)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=35)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-2-Small"
        with open("wm_lns/planar_navigation_experiments/data/instance2_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")


    for _ in range(50):
        # Instance 2
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=1, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_2(environment_bounds, 8.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)

        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.012, 11.5, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[1] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="guided", iterations=75, non_improving_limit=25)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Budgets are tuned so sub-optimal WM solvers have a similar runtime than WM-LNS
        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=50)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=30)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-2-Medium"
        with open("wm_lns/planar_navigation_experiments/data/instance2_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row 
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")


    for _ in range(50):
        # Instance 2
        environment_bounds = (500, 500)
        start_node, goal_node = sample_edge_reflected_points(environment_bounds, setting=1, margin=25)
        G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_2(environment_bounds, 7.0, 5000, 0.075 * min(environment_bounds), start_node, goal_node)

        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [0.012, 11.5, 1]
        normalized_weights = normalize_weights(weights)

        if start_node[1] > 50:
            print(start_node)
            plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds)
            raise RuntimeError("Wrong Start Goal Setting")

        # Run LNS
        start_lns = time.time()
        optimal_path_lns, cost_lns, _, _, _ = lns( G, start_id, goal_id, normalized_weights, weight_sampling="guided", iterations=75, non_improving_limit=25)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns

        # Run h-WM
        start_h_wm = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm, optimal_path_h_wm, cost_h_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Run h-WM-poly
        start_h_wm_poly = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_poly, optimal_path_h_wm_poly, cost_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=50)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run h-WM-greedy
        start_h_wm_greedy = time.time()
        G_wm, nodes_wm, obstacle_list_wm, start_id_wm, goal_id_wm_greedy, optimal_path_h_wm_greedy, cost_h_wm_greedy = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=30)
        end_h_wm_greedy = time.time()
        runtime_h_wm_greedy = end_h_wm_greedy - start_h_wm_greedy

        # Run WS
        start_ws = time.time()
        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Compute final costs on optimal WM and WS with same weights
        cost_h_wm, all_objectives_h_wm = compute_wm_cost(optimal_path_h_wm, G, normalized_weights)
        cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
        cost_h_wm_greedy, all_objectives_h_wm_greedy = compute_wm_cost(optimal_path_h_wm_greedy, G, normalized_weights)
        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
        cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)

        # Compute cost ratios
        h_wm_poly_ratio = cost_h_wm_poly/cost_h_wm
        h_wm_greedy_ratio = cost_h_wm_greedy/cost_h_wm
        ws_ratio = cost_ws_on_wm/cost_h_wm
        lns_ratio = cost_lns/cost_h_wm

        # Compute the optimality error between objectives that are performing strictly worse
        error_h_wm_poly_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_poly)
        error_h_wm_greedy_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_h_wm_greedy)
        error_ws_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_ws)
        error_lns_wm = compute_optimality_error(all_objectives_h_wm, all_objectives_lns)

        # Compute spatial distance between WS vs WM and LNS vs WM solution on the pareto front
        # Use unweighted objective values
        all_objectives_h_wm_unweighted = [all_objectives_h_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_h_wm_greedy_unweighted = [all_objectives_h_wm_greedy[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
        all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

        (
            all_objectives_h_wm_unweighted_normalized,
            all_objectives_h_wm_poly_unweighted_normalized,
            all_objectives_h_wm_greedy_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(all_objectives_h_wm_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_h_wm_greedy_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

        distance_h_wm_poly_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_poly_unweighted_normalized)
        distance_h_wm_greedy_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_h_wm_greedy_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_h_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        # Write Results to file
        environment_label = "Instance-2-Large"
        with open("wm_lns/planar_navigation_experiments/data/instance2_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_h_wm:.5f},1.00000,0.00000,0.00000\n")
            # h-WM-poly row
            f.write(f"{environment_label},WM-poly,{runtime_h_wm_poly:.5f},{cost_h_wm_poly:.5f},{h_wm_poly_ratio:.5f},{error_h_wm_poly_wm:.5f},{distance_h_wm_poly_wm:.5f}\n")
            # h-WM-greedy row
            f.write(f"{environment_label},WM-beam,{runtime_h_wm_greedy:.5f},{cost_h_wm_greedy:.5f},{h_wm_greedy_ratio:.5f},{error_h_wm_greedy_wm:.5f},{distance_h_wm_greedy_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws_on_wm:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")


if __name__ == "__main__":
    # Run runtime and solution quality experiments
    instance_1()
    instance_2()
