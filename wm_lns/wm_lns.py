"""Solves the discrete multi-objective-optimization path planning via Large Neighborhood Search (LNS)"""

import math
import time
import random
import cProfile
import networkx as nx
import numpy as np
from wm_lns.environments.generate_environments import *
from wm_lns.utils.plotting_utils import plot_graph_full, plot_graph_sparse
from benchmarks.weighted_sum import main_ws, weighted_sum_solver
from benchmarks.heuristic_weighted_max import main_heuristic_wm
from benchmarks.beam_search_weighted_max import main_beam_search_wm,  beam_weighted_max_solver
from wm_lns.utils.common_utils import *


def compute_wm_cost(path, G, weights):
    """
    Compute the Weighted Max (WM) value for provided path
    """
    num_objectives = len(weights)

    cum_objectives = [0.0] * num_objectives

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_objectives = G.edges[u, v]['objectives']

        for j in range(num_objectives):
            cum_objectives[j] += weights[j] * edge_objectives[j]
    
    wm_cost = max(cum_objectives) 
    return wm_cost, cum_objectives


def compute_wm_cost_tie_breaker(path, G, weights):
    """
    Compute the Weighted Max (WM) value for provided path, used when solving LNS 
    """
    num_objectives = len(weights)
    cum_objectives = [0.0] * num_objectives
    raw_cum_objectives = [0.0] * num_objectives

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_objectives = G.edges[u, v]['objectives']

        for j in range(num_objectives):
            cum_objectives[j] += weights[j] * edge_objectives[j]
            raw_cum_objectives[j] += edge_objectives[j]
    
    rho = 1e-6  # Tie breaker constant
    wm_cost = max(cum_objectives) + (rho * sum(raw_cum_objectives))
    return wm_cost, cum_objectives


def random_removal(path, k):
    """
    Destroy operation: Randomly remove k vertices from the current solution
    """

    if k <= 0 or k > len(path) - 2:
        print("Not enough vertices to remove")
        return None
    
    min_start = 1
    max_start = len(path) - k - 1

    start_index = random.randint(min_start, max_start)
    removed_indices = range(start_index, start_index + k)

    return removed_indices


def remove_worst_edges(path, k, prefix_sums, num_objectives):
    """
    Destroy operation: Remove k vertices based which segment has the best WM cost (best)
    """
    max_wm_cost = -float('inf')
    removed_indices = None

    if k > len(path) - 2:
        print("Not enough vertices to remove")
        return None

    # First score segments of k vertices, i.e for each k segment compute a WM score 
    for i in range(1, len(path) - k):
        segment = [ 
            prefix_sums[j][i+k] - prefix_sums[j][i]
            for j in range(num_objectives)
        ]
        current_wm_cost = max(segment)

        if current_wm_cost > max_wm_cost:
            max_wm_cost = current_wm_cost
            removed_indices = range(i, i + k)
    
    if removed_indices is None:
        print("No valid segment found to remove.")
    
    return removed_indices


def remove_best_edges(path, k, prefix_sums, num_objectives):
    """
    Destroy operation: Remove k vertices based which segment has the lowest WM cost (best)
    """
    min_wm_cost = float('inf')
    removed_indices = None

    if k > len(path) - 2:
        print("Not enough vertices to remove")
        return None

    # First score segments of k vertices, i.e for each k segment compute a WM score 
    for i in range(1, len(path) - k):
        segment = [ 
            prefix_sums[j][i+k] - prefix_sums[j][i]
            for j in range(num_objectives)
        ]
        current_wm_cost = max(segment)

        if current_wm_cost <  min_wm_cost:
            min_wm_cost = current_wm_cost
            removed_indices = range(i, i + k)
    
    if removed_indices is None:
        print("No valid segment found to remove.")
    
    return removed_indices


def remove_balanced_objective_edges(path, k, prefix_sums, num_objectives):
    """
    Destroy operation: Remove k vertices based which segment has the most balanced WM cost
    """
    min_objective_mad = float('inf')
    removed_indices = None

    if k > len(path) - 2:
        print("Not enough vertices to remove")
        return None

    # First score segments of k vertices, i.e for each k segment compute a WM score
    for i in range(1, len(path) - k):
        segment_costs = [
            prefix_sums[j][i+k] - prefix_sums[j][i]
            for j in range(num_objectives)
        ]

        objective_mad = mean_absolute_deviation(segment_costs)

        if objective_mad <  min_objective_mad:
            min_objective_mad = objective_mad
            removed_indices = range(i, i + k)

    if removed_indices is None:
        print("No valid segment found to remove.")

    return removed_indices


def remove_unbalanced_objective_edges(path, k, prefix_sums, num_objectives):
    """
    Destroy operation: Remove k vertices based which segment has the most unbalanced WM cost
    """
    max_objective_mad = float('-inf')
    removed_indices = None

    if k > len(path) - 2:
        print("Not enough vertices to remove")
        return None

    # First score segments of k vertices, i.e for each k segment compute a WM score
    for i in range(1, len(path) - k):
        segment_costs = [
            prefix_sums[j][i+k] - prefix_sums[j][i]
            for j in range(num_objectives)
        ]

        objective_mad = mean_absolute_deviation(segment_costs)

        if objective_mad >  max_objective_mad:
            max_objective_mad = objective_mad
            removed_indices = range(i, i + k)

    if removed_indices is None:
        print("No valid segment found to remove.")

    return removed_indices


def repair_with_ws(G, removed_indices, path, weights, modified_weights, heuristic):
    """
    Reconnect paths using WS with the provided weights
    """

    removed_start_idx = removed_indices.start
    removed_end_idx = removed_indices.stop - 1
    if removed_start_idx <= 0 or removed_end_idx >= len(path) - 1:
        print("Removed indices are out of valid range.")
        return None, None, None, None

    start_repair = path[removed_start_idx - 1]
    end_repair = path[removed_end_idx + 1]

    try:        
        new_subpath = weighted_sum_solver(G, start_repair, end_repair, modified_weights, heuristic)
        assert(new_subpath[0] == start_repair)
        assert(new_subpath[-1] == end_repair)

        # If new_subpath is trivial, return the original path.
        if len(new_subpath) == 1:
            path_cost, path_objectives = compute_wm_cost_tie_breaker(path, G, weights)
            return path, path_cost, modified_weights, path_objectives
        
        repaired_path = path[:removed_start_idx] + new_subpath[1:-1] + path[removed_end_idx + 1:]
        repaired_path_cost, path_objectives = compute_wm_cost_tie_breaker(repaired_path, G, weights)
        return repaired_path, repaired_path_cost, modified_weights, path_objectives
    except nx.NetworkXNoPath:
        print(f"No path found between {start_repair} and {end_repair} during repair.")
        return None, None, None, None


def project_to_simplex(y):
    """
    Projection of vector v onto simplex
    using Condat's algorithm (2016) (see https://lcondat.github.io/publis/Condat_simplexproj.pdf)

    Recreated pseudo code from the proposed algorithm of the paper.
    """
    N = len(y)
    if N == 0:
        return np.array([])

    # Step 1
    v = [y[0]]           
    v_tilde = []         
    rho = y[0] - 1.0    

    # Step 2
    for yn in y[1:]:
        if yn > rho:                         
            rho += (yn - rho) / (len(v) + 1)
            if rho > yn - 1.0:
                v.append(yn)                  
            else:
                v_tilde.append(v.copy())        
                v = [yn]                        
                rho = yn - 1.0                  

    # Step 3
    for block in v_tilde:
        for yi in block:
            if yi > rho:                       
                v.append(yi)
                rho += (yi - rho) / len(v)

    # Step 4
    changed = True
    while changed:
        changed = False
        for yi in v.copy():
            if yi <= rho:                     
                v.remove(yi)
                if len(v) > 0:
                    rho += (rho - yi) / len(v)
                changed = True

    # Step 5
    tau = rho

    # Step 6
    return np.maximum(y - tau, 0.0)


def guided_repair(G, removed_indices, path, weights, heuristic):
    """
    Conduct a repair operation on the destroyed with weight sampling guided by a direct search
    """

    # Sample initial weights
    initial_weights = [sample_log_scale(1, 100000000) for _ in range(len(weights))]
    best_weights = np.array(normalize_weights(initial_weights))
    best_path, best_cost, _, best_objectives = repair_with_ws(G, removed_indices, path, weights, best_weights, heuristic)

    # Determine polling directions
    dim_weight_space = len(weights)
    directions = []
    for i in range(dim_weight_space):
        e = np.zeros(dim_weight_space)
        e[i] = 1
        directions.append(e)
    directions.append(-np.ones(dim_weight_space))

    # Search parameters
    delta_max = 0.25
    delta_min = 0.5 * delta_max 
    iter_count = 0

    while delta_max > 1e-4 and iter_count < 2:
        improvement = False
        for d in directions:

            candidate_weights = best_weights + random.uniform(delta_min, delta_max) * d

            candidate_weights = project_to_simplex(np.array(candidate_weights, dtype=float))
            path_candidate, candidate_cost, _, candidate_objectives = repair_with_ws(G, removed_indices, path, weights, candidate_weights, heuristic)
            if candidate_cost < best_cost:
                best_path = path_candidate
                best_weights = candidate_weights
                best_cost = candidate_cost
                best_objectives = candidate_objectives
                improvement = True
                break
        if improvement:
            delta_max = delta_max * 2.0   
        else:
            delta_max = delta_max * 0.25
        delta_min = 0.5 * delta_max 
        iter_count += 1

    return best_path, best_cost, best_weights, best_objectives


def random_repair(G, removed_indices, path, weights, heuristic):
    """
    Conduct a repair operation on the destroyed with random weight sampling
    """
    if not path:
        print("Original path is empty. Cannot repair.")
        return None, None, None, None

    modified_weights = weights.copy() 
    modified_weights = [sample_log_scale(1, 100000000) for _ in range(len(weights))]
    modified_weights = normalize_weights(modified_weights)

    # Extract start and end indices from the range
    removed_start_idx = removed_indices.start
    removed_end_idx = removed_indices.stop - 1 
    if removed_start_idx <= 0 or removed_end_idx >= len(path) - 1:
        print("Removed indices are out of valid range.")
        path_cost, path_objectives = compute_wm_cost_tie_breaker(path, G, weights) 
        return path, path_cost, modified_weights, path_objectives

    # Identify the nodes before and after the removed segment
    start_repair = path[removed_start_idx - 1]
    end_repair = path[removed_end_idx + 1]

    try:
        # Repair destroyed segment using WS as a subroutine
        new_subpath = weighted_sum_solver(G, start_repair, end_repair, modified_weights, heuristic)

        assert(new_subpath[0] == start_repair)
        assert(new_subpath[-1] == end_repair)

        # If new_subpath is trivial, return the original path.
        if len(new_subpath) == 1:
            path_cost, path_objectives = compute_wm_cost_tie_breaker(path, G, weights)
            return path, path_cost, modified_weights, path_objectives

        repaired_path = path[:removed_start_idx] + new_subpath[1:-1] + path[removed_end_idx + 1:]

        repaired_path_cost, path_objectives = compute_wm_cost_tie_breaker(repaired_path, G, weights) 
        return repaired_path, repaired_path_cost, modified_weights, path_objectives

    except nx.NetworkXNoPath:
        print(f"No path found between {start_repair} and {end_repair} during repair.")
        return None, None, None, None


def lns(G, start_id, goal_id, weights, weight_sampling="random" ,iterations=1000, non_improving_limit=100, three_dim=False):
    """
    LNS framework for solving MOO path planning problems with WM
    """
    # Initialize LNS Parameters
    # Adaptive heuristic selection parameters
    smoothing_window = 50
    window_adjustment_factor = 0.75
    global_improvement_factor = 15
    local_improvement_factor = 3
    accepted_deterioration_factor = 1

    start_time = time.time()
    time_to_best = None

    # Simulated annealing parameters
    start_temp_control = 0.5  # (w% worse solution will have 50% acceptance probability)
    cooling_rate = 0.985
    reheat = False

    # Initialize Adaptive LNS weighting of heuristics
    destroy_heuristic_weights = {
        "random" : 1.0,
        "worst" : 1.0,
        "best" : 1.0,
        "balanced" : 1.0,
        "unbalanced" : 1.0,
    }
    window_rewards = {
        destroy_heuristic: 0.0 for destroy_heuristic in destroy_heuristic_weights
    }

    window_uses = {
        destroy_heuristic: 0 for destroy_heuristic in destroy_heuristic_weights
    }

    # Compute heuristic for WS solver
    if three_dim:
        N = G.number_of_nodes()
        gx, gy, gz = G.nodes[goal_id]['pos']
        heuristic = [0.0] * N 
        for n in G.nodes:
            x, y, z = G.nodes[n]['pos']
            heuristic[n] = weights[0] * math.sqrt(
                (x - gx)**2 + (y - gy)**2 + (z - gz)**2
            )
    else:
        N = G.number_of_nodes()
        gx, gy = G.nodes[goal_id]['pos']
        heuristic = [0.0] * N
        for n in range(N):
            x, y = G.nodes[n]['pos']
            heuristic[n] = weights[0] * math.hypot(x - gx, y - gy)

    # Generate initial solution
    print("Generating Initial Solution")
    # NOTE: could also use weighted_sum_solver(G, start_id, goal_id, weights, heuristic), but beam generally works better
    current_path = beam_weighted_max_solver(G, start_id, goal_id, weights, budget=1, three_dim=three_dim)
    if current_path is None:
        print("ERROR: No solution possible")
        return None, None, None, None
    current_cost, current_objectives = compute_wm_cost_tie_breaker(current_path, G, weights)
    
    time_to_best = time.time() - start_time
    print(f"Initial Solution: {current_cost} (t={time_to_best:.3f}s)")

    # Initialize LNS variables
    num_objectives = len(weights)
    temperature = (start_temp_control * current_cost) / math.log(2)
    start_temp = temperature
    best_path = current_path
    best_cost = current_cost
    best_objectives = current_objectives
    non_improving_iterations = 0
    i = 0

    for i in range(iterations):
        # Determine the solution destruction ratio based on the size of the solution
        if i % 5 == 0:
            destroy_percentage = random.uniform(0.4, 0.95)
        else:
            destroy_percentage = random.uniform(0.05, 0.65)
        destroy_size = int(len(current_path) * destroy_percentage)

        # Destroy operation: Choose a heuristic
        # Build prefix sums of the weighted objectives along current_path before calling destroy operation
        # Save repeated calls to evaluate WM cost on segment
        # NOTE prefix_sums[obj][i] is the weighted‐sum of objective obj up to node index i
        prefix_sums = [[0.0] for _ in range(num_objectives)]
        for u, v in zip(current_path, current_path[1:]):
            edge_obj = G.edges[u, v]['objectives']
            for obj in range(num_objectives):
                # prefix_sums[j][-1] is the cumulative cost up to the previous edge
                prefix_sums[obj].append(
                    prefix_sums[obj][-1]
                    + weights[obj] * edge_obj[obj]
                )

        # Choose destroy operator with a probability proportional to its weight
        destroy_heuristics, destroy_weights = zip(*destroy_heuristic_weights.items())
        choice = random.choices(destroy_heuristics, weights=destroy_weights, k=1)[0]

        if choice == 'random':
            removed_indices = random_removal(current_path, destroy_size)
            if removed_indices is None:
                print(f"Iteration {i}: Random Removal - No vertices removed.")
                continue
        elif choice == 'worst':
            removed_indices = remove_worst_edges(current_path, destroy_size, prefix_sums, num_objectives) 
            if removed_indices is None:
                print(f"Iteration {i}: Worst Removal - No vertices removed.")
                continue
        elif choice == 'best':
            removed_indices = remove_best_edges(current_path, destroy_size, prefix_sums, num_objectives)
            if removed_indices is None:
                print(f"Iteration {i}: Worst Removal - No vertices removed.")
                continue
        elif choice == 'balanced':
            removed_indices = remove_balanced_objective_edges(current_path, destroy_size, prefix_sums, num_objectives)
            if removed_indices is None:
                print(f"Iteration {i}: Worst Removal - No vertices removed.")
                continue
        elif choice == 'unbalanced':
            removed_indices = remove_unbalanced_objective_edges(current_path, destroy_size, prefix_sums, num_objectives)
            if removed_indices is None:
                print(f"Iteration {i}: Worst Removal - No vertices removed.")
                continue

        # Repair operation
        if weight_sampling == "random":
            new_path, new_path_cost, modified_weights, new_objectives = random_repair(G, removed_indices, current_path, weights, heuristic)
        else:
            new_path, new_path_cost, modified_weights, new_objectives = guided_repair(G, removed_indices, current_path, weights, heuristic)

        if new_path is None:
            print(f"Iteration {i}: Repair failed, skipping to next iteration")
            continue  

        # Acceptance criteria
        delta = (new_path_cost - current_cost) / (current_cost + 1e-5)

        heuristic_reward = 0

        if delta < 0:
            heuristic_reward = local_improvement_factor
            accept = True
        elif delta == 0:
            accept = False  # No need to accept the same path
        else:
            try:
                acceptance_probability = math.exp(-delta / temperature)
            except OverflowError:
                continue
            random_probability = random.random()
            accept = random_probability < acceptance_probability

        if accept:
            current_path = new_path
            current_cost = new_path_cost
            current_objectives = new_objectives

            if (current_cost + 1e-5 < best_cost) and delta < 0:
                heuristic_reward = global_improvement_factor
                best_path = current_path
                best_cost = current_cost
                best_objectives = current_objectives
                time_to_best = time.time() - start_time
                print(f"Iteration {i}: Found new best {best_cost} (t={time_to_best:.3f}s)")
            elif delta < 0:
                heuristic_reward = local_improvement_factor
            else:
                heuristic_reward = accepted_deterioration_factor

            non_improving_iterations = 0
            window_uses[choice] += 1
            window_rewards[choice] += heuristic_reward

        else:
            non_improving_iterations += 1

        # Recompute heuristic probabilities for current window
        if i > 0 and i % smoothing_window == 0:
            for destroy_heuristic in destroy_heuristic_weights:
                uses = window_uses[destroy_heuristic]
                reward = window_rewards[destroy_heuristic]
                avg = (reward / uses) if uses > 0 else 0.0

                # Pull the weights toward its segment average
                destroy_heuristic_weights[destroy_heuristic] = (
                    1 - window_adjustment_factor
                ) * destroy_heuristic_weights[destroy_heuristic] + window_adjustment_factor * avg

                # Reset for next window
                window_uses[destroy_heuristic] = 0
                window_rewards[destroy_heuristic] = 0.0

        # Reheat one time to see if we can break out of a deep local minima 
        if non_improving_iterations >= 0.95 * non_improving_limit and reheat == False:
            print(f"Iteration {i}: Reheating ...")
            temperature = start_temp / 2
            reheat = True

        # Check stopping criteria
        if non_improving_iterations >= non_improving_limit:
            print("Stopping LNS: Reached non-improving iteration limit.")
            print(f"Iterations completed: {i}")
            break

        # Only reduce temperature until a small value reached after which stability issues 
        # can occur so just stop reducing it
        if not (temperature < 10e-10):
            temperature = temperature * cooling_rate

    print(f"Iterations completed: {i}")
    return best_path, best_cost, i, destroy_heuristic_weights, time_to_best


if __name__ == "__main__":
    environment_bounds = (500, 500)
    start_node = (25, 25)
    goal_node = (475, 475)
    G, nodes, start_id, goal_id, obstacle_list = generate_indoor_instance_2(environment_bounds, 10, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
    
    # Weights chosen  such that the weighted objective values are approximately equal
    # since that results in a large set of non-dominated paths to explore.
    # Emphasizing a particular objective generally reduces the search space
    weights = [0.012, 11.5, 1]
    normalized_weights = normalize_weights(weights)

    # Run all solvers
    # Run LNS
    start_time = time.time()
    optimal_path_lns, cost_lns, iter_lns, heuristics, time_to_best = lns(
        G, start_id, goal_id, normalized_weights, weight_sampling="guided",
        iterations=75, non_improving_limit=25
    )
    # optimal_path_lns, cost_lns, iter_lns, heuristics, time_to_best = lns( G, start_id, goal_id, normalized_weights, weight_sampling="guided", iterations=150, non_improving_limit=25)
    end_time = time.time()

    # Output results
    if optimal_path_lns:
        cost_lns, all_objectives_lns = compute_wm_cost_tie_breaker(optimal_path_lns, G, normalized_weights)
        print(f"Best Path Found with WM Score: {cost_lns}")
        print("Path:", optimal_path_lns)
        print(f"Cumulative Objectives for Optimal Path: {all_objectives_lns}")
        print("Final Heuristics Weightings: ", heuristics)

    else:
        print("No valid path found.")

    runtime_lns = end_time - start_time
    print(f"LNS Runtime: {runtime_lns:.4f} seconds")

    # Run WM
    print()
    print("Starting WM ...")
    start_wm = time.time()
    G_wm, nodes_wm, _, start_id_wm, goal_id_wm, optimal_path_wm, objectives_wm = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights)
    end_wm = time.time()
    runtime_wm = end_wm - start_wm
    print(f"WM - Cost: {max(objectives_wm):.4f}, Time: {runtime_wm:.4f} seconds")

    # Run Heuristic WM
    print()
    print("Starting WM-Beam ...")
    start_wm = time.time()
    G_wm, nodes_wm, _, start_id_wm, goal_id_wm, optimal_path_h_wm_beam, objectives_h_wm_beam = main_beam_search_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=35)
    end_wm = time.time()
    runtime_h_wm_beam = end_wm - start_wm
    print(f"WM Beam - Cost: {max(objectives_h_wm_beam):.4f}, Time: {runtime_h_wm_beam:.4f} seconds")

    # Run Heuristic WM-Poly
    print()
    print("Starting WM-Poly ...")
    start_wm = time.time()
    G_wm, nodes_wm, _, start_id_wm, goal_id_wm, optimal_path_h_wm_poly, objectives_h_wm_poly = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, normalized_weights, budget=50)
    end_wm = time.time()
    runtime_h_wm_poly = end_wm - start_wm
    print(f"WM Poly - Cost: {max(objectives_h_wm_poly):.4f}, Time: {runtime_h_wm_poly:.4f} seconds")

    # Run WS
    print()
    print("Starting WS ...")
    start_ws = time.time()
    G_ws, _, start_id_ws, goal_id_ws, optimal_path_ws, objectives_ws = main_ws(G, start_id, goal_id, obstacle_list, normalized_weights)
    end_ws = time.time()
    runtime_ws = end_ws - start_ws
    print(f"WS - Cost: {max(objectives_ws):.4f}, Time: {runtime_ws:.4f} seconds")

    # Post Processing of Results
    # Compute final costs on optimal WM and WS with same weights
    cost_wm, all_objectives_wm = compute_wm_cost(optimal_path_wm, G, normalized_weights)
    cost_h_wm_beam, all_objectives_h_wm_beam = compute_wm_cost(optimal_path_h_wm_beam, G, normalized_weights)
    cost_h_wm_poly, all_objectives_h_wm_poly = compute_wm_cost(optimal_path_h_wm_poly, G, normalized_weights)
    cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, normalized_weights)
    cost_lns, all_objectives_lns = compute_wm_cost(optimal_path_lns, G, normalized_weights)
    
    (
        all_objectives_wm_normalized,
        all_objectives_h_wm_beam_normalized,
        all_objectives_h_wm_poly_normalized,
        all_objectives_ws_normalized,
        all_objectives_lns_normalized,
    ) = normalize_solutions(all_objectives_wm, all_objectives_h_wm_beam, all_objectives_h_wm_poly, all_objectives_ws, all_objectives_lns)

    # Compute cost ratios
    h_wm_beam_ratio = cost_h_wm_beam/cost_wm
    h_wm_poly_ratio = cost_h_wm_poly/cost_wm
    ws_ratio = cost_ws_on_wm/cost_wm
    lns_ratio = cost_lns/cost_wm

    all_objectives_wm_unweighted = [all_objectives_wm[i] / normalized_weights[i] for i in range(len(normalized_weights))]
    all_objectives_h_wm_beam_unweighted = [all_objectives_h_wm_beam[i] / normalized_weights[i] for i in range(len(normalized_weights))]
    all_objectives_h_wm_poly_unweighted = [all_objectives_h_wm_poly[i] / normalized_weights[i] for i in range(len(normalized_weights))]
    all_objectives_lns_unweighted = [all_objectives_lns[i] / normalized_weights[i] for i in range(len(normalized_weights))]
    all_objectives_ws_unweighted = [all_objectives_ws[i] / normalized_weights[i] for i in range(len(normalized_weights))]

    (
        all_objectives_wm_unweighted_normalized,
        all_objectives_h_wm_beam_unweighted_normalized,
        all_objectives_h_wm_poly_unweighted_normalized,
        all_objectives_ws_unweighted_normalized,
        all_objectives_lns_unweighted_normalized,
    ) = normalize_solutions(all_objectives_wm_unweighted, all_objectives_h_wm_beam_unweighted, all_objectives_h_wm_poly_unweighted, all_objectives_ws_unweighted, all_objectives_lns_unweighted)

    print("=" * 50)
    print("Summary Stats (Unweighted)")
    print(f"WM Cost: {cost_wm:.5f}")
    print(f"Heuristic WM-Beam Cost: {cost_h_wm_beam:.5f}")
    print(f"Heuristic WM-Poly Cost: {cost_h_wm_poly:.5f}")
    print(f"WS Cost (Evaluated on WM): {cost_ws_on_wm:.5f}")
    print(f"LNS Cost: {cost_lns:.5f}")
    print()
    print(f"WM Objectives: {all_objectives_wm}")
    print(f"Heuristic WM-Beam Objectives: {all_objectives_h_wm_beam}")
    print(f"Heuristic WM-Poly Objectives: {all_objectives_h_wm_poly}")
    print(f"WS Objectives (Evaluated on WM): {all_objectives_ws}")
    print(f"LNS Objectives: {all_objectives_lns}")
    print()
    print(f"WM Objectives (Unweighted): {all_objectives_wm_unweighted}")
    print(f"Heuristic WM-Beam Objectives (Unweighted): {all_objectives_h_wm_beam_unweighted}")
    print(f"Heuristic WM-Poly Objectives (Unweighted): {all_objectives_h_wm_poly_unweighted}")
    print(f"WS Objectives (Evaluated on WM) (Unweighted): {all_objectives_ws_unweighted}")
    print(f"LNS Objectives (Unweighted): {all_objectives_lns_unweighted}")
    print()
    print(f"WM Runtime: {runtime_wm:.5f}")
    print(f"Heuristic WM-Beam Runtime: {runtime_h_wm_beam:.5f}")
    print(f"Heuristic WM-Poly Runtime: {runtime_h_wm_poly:.5f}")
    print(f"WS Runtime: {runtime_ws:.5f}")
    print(f"LNS Runtime: {runtime_lns:.5f}, Time to Best Solution: {time_to_best}")
    print()
    print(f"Heuristic WM-Beam Ratio (Heuristic WM)/(WM) {(h_wm_beam_ratio):.5f}")
    print(f"Heuristic WM-Poly Ratio (Heuristic WM)/(WM) {(h_wm_poly_ratio):.5f}")
    print(f"WS Ratio (WS evaluated on WM)/(WM) {(ws_ratio):.5f}")
    print(f"LNS Ratio (LNS)/(WM) {(lns_ratio):.5f}")

    # Plot the best LNS, WS and WM path
    plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, compare_paths=[optimal_path_wm, optimal_path_ws, optimal_path_lns], risk_zones=True)
    #plot_graph_full(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, compare_paths=[optimal_path_wm, optimal_path_h_wm_poly, optimal_path_h_wm, optimal_path_ws, optimal_path_lns])
