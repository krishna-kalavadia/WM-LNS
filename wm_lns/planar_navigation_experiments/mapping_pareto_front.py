import random
import random
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, json
from wm_lns.environments.generate_environments import *
from wm_lns.utils.plotting_utils import plot_graph_full, plot_graph_n_paths
from wm_lns.utils.common_utils import sample_log_scale, normalize_weights
from wm_lns.wm_lns import lns, compute_wm_cost
from benchmarks.weighted_sum import main_ws
from benchmarks.heuristic_weighted_max import main_heuristic_wm
from benchmarks.beam_search_weighted_max import main_beam_search_wm
from concurrent.futures import ProcessPoolExecutor, as_completed


plt.rcParams.update(
    {
        "font.size": 1,
        "axes.titlesize": 1,
        "axes.labelsize": 34,
        "legend.fontsize": 1,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
    }
)


def worker_sample_wm_beam(environment_params, weights):
    """
    Worker function for WM beam parallel sampling
    """
    G, nodes, start_id, goal_id, obstacle_list = environment_params

    (
        G_wm,
        nodes_wm,
        obstacle_list_wm,
        start_id_wm,
        goal_id_wm,
        optimal_path_wm_beam,
        cost_wm,
    ) = main_beam_search_wm(
        G, nodes, start_id, goal_id, obstacle_list, weights, budget=32 # Budget tuned so LNS and WM have similar runtimes
    )
    cost_wm_computed, all_objectives_wm_beam = compute_wm_cost(
        optimal_path_wm_beam, G, weights
    )

    return optimal_path_wm_beam, all_objectives_wm_beam, weights


def sample_wm_beam_parallel(graph_data, sample_weights, num_samples):
    """
    WM beam parallel sampling for the weights provided
    """
    unique_paths = set()
    unique_path_data = []
    wm_beam_data = []
    wm_beam_merged_data = []
    wm_beam_all_obj_data = []

    G, nodes, start_id, goal_id, obstacle_list = graph_data

    environment_params = (G, nodes, start_id, goal_id, obstacle_list)

    print("Starting parallel WM-Beam sampling...")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(worker_sample_wm_beam, environment_params, w)
            for w in sample_weights[:num_samples]
        ]

        for future in as_completed(futures):
            try:
                optimal_path_beam, all_objectives_wm_beam, weights = future.result()

                path_tuple = tuple(optimal_path_beam)

                all_objectives_wm_beam_unweighted = [
                    all_objectives_wm_beam[i] / weights[i] for i in range(len(weights))
                ]
                # Collect only unique solutions
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)
                    unique_path_data.append(optimal_path_beam)
                    wm_beam_data.append(all_objectives_wm_beam_unweighted)
                    wm_beam_merged_data.append(
                        [all_objectives_wm_beam_unweighted, weights]
                    )

                wm_beam_all_obj_data.append(
                    [all_objectives_wm_beam_unweighted, weights]
                )

            except Exception as e:
                print(f"An error occurred: {e}")

    return wm_beam_data, unique_path_data, wm_beam_merged_data, wm_beam_all_obj_data


def worker_sample_wm_poly(environment_params, weights):
    """
    Worker function for WM poly parallel sampling
    """
    G, nodes, start_id, goal_id, obstacle_list = environment_params

    (
        G_wm,
        nodes_wm,
        obstacle_list_wm,
        start_id_wm,
        goal_id_wm,
        optimal_path_wm_poly,
        cost_wm,
    ) = main_heuristic_wm(
        G, nodes, start_id, goal_id, obstacle_list, weights, budget=35 # Budget tuned so LNS and WM have similar runtimes
    )
    cost_wm_computed, all_objectives_wm_poly = compute_wm_cost(
        optimal_path_wm_poly, G, weights
    )

    return optimal_path_wm_poly, all_objectives_wm_poly, weights


def sample_wm_poly_parallel(graph_data, sample_weights, num_samples):
    """
    WM poly parallel sampling for the weights provided
    """
    unique_paths = set()
    unique_path_data = []
    wm_poly_data = []
    wm_poly_merged_data = []
    wm_poly_all_obj_data = []

    G, nodes, start_id, goal_id, obstacle_list = graph_data

    environment_params = (G, nodes, start_id, goal_id, obstacle_list)

    print("Starting parallel WM-Poly sampling...")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(worker_sample_wm_poly, environment_params, w)
            for w in sample_weights[:num_samples]
        ]

        for future in as_completed(futures):
            try:
                optimal_path, all_objectives_wm_poly, weights = future.result()

                path_tuple = tuple(optimal_path)

                all_objectives_wm_poly_unweighted = [
                    all_objectives_wm_poly[i] / weights[i] for i in range(len(weights))
                ]
                # Collect only unique solutions
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)
                    unique_path_data.append(optimal_path)
                    wm_poly_data.append(all_objectives_wm_poly_unweighted)
                    wm_poly_merged_data.append(
                        [all_objectives_wm_poly_unweighted, weights]
                    )

                wm_poly_all_obj_data.append(
                    [all_objectives_wm_poly_unweighted, weights]
                )

            except Exception as e:
                print(f"An error occurred: {e}")

    return wm_poly_data, unique_path_data, wm_poly_merged_data, wm_poly_all_obj_data


def worker_sample_wm(environment_params, weights):
    """
    Worker function for WM parallel sampling
    """
    G, nodes, start_id, goal_id, obstacle_list = environment_params

    (
        G_wm,
        nodes_wm,
        obstacle_list_wm,
        start_id_wm,
        goal_id_wm,
        optimal_path_wm,
        cost_wm,
    ) = main_heuristic_wm(G, nodes, start_id, goal_id, obstacle_list, weights)
    cost_wm_computed, all_objectives_wm = compute_wm_cost(optimal_path_wm, G, weights)

    return optimal_path_wm, all_objectives_wm, weights


def sample_wm_parallel(graph_data, sample_weights, num_samples):
    """
    WM parallel sampling for the weights provided
    """
    unique_paths = set()
    unique_path_data = []
    wm_data = []
    wm_merged_data = []
    wm_all_obj_data = []

    G, nodes, start_id, goal_id, obstacle_list = graph_data

    environment_params = (G, nodes, start_id, goal_id, obstacle_list)

    print("Starting parallel WM sampling...")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(worker_sample_wm, environment_params, w)
            for w in sample_weights[:num_samples]
        ]

        for future in as_completed(futures):
            try:
                optimal_path, all_objectives_wm, weights = future.result()

                path_tuple = tuple(optimal_path)

                all_objectives_wm_unweighted = [
                    all_objectives_wm[i] / weights[i] for i in range(len(weights))
                ]
                # Collect only unique solutions
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)
                    unique_path_data.append(optimal_path)
                    wm_data.append(all_objectives_wm_unweighted)
                    wm_merged_data.append([all_objectives_wm_unweighted, weights])

                wm_all_obj_data.append([all_objectives_wm_unweighted, weights])

            except Exception as e:
                print(f"An error occurred: {e}")

    return wm_data, unique_path_data, wm_merged_data, wm_all_obj_data


def worker_sample_lns(environment_params, weights):
    """
    Worker function for WM parallel sampling
    """
    G, nodes, start_id, goal_id, obstacle_list = environment_params

    # optimal_path_lns, cost_lns, _, _, _ = lns(
    #     G, start_id, goal_id, weights, weight_sampling="guided",
    #     iterations=25, non_improving_limit=25
    # )

    optimal_path_lns, cost_lns, _, _, _ = lns(
        G,
        start_id,
        goal_id,
        weights,
        weight_sampling="random",
        iterations=400,
        non_improving_limit=50,
    )

    cost_lns_computed, all_objectives_lns = compute_wm_cost(
        optimal_path_lns, G, weights
    )

    return optimal_path_lns, all_objectives_lns, weights


def sample_lns_parallel(graph_data, sample_weights, num_samples):
    """
    LNS parallel sampling for the weights provided
    """
    unique_paths = set()
    unique_path_data = []
    lns_data = []
    lns_merged_data = []
    lns_all_obj_data = []

    G, nodes, start_id, goal_id, obstacle_list = graph_data

    environment_params = (G, nodes, start_id, goal_id, obstacle_list)

    print("Starting parallel LNS sampling...")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(worker_sample_lns, environment_params, w)
            for w in sample_weights[:num_samples]
        ]

        for future in as_completed(futures):
            try:
                optimal_path, all_objectives_lns, weights = future.result()

                path_tuple = tuple(optimal_path)

                all_objectives_lns_unweighted = [
                    all_objectives_lns[i] / weights[i] for i in range(len(weights))
                ]
                # Collect only unique solutions
                if path_tuple not in unique_paths:
                    unique_paths.add(path_tuple)
                    unique_path_data.append(optimal_path)
                    lns_data.append(all_objectives_lns_unweighted)
                    lns_merged_data.append([all_objectives_lns_unweighted, weights])

                lns_all_obj_data.append([all_objectives_lns_unweighted, weights])

            except Exception as e:
                print(f"An error occurred: {e}")

    return lns_data, unique_path_data, lns_merged_data, lns_all_obj_data


def sample_ws(graph_data, sample_weights, samples):
    """
    WS sampling
    """
    ws_data = []
    unique_paths = set()
    unique_path_data = []
    ws_weight_data = []
    ws_merged_data = []
    ws_all_obj_data = []

    G, nodes, start_id, goal_id, obstacle_list = graph_data

    print("Starting sampling ... ")
    for i in range(samples):
        weights = sample_weights[i]

        G_ws, obstacle_list_ws, start_id_ws, goal_id_ws, optimal_path_ws, cost_ws = (
            main_ws(G, start_id, goal_id, obstacle_list, weights)
        )

        path_tuple = tuple(optimal_path_ws)

        cost_ws_on_wm, all_objectives_ws = compute_wm_cost(optimal_path_ws, G, weights)
        all_objectives_ws_unweighted = [
            all_objectives_ws[i] / weights[i] for i in range(len(weights))
        ]

        # Collect only unique solutions
        if path_tuple not in unique_paths:
            unique_paths.add(path_tuple)
            ws_data.append(all_objectives_ws_unweighted)
            unique_path_data.append(optimal_path_ws)
            ws_weight_data.append(weights)
            ws_merged_data.append([all_objectives_ws_unweighted, weights])

        ws_all_obj_data.append([all_objectives_ws_unweighted, weights])

    return ws_data, unique_path_data, ws_merged_data, ws_all_obj_data


def plot_pareto_front_main(
    graph_data,
    samples,
    complete_ws=True,
    complete_wm=True,
    complete_wm_poly=True,
    complete_wm_beam=True,
    complete_lns=True,
):
    G, nodes, start_id, goal_id, obstacle_list = graph_data

    # WS DATA
    ws_data = []
    paths_ws = []
    ws_merged = []
    ws_all = []

    # WM DATA
    wm_data = []
    paths_wm = []
    wm_merged = []
    wm_all = []

    # WM-Poly DATA
    wm_poly_data = []
    paths_wm_poly = []
    wm_poly_merged = []
    wm_poly_all = []

    # WM-Beam DATA
    wm_beam_data = []
    paths_wm_beam = []
    wm_beam_merged = []
    wm_beam_all = []

    # LNS DATA
    lns_data = []
    paths_lns = []
    lns_merged = []
    lns_all = []

    # Generate weight samples
    sample_weights = []
    for i in range(samples):
        weights = [100, sample_log_scale(1, 100000000)]  # Use for 2 objectives
        # weights = [100, sample_log_scale(1, 100000000), sample_log_scale(1, 100000000)] # Use for 3 objectives
        weights = normalize_weights(weights)
        sample_weights.append(weights)

    save_weight_samples(
        sample_weights,
        filepath="wm_lns/planar_navigation_experiments/mapping_pf_data/sampled_weights.csv",
    )
    # To load previously used weights 
    # sample_weights = load_weight_samples("wm_lns/planar_navigation_experiments/mapping_pf_data/sampled_weights.csv")

    if complete_ws:
        start_time = time.time()
        ws_data, paths_ws, ws_merged, ws_all = sample_ws(
            graph_data, sample_weights, samples
        )
        end_time = time.time()
        ws_time = end_time - start_time
        save_method_data_csv("ws", ws_data, paths_ws)
    if complete_wm:
        start_time = time.time()
        wm_data, paths_wm, wm_merged, wm_all = sample_wm_parallel(
            graph_data, sample_weights, samples
        )
        end_time = time.time()
        wm_time = end_time - start_time
        save_method_data_csv("wm", wm_data, paths_wm)
    if complete_wm_poly:
        start_time = time.time()
        wm_poly_data, paths_wm_poly, wm_poly_merged, wm_poly_all = (
            sample_wm_poly_parallel(graph_data, sample_weights, samples)
        )
        end_time = time.time()
        wm_poly_time = end_time - start_time
        save_method_data_csv("wm-poly", wm_poly_data, paths_wm_poly)
    if complete_wm_beam:
        start_time = time.time()
        wm_beam_data, paths_wm_beam, wm_beam_merged, wm_beam_all = (
            sample_wm_beam_parallel(graph_data, sample_weights, samples)
        )
        end_time = time.time()
        wm_beam_time = end_time - start_time
        save_method_data_csv("wm-beam", wm_beam_data, paths_wm_beam)
    if complete_lns:
        start_time = time.time()
        lns_data, paths_lns, lns_merged, lns_all = sample_lns_parallel(
            graph_data, sample_weights, samples
        )
        end_time = time.time()
        lns_time = end_time - start_time
        save_method_data_csv("lns", lns_data, paths_lns)

    if complete_ws:
        print("WS PF Mapping Time: ", ws_time)
    if complete_wm:
        print("WM PF Mapping Time: ", wm_time)
    if complete_wm_poly:
        print("WM Poly PF Mapping Time: ", wm_poly_time)
    if complete_wm_beam:
        print("WM Beam PF Mapping Time: ", wm_beam_time)
    if complete_lns:
        print("LNS PF Mapping Time: ", lns_time)

    return (
        ws_data,
        paths_ws,
        ws_merged,
        ws_all,
        wm_data,
        paths_wm,
        wm_merged,
        wm_all,
        wm_poly_data,
        paths_wm_poly,
        wm_poly_merged,
        wm_poly_all,
        wm_beam_data,
        paths_wm_beam,
        wm_beam_merged,
        wm_beam_all,
        lns_data,
        paths_lns,
        lns_merged,
        lns_all,
    )


def save_weight_samples(
    weights_list,
    filepath="wm_lns/planar_navigation_experiments/mapping_pf_data/sampled_weights.csv",
):
    arr = np.asarray(weights_list, dtype=float)
    df = pd.DataFrame(arr, columns=[f"w{i}" for i in range(arr.shape[1])])

    make_dir(os.path.dirname(filepath))

    df.to_csv(filepath, index=False, mode="w")
    print(f"Saved {len(df)} weight rows to {filepath}")


def load_weight_samples(
    filepath="wm_lns/planar_navigation_experiments/mapping_pf_data/sampled_weights.csv",
):
    df = pd.read_csv(filepath)
    wcols = [c for c in df.columns if c.lower().startswith("w")]
    if not wcols:
        wcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not wcols:
        raise ValueError(f"No numeric weight columns found in {filepath}")
    weights = df[wcols].to_numpy(dtype=float).tolist()
    print(f"Loaded {len(weights)} weight rows from {filepath}")
    return weights


def make_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_method_data_csv(
    method, objectives, paths, outdir="mapping_pf_data", filename=None
):
    """
    Save one method's objectives and paths
    """
    if len(objectives) != len(paths):
        raise ValueError("objectives and paths must have the same length.")

    make_dir(outdir)
    if filename is None:
        filename = f"{method}_results.csv"
    filepath = os.path.join(outdir, filename)

    rows = []
    for i, obj in enumerate(objectives):
        row = {"id": i}
        for j, val in enumerate(obj):
            row[f"o{j}"] = float(val)
        row["path_json"] = json.dumps(paths[i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(filepath, index=False, float_format="%.10g")
    print(f"[{method}] Saved {len(rows)} rows to {filepath}")
    return filepath


def load_method_data_csv(method, outdir="mapping_pf_data", filename=None):
    """
    Load objectives and paths
    """
    if filename is None:
        filename = f"{method}_results.csv"
    filepath = os.path.join(outdir, filename)

    df = pd.read_csv(filepath)
    # Objective val columns are o0, o1, o3 .. etc
    ocols = sorted(
        [c for c in df.columns if c.startswith("o") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )

    objectives = df[ocols].to_numpy(float).tolist() if ocols else []
    paths = [json.loads(s) for s in df["path_json"]]
    print(f"[{method}] Loaded {len(objectives)} rows from {filepath}")
    return objectives, paths


def normalize_objectives_per_axis(data):
    """
    Normalize objectives per each individual objective
    """
    if not data:
        return []

    A = np.asarray(data, dtype=float)
    mins = A.min(axis=0)
    maxs = A.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)  #

    return ((A - mins) / denom).tolist()


def find_global_minmax(*sets):
    """
    Find the min values and max values across each set per each axis in each set
    """
    arrays = [np.asarray(s, float) for s in sets if len(s)]
    if not arrays:
        return None, None

    all_pts = np.vstack(arrays)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    return mins, maxs


def transform_with_minmax(points, mins, maxs):
    """
    Given the global min and maxs normalize all the provided points
    """
    if mins is None:
        return points
    A = np.asarray(points, float)

    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return ((A - mins) / denom).tolist()


def dominates(path1_objectives, path2_objectives, eps=1e-8):
    """
    Check if path1's objectives dominates path2's objectives.

    Note: path1 dominates path2 if all are path1's objectives are better than or equal to path2
            and there is at least one objective in path one that is better
    """
    if len(path1_objectives) != len(path2_objectives):
        raise ValueError("Objective vectors must have same length.")

    better_in_at_least_one = False
    for a, b in zip(path1_objectives, path2_objectives):
        if a > b + eps:  
            return False
        if a < b - eps:  
            better_in_at_least_one = True
    return better_in_at_least_one


def unique_rows_round(points, decimals=8):
    """
    Get rid of duplicate objective vectors after rounding
    """
    A = np.asarray(points, float)
    if len(A) == 0:
        return []
    B = np.round(A, decimals)
    _, idx = np.unique(B, axis=0, return_index=True)
    return A[idx].tolist()


def find_pareto_front(objectives_solutions, eps=1e-8, dedup_decimals=8):
    """
    Return non-dominated solutions
    """
    X = unique_rows_round(objectives_solutions, dedup_decimals)
    pareto_front = []
    dominated_solutions = []

    for i, candidate in enumerate(X):
        dominated = False
        for j, other in enumerate(X):
            if i == j:
                continue
            if dominates(other, candidate, eps=eps):
                dominated = True
                dominated_solutions.append(candidate)
                break
        if not dominated:
            pareto_front.append(candidate)

    return pareto_front, dominated_solutions


def find_pareto_front_wm(objectives_solutions, wm_solutions, eps=1e-8, decimals=8):
    """
    Return sol in X that are not dominated by the WM set (true set)
    """
    X = unique_rows_round(objectives_solutions, decimals)
    WM = unique_rows_round(wm_solutions, decimals)

    wm_pf, _ = find_pareto_front(WM, eps=eps, dedup_decimals=decimals)
    X_survivors = [c for c in X if not any(dominates(w, c, eps=eps) for w in wm_pf)]
    X_survivors_pf, _ = find_pareto_front(X_survivors, eps=eps, dedup_decimals=decimals)
    dominated_by_wm = [c for c in X if any(dominates(w, c, eps=eps) for w in wm_pf)]

    return X_survivors_pf, dominated_by_wm


def build_pf_colors(objs, paths, scalar_axis=0, cmap_name="viridis"):
    A = np.asarray(objs, float)
    scalars = A[:, scalar_axis]
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(float(s)) for s in scalars]
    return {"points": A.tolist(), "paths": paths, "colors": colors}


def plot_pareto_front(
    ws_objectives,
    wm_objectives,
    wm_poly_objectives,
    wm_beam_objectives,
    lns_objectives,
    pareto_front,
    dominated_solutions,
    highlight_point=None,
    pf_colors=None,
):
    plt.figure(figsize=(10, 7))

    # WM
    if len(wm_objectives) > 0:
        wm_lengths = [obj[0] for obj in wm_objectives]
        wm_obstacles = [obj[1] for obj in wm_objectives]
        plt.scatter(
            wm_lengths,
            wm_obstacles,
            s=120,
            alpha=0.9,
            label="WM Solutions",
            c=(pf_colors if pf_colors is not None else "blue"),
        )

    # WM-Poly
    if len(wm_poly_objectives) > 0:
        wm_poly_lengths = [obj[0] for obj in wm_poly_objectives]
        wm_poly_obstacles = [obj[1] for obj in wm_poly_objectives]
        plt.scatter(
            wm_poly_lengths,
            wm_poly_obstacles,
            s=120,
            alpha=0.9,
            label="WM-Poly Solutions",
            c=(pf_colors if pf_colors is not None else "blue"),
        )

    # WM-Beam
    if len(wm_beam_objectives) > 0:
        wm_beam_lengths = [obj[0] for obj in wm_beam_objectives]
        wm_beam_obstacles = [obj[1] for obj in wm_beam_objectives]
        plt.scatter(
            wm_beam_lengths,
            wm_beam_obstacles,
            s=120,
            alpha=0.9,
            label="WM-Beam Solutions",
            c=(pf_colors if pf_colors is not None else "blue"),
        )

    # LNS
    if len(lns_objectives) > 0:
        lns_lengths = [obj[0] for obj in lns_objectives]
        lns_obstacles = [obj[1] for obj in lns_objectives]
        plt.scatter(
            lns_lengths,
            lns_obstacles,
            s=120,
            alpha=0.9,
            label="WM-LNS Solutions",
            c=(pf_colors if pf_colors is not None else "blue"),
        )

    # WS
    if len(ws_objectives) > 0:
        ws_lengths = [obj[0] for obj in ws_objectives]
        ws_obstacles = [obj[1] for obj in ws_objectives]
        plt.scatter(
            ws_lengths,
            ws_obstacles,
            s=120,
            alpha=0.9,
            label="WS Solutions",
            c=(pf_colors if pf_colors is not None else "blue"),
        )

    # Dominated solutions
    if len(dominated_solutions) > 0:
        sol_lengths = [obj[0] for obj in dominated_solutions]
        sol_obstacles = [obj[1] for obj in dominated_solutions]
        plt.scatter(
            sol_lengths, sol_obstacles, label="Dominated Solutions", color="blue"
        )

    # Highlight a specific point on the PF
    if highlight_point is not None:
        hx, hy = highlight_point
        plt.scatter(
            [hx],
            [hy],
            s=250,
            marker="*",
            facecolors="gold",
            edgecolors="black",
            linewidths=1.75,
            label="Desired Solution",
        )

    # Pareto front line
    if pareto_front:
        pf_x = [obj[0] for obj in pareto_front]
        pf_y = [obj[1] for obj in pareto_front]
        pf_x, pf_y = zip(*sorted(zip(pf_x, pf_y), key=lambda t: t[0]))
        plt.plot(
            pf_x,
            pf_y,
            color="blue",
            linestyle="--",
            linewidth=0.25,
            label="Pareto Front Line",
        )

    plt.xlabel("Path Length")
    plt.ylabel("Obs. Closeness")
    # plt.legend()
    plt.grid(True)
    plt.show()


def estimate_coverage(points, n_samples=100_000, seed=None):
    """
    Compute fraction of points in our objective space dominated by solution set
    """
    P = np.asarray(points, dtype=float)
    d = P.shape[1]
    # print("Sampling hypercube of volume: ", d)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    U = rng.random((n_samples, d))
    dominated = (
        (U[:, None, :] >= P[None, :, :]).all(axis=2)
        & (U[:, None, :] > P[None, :, :]).any(axis=2)
    ).any(axis=1)

    return float(dominated.mean())


if __name__ == "__main__":
    environment_bounds = (500, 500)
    start_node = (200, 25)
    goal_node = (475, 300)
    G, nodes, start_id, goal_id, obstacle_list = generate_indoor_pf_map(
        environment_bounds,
        11.5,
        5000,
        0.075 * min(environment_bounds),
        start_node,
        goal_node,
    )
    # G, nodes, start_id, goal_id, obstacle_list = generate_indoor_pf_map_from_nodes(environment_bounds, 10, 5000, 0.075 * min(environment_bounds), start_node, goal_node, import_nodes_from_csv(filename="mapping_pf_data/saved_pf_mapping.csv"))

    # G, nodes, start_id, goal_id, obstacle_list = generate_indoor_pf_map_with_risk(environment_bounds, 11.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node)
    # G, nodes, start_id, goal_id, obstacle_list = generate_indoor_pf_map_with_risk_from_nodes(environment_bounds, 11.5, 5000, 0.075 * min(environment_bounds), start_node, goal_node, import_nodes_from_csv(filename="mapping_pf_data/saved_pf_mapping.csv"))

    graph_data = [G, nodes, start_id, goal_id, obstacle_list]

    ws_data = []
    ws_merged = []
    paths_ws = []
    ws_all = []

    wm_data = []
    wm_merged = []
    paths_wm = []
    wm_all = []

    wm_poly_data = []
    wm_poly_merged = []
    paths_wm_poly = []
    wm_poly_all = []

    wm_beam_data = []
    wm_beam_merged = []
    paths_wm_beam = []
    wm_beam_all = []

    lns_data = []
    lns_merged = []
    paths_lns = []
    lns_all = []

    (
        ws_data,
        paths_ws,
        ws_merged,
        ws_all,
        wm_data,
        paths_wm,
        wm_merged,
        wm_all,
        wm_poly_data,
        paths_wm_poly,
        wm_poly_merged,
        wm_poly_all,
        wm_beam_data,
        paths_wm_beam,
        wm_beam_merged,
        wm_beam_all,
        lns_data,
        paths_lns,
        lns_merged,
        lns_all,
    ) = plot_pareto_front_main(graph_data, 20, complete_lns=True)

    # if you want to load data instead run a new experiment
    # ws_data, paths_ws = load_method_data_csv("ws")
    # wm_data, paths_wm = load_method_data_csv("wm")
    # wm_poly_data, paths_wm_poly = load_method_data_csv("wm-poly")
    # wm_beam_data, paths_wm_beam = load_method_data_csv("wm-beam")
    # lns_data, paths_lns = load_method_data_csv("lns")

    # Normalize
    mins, maxs = find_global_minmax(
        ws_data, wm_data, wm_poly_data, wm_beam_data, lns_data
    )

    ws_data = transform_with_minmax(ws_data, mins, maxs)
    wm_data = transform_with_minmax(wm_data, mins, maxs)
    wm_poly_data = transform_with_minmax(wm_poly_data, mins, maxs)
    wm_beam_data = transform_with_minmax(wm_beam_data, mins, maxs)
    lns_data = transform_with_minmax(lns_data, mins, maxs)

    pareto_front_ws, _ = find_pareto_front_wm(ws_data, wm_data)
    pareto_front_wm, _ = find_pareto_front_wm(wm_data, wm_data)
    pareto_front_wm_poly, _ = find_pareto_front_wm(wm_poly_data, wm_data)
    pareto_front_wm_beam, _ = find_pareto_front_wm(wm_beam_data, wm_data)
    pareto_front_lns, _ = find_pareto_front_wm(lns_data, wm_data)

    print("Unique WS solutions: ", len(pareto_front_ws))
    print("Unique WM solutions: ", len(pareto_front_wm))
    print("Unique WM-Poly solutions: ", len(pareto_front_wm_poly))
    print("Unique WM-Beam solutions: ", len(pareto_front_wm_beam))
    print("Unique LNS solutions: ", len(pareto_front_lns))

    print(
        "Coverage WS solutions: ",
        estimate_coverage(pareto_front_ws, n_samples=5000000, seed=42),
    )
    print(
        "Coverage WM solutions: ",
        estimate_coverage(pareto_front_wm, n_samples=5000000, seed=42),
    )
    print(
        "Coverage WM-Poly solutions: ",
        estimate_coverage(pareto_front_wm_poly, n_samples=5000000, seed=42),
    )
    print(
        "Coverage WM-Beam solutions: ",
        estimate_coverage(pareto_front_wm_beam, n_samples=5000000, seed=42),
    )
    print(
        "Coverage LNS solutions: ",
        estimate_coverage(pareto_front_lns, n_samples=5000000, seed=42),
    )

    ws_data_with_colors = (
        build_pf_colors(ws_data, paths_ws, scalar_axis=0, cmap_name="viridis")
        if ws_data
        else None
    )
    wm_data_with_colors = (
        build_pf_colors(wm_data, paths_wm, scalar_axis=0, cmap_name="viridis")
        if wm_data
        else None
    )
    wm_poly_data_with_colors = (
        build_pf_colors(wm_poly_data, paths_wm_poly, scalar_axis=0, cmap_name="viridis")
        if wm_poly_data
        else None
    )
    wm_beam_data_with_colors = (
        build_pf_colors(wm_beam_data, paths_wm_beam, scalar_axis=0, cmap_name="viridis")
        if wm_beam_data
        else None
    )
    lns_data_with_colors = (
        build_pf_colors(lns_data, paths_lns, scalar_axis=0, cmap_name="viridis")
        if lns_data
        else None
    )

    if ws_data:
        plot_pareto_front(
            ws_data,
            [],
            [],
            [],
            [],
            pareto_front_ws,
            [],
            pf_colors=ws_data_with_colors["colors"],
        )
    if wm_data:
        plot_pareto_front(
            [],
            wm_data,
            [],
            [],
            [],
            pareto_front_wm,
            [],
            pf_colors=wm_data_with_colors["colors"],
        )
    if wm_poly_data:
        plot_pareto_front(
            [],
            [],
            wm_poly_data,
            [],
            [],
            pareto_front_wm_poly,
            [],
            pf_colors=wm_poly_data_with_colors["colors"],
        )
    if wm_beam_data:
        plot_pareto_front(
            [],
            [],
            [],
            wm_beam_data,
            [],
            pareto_front_wm_beam,
            [],
            pf_colors=wm_beam_data_with_colors["colors"],
        )
    if lns_data:
        plot_pareto_front(
            [],
            [],
            [],
            [],
            lns_data,
            pareto_front_lns,
            [],
            pf_colors=lns_data_with_colors["colors"],
        )

    if ws_data_with_colors and ws_data_with_colors["paths"]:
        plot_graph_n_paths(
            G,
            nodes,
            obstacle_list,
            start_id,
            goal_id,
            environment_bounds,
            paths=ws_data_with_colors["paths"],
            path_colors=ws_data_with_colors["colors"],
        )

    if wm_data_with_colors and wm_data_with_colors["paths"]:
        plot_graph_n_paths(
            G,
            nodes,
            obstacle_list,
            start_id,
            goal_id,
            environment_bounds,
            paths=wm_data_with_colors["paths"],
            path_colors=wm_data_with_colors["colors"],
        )

    if wm_poly_data_with_colors and wm_poly_data_with_colors["paths"]:
        plot_graph_n_paths(
            G,
            nodes,
            obstacle_list,
            start_id,
            goal_id,
            environment_bounds,
            paths=wm_poly_data_with_colors["paths"],
            path_colors=wm_poly_data_with_colors["colors"],
        )

    if wm_beam_data_with_colors and wm_beam_data_with_colors["paths"]:
        plot_graph_n_paths(
            G,
            nodes,
            obstacle_list,
            start_id,
            goal_id,
            environment_bounds,
            paths=wm_beam_data_with_colors["paths"],
            path_colors=wm_beam_data_with_colors["colors"],
        )

    if lns_data_with_colors and lns_data_with_colors["paths"]:
        plot_graph_n_paths(
            G,
            nodes,
            obstacle_list,
            start_id,
            goal_id,
            environment_bounds,
            paths=lns_data_with_colors["paths"],
            path_colors=lns_data_with_colors["colors"],
        )

    # plot_graph_n_paths(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, paths_ws)
    # plot_graph_n_paths(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, paths_wm)
    # plot_graph_n_paths(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, paths_wm_poly)
    # plot_graph_n_paths(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, paths_wm_beam)
    # plot_graph_n_paths(G, nodes, obstacle_list, start_id, goal_id, environment_bounds, paths_lns)

    export_nodes_to_csv(
        nodes,
        filename="wm_lns/planar_navigation_experiments/mapping_pf_data/saved_pf_mapping.csv",
    )
