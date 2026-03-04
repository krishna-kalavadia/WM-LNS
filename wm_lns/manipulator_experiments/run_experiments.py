"""Run Franka Kitchen Experiments"""
# NOTE: code to generate the environment is from https://github.com/kwonathan/franka-kitchen-pybullet

import time
import math
import numpy as np
import networkx as nx
import pybullet as p
import pybullet_data

import wm_lns.manipulator_experiments.config as config
from wm_lns.manipulator_experiments.env import Environment
from wm_lns.manipulator_experiments.robot import Robot
from benchmarks.heuristic_weighted_max import heuristic_weighted_max_solver
from benchmarks.beam_search_weighted_max import beam_weighted_max_solver
from benchmarks.weighted_sum import main_ws
from wm_lns.wm_lns import lns
import random
import pandas as pd
import pickle

# Robot Constants
END_LINK = 11

# PRM Constants
NUM_SAMPLES = 7500
NUM_NEIGHBORS = 25
COLLISION_DIST = 0.0125

# Local Planner Constants
NUM_INTERMEDIATE_POINTS = 25

# Execution Parameters
EXECUTED_INTERMEDIATE_POINTS = 25
STEPS = 25
TIME_SPEED_FACTOR = 0.05

# Control Parameters
POSITION_GAIN = 0.025
VELOCITY_GAIN = 1.5
MAX_VELOCITY = 0.75

# Colors 
RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]


def compute_optimality_error(wm_objectives, sol_objectives):
    """
    Compute the average percentage that each objective is off by
    We take the percentage error of the all objectives that perform worse and then take the average percentage error
    """
    if len(wm_objectives) != len(sol_objectives):
        raise ValueError("Vectors must have the same length.")

    total_pct = 0.0
    for wm_val, sol_val in zip(wm_objectives, sol_objectives):
        if sol_val > wm_val:
            overshoot = sol_val - wm_val
            total_pct += (overshoot / wm_val) * 100.0      

    return total_pct


def normalize_solutions(*vecs):
    """
    Normalized provided vectors using min max normalization
    """
    M = np.asarray(vecs, dtype=float)          

    mins   = np.min(M, axis=0)                 
    maxs   = np.max(M, axis=0)                 
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0                 

    M_norm = (M - mins) / ranges
    return [row.tolist() for row in M_norm]


def get_arm_joints_and_limits(robot_id):
    """
    Get joint data for provided robot and return:
        joint_indices: list of joint indices
        lower_limits: list of lower limits, where index is joint index
        upper_limits: list of upper limits, where index is joint index
        joint_ranges: list of travel range, where index is joint index
    """

    joint_indices = []
    lower_limits = []
    upper_limits = []

    # Iterate though robot's joints
    for i in range(p.getNumJoints(robot_id)):
        # Extract joint info
        joint_info = p.getJointInfo(robot_id, i)

        # Collect data for the joint
        joint_indices.append(i)
        lower_limits.append(joint_info[8])
        upper_limits.append(joint_info[9])
    
    # Create joint range intervals for each joint
    joint_ranges = []
    for lower, upper in zip(lower_limits, upper_limits):
        joint_range = upper - lower
        if joint_range > 0:
            joint_ranges.append(joint_range)
        else:
            joint_ranges.append(2 * math.pi)
    
    # Return data needed
    return joint_indices[:7], lower_limits[:7], upper_limits[:7], joint_ranges[:7]


def is_point_in_aabb(point, aabb):
    """
    Check if specified point lies inside specified aabb
    """
    (min_corner, max_corner) = aabb
    return all(min_corner[i] <= point[i] <= max_corner[i] for i in range(3))


def get_microwave_aabbs(kitchen):
    """
    Get all aabbs that define the microwave
    """
    aabbs = []
    for link in range(p.getNumJoints(kitchen)):
        info = p.getJointInfo(kitchen, link)
        name = info[12]

        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8")
            except:
                name = str(name)

        if "microwave" in name.lower():
            aabb = p.getAABB(kitchen, link)
            aabbs.append(aabb)
    return aabbs


def get_stove_center(kitchen):
    """
    Get center coordinate for the stove
    """
    centers = []

    for link in range(p.getNumJoints(kitchen)):
        info = p.getJointInfo(kitchen, link)
        name = info[12]
    
        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8")
            except:
                name = str(name)

        if any(k in name.lower() for k in ("stove", "cooktop", "burner")):
            aabb = p.getAABB(kitchen, link)
            mn, mx = aabb
            centers.append([(1/2) * (mn[i] + mx[i]) for i in range(3)])
    
    # If multiple matches
    if centers:
        return np.mean(np.array(centers), axis=0)


def is_node_valid(node_coordinates, joint_angles, joint_indices, joint_lower_limits, joint_upper_limits, robot_id, obstacles, microwave_aabbs):
    """
    Check if the provided node is valid
    """
    # Check if our coordinates are inside microwave
    if any(is_point_in_aabb(node_coordinates, aabb) for aabb in microwave_aabbs):
        return False
    
    # First check if our joint angles respect our limits
    for angle, lower_limit, upper_limit in zip(joint_angles, joint_lower_limits, joint_upper_limits):
        if angle < lower_limit or angle > upper_limit:
            return False
        
    # Save our state since we will return to this after collision checking
    try:
        prev_states = [p.getJointState(robot_id, i)[0] for i in joint_indices]
    except Exception:
        return False

    # Teleport our robot to the joint angles specified
    for angle, i in zip(joint_angles, joint_indices):
        p.resetJointState(robot_id, i, angle)
    
    # Check for collisions
    p.performCollisionDetection()

    collision = False
    for body in obstacles:
        if p.getContactPoints(bodyA=robot_id, bodyB=body):
            collision = True
            break
        if p.getClosestPoints(bodyA=robot_id, bodyB=body, distance=COLLISION_DIST):
            collision = True
            break
    
    # Check for self collisions
    pts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if pts:
        for c in pts:
            linkA, linkB = c[3], c[4]
            if linkA != linkB:         
                collision = True

    # Restore our robot to prev_states list
    for prev_angle, i in zip(prev_states, joint_indices):
        p.resetJointState(robot_id, i, prev_angle)
    
    if collision:
        return False

    return True


def local_planner(robot_id, joint_indices, joint_lower_limits, joint_upper_limits, joint_ranges, point1, point2, orientation, obstacles, microwave_aabbs):
    """
    Plan a straight line path between point1 and point2, representing an edge in the PRM
    """
    rest_pose = config.jointStartPositions[:7]
    last_ik = None
    traj = []

    for delta in np.linspace(0, 1, NUM_INTERMEDIATE_POINTS):
        target_point = ((1 - delta) * point1 + delta * point2).tolist()
        target_point_ik = p.calculateInverseKinematics(
            robot_id,
            END_LINK,
            targetPosition=target_point,
            targetOrientation=orientation,
            lowerLimits=joint_lower_limits,
            upperLimits=joint_upper_limits,
            jointRanges=joint_ranges,
            restPoses=last_ik if last_ik is not None else rest_pose,
            maxNumIterations=400,
            residualThreshold=1e-6,
        )[:7]

        if not is_node_valid(target_point, target_point_ik, joint_indices, joint_lower_limits, joint_upper_limits, robot_id, obstacles, microwave_aabbs):
            return None
        last_ik = target_point_ik
        traj.append(target_point_ik)
    
    return traj


def build_prm(robot_id, environment):
    """
    Builds a PRM in task space (3D euclidean space)
    """
    # First get our joint limits
    joint_indices, joint_lower_limits, joint_upper_limits, joint_ranges = get_arm_joints_and_limits(robot_id)

    # Define obstacles to avoid, explicitly define microwave to prevent sampling inside its cavity
    obstacles = [environment.kitchen, environment.kettle]
    microwave_aabbs = get_microwave_aabbs(environment.kitchen)
    stove_center = get_stove_center(environment.kitchen)

    # Define start location
    start_point = np.array(config.startPosition)

    # Define goal location
    goal_point = np.array(config.goalPosition)

    # Define down orientation
    down_orientation = p.getQuaternionFromEuler([math.pi, 0, math.pi / 2])

    # Lets now build the graph
    G = nx.Graph()

    # Start node
    start_ik = p.calculateInverseKinematics(
        robot_id,
        END_LINK,
        targetPosition=start_point.tolist(),
        targetOrientation=down_orientation,
        lowerLimits=joint_lower_limits,
        upperLimits=joint_upper_limits,
        jointRanges=joint_ranges,
        restPoses=config.jointStartPositions[:7],
        maxNumIterations=400,
        residualThreshold=1e-6,
    )[:7]
    if not is_node_valid(start_point, start_ik, joint_indices, joint_lower_limits, joint_upper_limits, robot_id, obstacles, microwave_aabbs):
        raise RuntimeError("Start pose invalid or in collision")
    G.add_node(0, pos=tuple(start_point), angles=start_ik)

    goal_ik = p.calculateInverseKinematics(
        robot_id,
        END_LINK,
        targetPosition=goal_point.tolist(),
        targetOrientation=down_orientation,
        lowerLimits=joint_lower_limits,
        upperLimits=joint_upper_limits,
        jointRanges=joint_ranges,
        restPoses=config.jointStartPositions[:7],
        maxNumIterations=400,
        residualThreshold=1e-6,
    )[:7]
    if not is_node_valid(goal_point, goal_ik, joint_indices, joint_lower_limits, joint_upper_limits, robot_id, obstacles, microwave_aabbs):
        raise RuntimeError("Start pose invalid or in collision")
    G.add_node(1, pos=tuple(goal_point), angles=goal_ik)

    # Define our x, y and z bounds for sampling the PRM
    bounds = ([-1.25, 1], [-0.8, 1.05], [0.5, 2.0])
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    # Sample Nodes
    for i in range(2, NUM_SAMPLES):
        while(True):
            sample_point = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), np.random.uniform(z_min, z_max)]

            sample_ik = p.calculateInverseKinematics(
                robot_id,
                END_LINK,
                targetPosition=sample_point,
                targetOrientation=down_orientation,
                lowerLimits=joint_lower_limits,
                upperLimits=joint_upper_limits,
                jointRanges=joint_ranges,
                restPoses=config.jointStartPositions[:7],
                maxNumIterations=400,
                residualThreshold=1e-6,
            )[:7]

            # Check if sample point if valid
            if is_node_valid(sample_point, sample_ik, joint_indices, joint_lower_limits, joint_upper_limits, robot_id, obstacles, microwave_aabbs):
                G.add_node(i, pos=tuple(sample_point), angles=sample_ik)
                if i % 1000 == 0:
                    print(f"Sampled {i} Nodes ...")
                break

    # Connect nodes and assign edge weights
    nodes = list(G.nodes)
    positions = {n: np.array(G.nodes[n]["pos"]) for n in nodes}
    for i in nodes:
        neighbors = sorted(nodes, key=lambda n: np.linalg.norm(positions[n] - positions[i]))
        for n in neighbors[1: NUM_NEIGHBORS + 1]:
            if G.has_edge(i, n):
                continue

            # Try to build an edge
            edge = local_planner(
                robot_id,
                joint_indices,
                joint_lower_limits,
                joint_upper_limits,
                joint_ranges,
                positions[i],
                positions[n],
                down_orientation,
                obstacles,
                microwave_aabbs,
            )

            if edge is not None:
                length = np.linalg.norm(positions[i] - positions[n])
                midpoint = 0.5 * (positions[i] + positions[n])
                dist_to_stove = np.linalg.norm(midpoint - stove_center)
                height = 0.5 * (positions[i][2] + positions[n][2])
                G.add_edge(i, n, objectives=(length, 1.0/dist_to_stove, height), traj = edge)
        
        if i % 1000 == 0:
            print(f"Connected {i} Nodes ...")

    print(f"PRM graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, 0, 1, down_orientation, stove_center


def compute_edge_scalar(u, v, edge_attr, weights):
    objs = edge_attr["objectives"]
    return sum(w * o for w, o in zip(weights, objs))


def compute_heuristic(u, v, G, weights):
    p_u = np.array(G.nodes[u]["pos"])
    p_v = np.array(G.nodes[v]["pos"])
    straight = np.linalg.norm(p_u - p_v)
    return weights[0] * straight


def compute_path_cost(G, path, weights):
    """
    Compute the Weighted Max (WM) value for provided path
    """
    num_objectives = len(weights)
    raw_objectives = [0.0] * num_objectives
    cumulative_objectives = [0.0] * num_objectives

    for u, v in zip(path[:-1], path[1:]):
        objs = G.edges[u, v]['objectives']
        for j, (w, o) in enumerate(zip(weights, objs)):
            raw_objectives[j] += float(o)
            cumulative_objectives[j] += float(w * o)
    
    print(f"Raw Objectives: {list(raw_objectives)}")
    print(f"Weighted Objectives: {list(cumulative_objectives)}")

    return raw_objectives, cumulative_objectives


def plot_prm(G, start=0, goal=1):
    for (u, v) in G.edges():
        point1 = np.array(G.nodes[u]["pos"])
        point2 = np.array(G.nodes[v]["pos"])
        p.addUserDebugLine(point1, point2, [0, 1, 0], lineWidth=1, lifeTime=0)
    p.addUserDebugText("START", G.nodes[start]["pos"], [1, 0, 0], textSize=1.5)
    p.addUserDebugText("GOAL", G.nodes[goal]["pos"], [0, 0, 1], textSize=1.5)


def plot_path(G, path_nodes, color=[1, 0, 0]):
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        point1 = np.array(G.nodes[u]["pos"])
        point2 = np.array(G.nodes[v]["pos"])
        p.addUserDebugLine(point1, point2, color, lineWidth=7.5, lifeTime=0)
        p.addUserDebugText(str(u), point1, [1, 1, 1], textSize=1.0)


def nearest_node_id(G, coord):
    """
    Return nearest node for some provided coordinate
    """
    coord = np.asarray(coord, dtype=float)
    pos = nx.get_node_attributes(G, "pos")  
    nodes = np.fromiter(pos.keys(), dtype=int)
    pts   = np.array([pos[n] for n in nodes], dtype=float)
    dists = np.linalg.norm(pts - coord, axis=1)
    i = int(np.argmin(dists))
    return int(nodes[i]), float(dists[i])


def main():
    # Start new PyBullet physics client and GUI
    p.connect(p.GUI)
    p.configureDebugVisualizer(rgbBackground=[0, 0, 0])

    # Load and configure settings
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf") 
    down_orientation = p.getQuaternionFromEuler([math.pi, 0, math.pi / 2])

    # Instantiate Franka-Kitchen Environment
    environment = Environment()
    environment.load()

    # Instantiate Franka robot 
    robot = Robot()

    # Build a PRM
    # G, start_id, goal_id, down_orientation, stove_center = build_prm(robot.id, environment)
    # pickle.dump(G, open('wm_lns/manipulator_experiments/graph.pickle', 'wb'))
    G = pickle.load(open('wm_lns/manipulator_experiments/graph.pickle', 'rb'))

    print(f"PRM graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Solve path within PRM 
    # Conduct 25 Trials:
    for _ in range(25):
        # Randomly sample Start and End position
        # Weights chosen  such that the weighted objective values are approximately equal
        # since that results in a large set of non-dominated paths to explore.
        # Emphasizing a particular objective generally reduces the search space
        weights = [10, 8.7, 9] 

        # Ranges chosen such that the search space is large, i.e large number of non-dominated paths to each node, 
        # otherwise instance is pretty easy.
        config.startPosition = [random.uniform(-0.2, -0.3), random.uniform(0.6, 0.7), 1.3]
        config.goalPosition = [random.uniform(0.25, 0.35), random.uniform(-0.4, -0.5), 1.3]

        print(config.startPosition)
        print(config.goalPosition)

        start_id, d = nearest_node_id(G, config.startPosition)
        goal_id, d = nearest_node_id(G, config.goalPosition)

        # Run WS
        start_ws = time.time()
        _, _, _, _, ws_path, _ = main_ws(G, start_id, goal_id, [], weights, three_dim=True)
        end_ws = time.time()
        runtime_ws = end_ws - start_ws

        # Budgets are tuned so sub-optimal WM solvers have a similar, equal to or greater runtime than WM-LNS
        # Run WM
        start_h_wm = time.time()
        wm_path = heuristic_weighted_max_solver(G, start_id, goal_id, weights, three_dim=True)
        end_h_wm = time.time()
        runtime_h_wm = end_h_wm - start_h_wm

        # Run WM-Poly
        start_h_wm_poly = time.time()
        wm_poly_path = heuristic_weighted_max_solver(G, start_id, goal_id, weights, budget=13, three_dim=True)
        end_h_wm_poly = time.time()
        runtime_h_wm_poly = end_h_wm_poly - start_h_wm_poly

        # Run WM-Beam
        start_h_wm_beam = time.time()
        wm_beam_path = beam_weighted_max_solver(G, start_id, goal_id, weights, budget=10, three_dim=True)
        end_h_wm_beam = time.time()
        runtime_h_wm_beam = end_h_wm_beam - start_h_wm_beam

        # LNS
        start_lns = time.time()
        lns_path, cost_lns, iter_lns, heuristics, time_to_best = lns(G, start_id, goal_id, weights, weight_sampling="guided", iterations=30, non_improving_limit=25, three_dim=True)
        end_lns = time.time()
        runtime_lns = end_lns - start_lns
        print(f"LNS Path found in {runtime_lns:.4f} seconds, Time to best solution {time_to_best}")
        
        # Post Process
        ws_raw_objectives, ws_weighted_objectives = compute_path_cost(G, ws_path, weights)
        wm_raw_objectives, wm_weighted_objectives = compute_path_cost(G, wm_path, weights)
        wm_poly_raw_objectives, wm_poly_weighted_objectives = compute_path_cost(G, wm_poly_path, weights)
        wm_beam_raw_objectives, wm_beam_weighted_objectives = compute_path_cost(G, wm_beam_path, weights)
        lns_raw_objectives, lns_weighted_objectives = compute_path_cost(G, lns_path, weights)

        cost_ws = max(ws_weighted_objectives)
        cost_wm = max(wm_weighted_objectives)
        cost_wm_poly = max(wm_poly_weighted_objectives)
        cost_wm_beam = max(wm_beam_weighted_objectives)
        cost_lns = max(lns_weighted_objectives)

        # Compute Cost Ratios
        ws_ratio = cost_ws/cost_wm
        wm_poly_ratio = cost_wm_poly/cost_wm
        wm_beam_ratio = cost_wm_beam/cost_wm
        lns_ratio = cost_lns/cost_wm

        # Compute Optimality Error
        # Compute the optimality error between objectives that are performing strictly worse
        error_ws_wm = compute_optimality_error(wm_weighted_objectives, ws_weighted_objectives)
        error_wm_poly_wm = compute_optimality_error(wm_weighted_objectives, wm_poly_weighted_objectives)
        error_wm_beam_wm = compute_optimality_error(wm_weighted_objectives, wm_beam_weighted_objectives)
        error_lns_wm = compute_optimality_error(wm_weighted_objectives, lns_weighted_objectives)

        # Normalize Objectives
        (
            all_objectives_wm_unweighted_normalized,
            all_objectives_wm_poly_unweighted_normalized,
            all_objectives_wm_beam_unweighted_normalized,
            all_objectives_ws_unweighted_normalized,
            all_objectives_lns_unweighted_normalized,
        ) = normalize_solutions(wm_raw_objectives, wm_poly_raw_objectives, wm_beam_raw_objectives, ws_raw_objectives, lns_raw_objectives)

        # Compute Spatial Distance
        distance_wm_poly_wm = math.dist(all_objectives_wm_unweighted_normalized, all_objectives_wm_poly_unweighted_normalized)
        distance_wm_beam_wm = math.dist(all_objectives_wm_unweighted_normalized, all_objectives_wm_beam_unweighted_normalized)
        distance_lns_wm = math.dist(all_objectives_wm_unweighted_normalized, all_objectives_lns_unweighted_normalized)
        distance_ws_wm = math.dist(all_objectives_wm_unweighted_normalized, all_objectives_ws_unweighted_normalized)

        environment_label = "Kitchen"

        with open("wm_lns/manipulator_experiments/kitchen_experiments.txt", "a") as f:
            #f.write("Environment,Method,Runtime (s),Cost (WM),Cost Ratio,RSS Error,Euclidean Distance\n")
            # h-WM row 
            f.write(f"{environment_label},WM,{runtime_h_wm:.5f},{cost_wm:.5f},1.00000,0.00000,0.00000\n")
            # WM Poly row
            f.write(f"{environment_label},WM-Poly,{runtime_h_wm_poly:.5f},{cost_wm_poly:.5f},{wm_poly_ratio:.5f},{error_wm_poly_wm:.5f},{distance_wm_poly_wm:.5f}\n")
            # WM Beam row
            f.write(f"{environment_label},WM-Beam,{runtime_h_wm_beam:.5f},{cost_wm_beam:.5f},{wm_beam_ratio:.5f},{error_wm_beam_wm:.5f},{distance_wm_beam_wm:.5f}\n")
            # WS row
            f.write(f"{environment_label},WS,{runtime_ws:.5f},{cost_ws:.5f},{ws_ratio:.5f},{error_ws_wm:.5f},{distance_ws_wm:.5f}\n")
            # WM-LNS row
            f.write(f"{environment_label},WM-LNS,{runtime_lns:.5f},{cost_lns:.5f},{lns_ratio:.5f},{error_lns_wm:.5f},{distance_lns_wm:.5f}\n")

    # Simulate
    while True:
        environment.update()
        p.stepSimulation()
        time.sleep(config.control_dt * TIME_SPEED_FACTOR)

if __name__ == "__main__":
    main()
