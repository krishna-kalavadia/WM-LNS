import csv
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 35,
        "axes.labelsize": 35, 
        "xtick.labelsize": 20, 
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "legend.title_fontsize": 22,
        "figure.titlesize": 20,
    }
)
plt.rcParams["axes.linewidth"] = 2


def read_comparison_experiment_data():
    data = defaultdict(lambda: defaultdict(list))

    filename = "data/instance1_experiments.txt"
    # filename = "data/instance2_experiments.txt"

    with open(filename, "r") as f:
        reader = csv.reader(f)
        first = next(reader)
        if not any("Runtime" in cell for cell in first):
            f.seek(0)
            reader = csv.reader(f)

        for row in reader:
            if len(row) < 7:
                continue
            env, method, rt_s, cost_s, cr_s, err_s, ed_s = row[:7]
            try:
                rt = float(rt_s)
                cost = float(cost_s)
                cr = float(cr_s)
                err = float(err_s)
                ed = float(ed_s)
            except ValueError:
                print("Error trying to deal with row:", row)
                continue
            data[env.strip()][method.strip()].append((rt, cost, cr, err, ed))

    def stats(vals):
        """Return mean and std"""
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)
        s = math.sqrt(var)
        return m, s

    for env in sorted(data):
        print("Environment:", env)
        for method in sorted(data[env]):
            rows = data[env][method]
            runtimes = [r[0] for r in rows]
            costs = [r[1] for r in rows]
            ratios = [r[2] for r in rows]
            errors = [r[3] for r in rows]
            dists = [r[4] for r in rows]

            mean_rt, std_rt = stats(runtimes)
            mean_cost, std_cost = stats(costs)
            mean_cr, std_cr = stats(ratios)
            mean_err, std_err = stats(errors)
            mean_ed, std_ed = stats(dists)

            print(f" Method: {method}")
            print(f"    Runtime    {mean_rt:.5f} s ± {std_rt:.5f}")
            print(f"    Cost       {mean_cost:.5f} ± {std_cost:.5f}")
            print(f"    Cost ratio {mean_cr:.5f} ± {std_cr:.5f}")
            print(f"    RSS error  {mean_err:.5f} ± {std_err:.5f}")
            print(f"    Distance   {mean_ed:.5f} ± {std_ed:.5f}")

    map_types = sorted(data.keys())
    methods = sorted(data[map_types[0]].keys())

    if len(map_types) == 1:
        simple_map_types = ["Small"]
    elif len(map_types) == 2:
        simple_map_types = ["Medium", "Small"]
    else:
        simple_map_types = ["Large", "Medium", "Small"]

    map_types.reverse()
    simple_map_types.reverse()

    n_maps = len(map_types)
    n_methods = len(methods)
    indices = np.arange(n_maps)
    width = 0.6 / n_methods

    # Define our colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_cols = prop_cycle.by_key()["color"]
    print(default_cols)
    method_colors = {
        m: default_cols[i % len(default_cols)] for i, m in enumerate(methods)
    }
    CUSTOM_PALETTE = {
        "WM-LNS": "#ff7f0e",  # orange
        "WM-beam": "#e15759",  # red
        "WM-poly": "#9467bd",  # purple
        "WM": "#2ca02c",  # green
        "WS": "#1f77b4",  # blue
    }
    fallback = "#7f7f7f"  # grey
    method_colors = {m: CUSTOM_PALETTE.get(m, fallback) for m in methods}

    # Runtime‐ratio to WS
    base = "WS"
    desired_order = ["WM-LNS", "WM-beam", "WM-poly", "WM"]
    runtime_methods = [m for m in desired_order if m in methods and m != base]
    n_rm = len(runtime_methods)
    width = 0.75 / n_rm
    indices = np.arange(len(map_types))

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(runtime_methods):
        ratio_per_map = [
            [
                rt_m / rt_ws
                for rt_m, rt_ws in zip(
                    [rt for rt, *_ in data[env][m]], [rt for rt, *_ in data[env][base]]
                )
            ]
            for env in map_types
        ]
        pos = indices + i * width
        plt.boxplot(
            ratio_per_map,
            positions=pos,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=method_colors[m], alpha=0.6),
            medianprops=dict(color="black"),
        )

    plt.xticks(indices + width * (n_rm - 1) / 2, simple_map_types)
    plt.ylabel("Comp Time Ratio")
    plt.xlabel("Map Scale")
    plt.yscale("log")

    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    label_map = {
        "WM-LNS": "WM-LNS (Ours)",
        "WM-beam": "WM-beam (Ours)",
        "WM-poly": "WM-poly",
        "WM": "WM",
    }
    handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", alpha=0.6)
        for m in runtime_methods
    ]
    labels = [label_map[m] for m in runtime_methods]
    plt.legend(
        handles, labels, title="Method", loc="upper left", frameon=True, fancybox=True
    )
    plt.tight_layout()
    plt.show()

    # Cost Ratio
    exclude = {"WM", "h-WM"}
    cost_methods = [m for m in methods if m not in exclude]
    desired_order = ["WM-LNS", "WM-beam", "WM-poly", "WS"]
    cost_methods = [m for m in desired_order if m in methods and m not in exclude]
    n_cm = len(cost_methods)
    width = 0.75 / n_cm

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(cost_methods):
        ratios_per_map = [
            [(cr - 1) * 100 for *_, cr, _, _ in data[env][m]] for env in map_types
        ]
        pos = indices + i * width
        plt.boxplot(
            ratios_per_map,
            positions=pos,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=method_colors[m], alpha=0.6),
            medianprops=dict(color="black"),
        )

    plt.xticks(indices + width * (n_cm - 1) / 2, simple_map_types)
    plt.ylabel("Percentage Error")
    plt.xlabel("Map Scale")

    label_map = {
        "WM-LNS": "WM-LNS (Ours)",
        "WM-beam": "WM-beam (Ours)",
        "WM-poly": "WM-poly",
        "WS": "WS",
    }
    handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", alpha=0.6)
        for m in cost_methods
    ]
    labels = [label_map[m] for m in cost_methods]
    plt.legend(
        handles, labels, title="Method", loc="upper left", frameon=True, fancybox=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    read_comparison_experiment_data()
