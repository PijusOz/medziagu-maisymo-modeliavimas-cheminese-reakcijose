import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np


SHAPE_OPTIONS = {
    "Square": "square",
    "Triangle": "triangle",
    "Hexagon": "hexagon",
}


def stable_dt_diffusion(D, dx, dy):
    return 0.95 / (2.0 * D * (1.0 / dx**2 + 1.0 / dy**2))


def apply_no_flux(C):
    C[:, 0] = C[:, 1]
    C[:, -1] = C[:, -2]
    C[0, :] = C[1, :]
    C[-1, :] = C[-2, :]
    return C


def laplacian(C, dx, dy):
    L = np.zeros_like(C)
    L[1:-1, 1:-1] = (
        (C[1:-1, 2:] - 2.0 * C[1:-1, 1:-1] + C[1:-1, :-2]) / dx**2
        + (C[2:, 1:-1] - 2.0 * C[1:-1, 1:-1] + C[:-2, 1:-1]) / dy**2
    )
    return L


def monte_carlo_balanced_pattern(tile_coords, grid_shape, rng, trials):
    unique_tiles, tile_inverse, tile_weights = np.unique(
        tile_coords, axis=0, return_inverse=True, return_counts=True
    )
    num_tiles = len(unique_tiles)
    if num_tiles < 2:
        raise ValueError("At least two visible starting pieces are needed")

    b_counts = [num_tiles // 2]
    if num_tiles % 2 == 1:
        b_counts.append(num_tiles // 2 + 1)

    best_labels = None
    best_score = None
    best_a_points = 0
    best_b_points = 0

    for _ in range(trials):
        order = rng.permutation(num_tiles)

        for b_count in b_counts:
            labels = np.zeros(num_tiles, dtype=int)
            labels[order[:b_count]] = 1

            a_points = int(tile_weights[labels == 0].sum())
            b_points = int(tile_weights[labels == 1].sum())
            point_difference = abs(a_points - b_points)
            tile_difference = abs((num_tiles - b_count) - b_count)
            score = (point_difference, tile_difference)

            if best_score is None or score < best_score:
                best_score = score
                best_labels = labels.copy()
                best_a_points = a_points
                best_b_points = b_points

                if point_difference <= 1 and tile_difference == 0:
                    break

        if best_score is not None and best_score[0] <= 1 and best_score[1] == 0:
            break

    pattern = best_labels[tile_inverse].reshape(grid_shape)
    return pattern, num_tiles, best_a_points, best_b_points


def square_pattern(x, y, pieces_per_side, random_layout, rng, trials):
    tile_x = np.minimum((x / x[-1] * pieces_per_side).astype(int), pieces_per_side - 1)
    tile_y = np.minimum((y / y[-1] * pieces_per_side).astype(int), pieces_per_side - 1)
    tile_x_grid, tile_y_grid = np.meshgrid(tile_x, tile_y)

    if random_layout:
        tile_coords = np.stack((tile_x_grid, tile_y_grid), axis=-1).reshape(-1, 2)
        return monte_carlo_balanced_pattern(tile_coords, tile_x_grid.shape, rng, trials)

    pattern = (tile_x_grid + tile_y_grid) % 2
    a_points = int(np.count_nonzero(pattern == 0))
    b_points = int(np.count_nonzero(pattern == 1))
    return pattern, pieces_per_side * pieces_per_side, a_points, b_points


def triangle_pattern(X, Y, a, pieces_per_side, random_layout, rng, trials):
    triangle_side = a / pieces_per_side
    triangle_height = np.sqrt(3.0) * triangle_side / 2.0

    v = Y / triangle_height
    u = X / triangle_side - 0.5 * v
    u_floor = np.floor(u).astype(int)
    v_floor = np.floor(v).astype(int)
    u_local = u - u_floor
    v_local = v - v_floor

    upward_triangle = (u_local + v_local) <= 1.0

    if random_layout:
        tile_coords = np.stack(
            (u_floor, v_floor, upward_triangle.astype(int)), axis=-1
        ).reshape(-1, 3)
        return monte_carlo_balanced_pattern(tile_coords, X.shape, rng, trials)

    pattern = np.where(upward_triangle, 0, 1)
    a_points = int(np.count_nonzero(pattern == 0))
    b_points = int(np.count_nonzero(pattern == 1))
    visible_tiles = len(
        np.unique(
            np.stack((u_floor, v_floor, upward_triangle.astype(int)), axis=-1).reshape(-1, 3),
            axis=0,
        )
    )
    return pattern, visible_tiles, a_points, b_points


def hexagon_pattern(X, Y, a, pieces_per_side, random_layout, rng, trials):
    hex_size = a / (np.sqrt(3.0) * pieces_per_side)
    x_shift = 0.5 * np.sqrt(3.0) * hex_size
    y_shift = hex_size

    X_hex = X - x_shift
    Y_hex = Y - y_shift

    q = (np.sqrt(3.0) / 3.0 * X_hex - 1.0 / 3.0 * Y_hex) / hex_size
    r = (2.0 / 3.0 * Y_hex) / hex_size

    x_cube = q
    z_cube = r
    y_cube = -x_cube - z_cube

    rx = np.round(x_cube)
    ry = np.round(y_cube)
    rz = np.round(z_cube)

    x_diff = np.abs(rx - x_cube)
    y_diff = np.abs(ry - y_cube)
    z_diff = np.abs(rz - z_cube)

    x_biggest = (x_diff > y_diff) & (x_diff > z_diff)
    y_biggest = ~x_biggest & (y_diff > z_diff)
    z_biggest = ~(x_biggest | y_biggest)

    rx[x_biggest] = -ry[x_biggest] - rz[x_biggest]
    ry[y_biggest] = -rx[y_biggest] - rz[y_biggest]
    rz[z_biggest] = -rx[z_biggest] - ry[z_biggest]

    q_hex = rx.astype(int)
    r_hex = rz.astype(int)

    if random_layout:
        tile_coords = np.stack((q_hex, r_hex), axis=-1).reshape(-1, 2)
        return monte_carlo_balanced_pattern(tile_coords, X.shape, rng, trials)

    pattern = (q_hex + r_hex) % 2
    a_points = int(np.count_nonzero(pattern == 0))
    b_points = int(np.count_nonzero(pattern == 1))
    visible_tiles = len(np.unique(np.stack((q_hex, r_hex), axis=-1).reshape(-1, 2), axis=0))
    return pattern, visible_tiles, a_points, b_points


def make_initial_conditions(params):
    a = params["a"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    pieces_per_side = params["pieces_per_side"]
    shape = params["shape"]
    random_layout = params["random_layout"]
    trials = params["monte_carlo_trials"]
    random_seed = params["random_seed"]

    if pieces_per_side < 1:
        raise ValueError("Pieces per side must be at least 1")
    if trials < 1:
        raise ValueError("Monte Carlo trials must be at least 1")

    x = np.linspace(0, a, Nx)
    y = np.linspace(0, a, Ny)
    X, Y = np.meshgrid(x, y)

    rng = np.random.default_rng(random_seed)

    if shape == "square":
        pattern, visible_tiles, a_points, b_points = square_pattern(
            x, y, pieces_per_side, random_layout, rng, trials
        )
    elif shape == "triangle":
        pattern, visible_tiles, a_points, b_points = triangle_pattern(
            X, Y, a, pieces_per_side, random_layout, rng, trials
        )
    elif shape == "hexagon":
        pattern, visible_tiles, a_points, b_points = hexagon_pattern(
            X, Y, a, pieces_per_side, random_layout, rng, trials
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")

    A = np.zeros((Ny, Nx), dtype=float)
    B = np.zeros((Ny, Nx), dtype=float)
    AB = np.zeros((Ny, Nx), dtype=float)

    A[pattern == 0] = 1.0
    B[pattern == 1] = 1.0

    info = {
        "x": x,
        "y": y,
        "dx": x[1] - x[0],
        "dy": y[1] - y[0],
        "visible_tiles": visible_tiles,
        "a_points": a_points,
        "b_points": b_points,
    }
    return A, B, AB, info


def simulate_until_done(params):
    A, B, AB, info = make_initial_conditions(params)

    dx = info["dx"]
    dy = info["dy"]
    D1 = params["D1"]
    D2 = params["D2"]
    D_AB = params["D_AB"]
    k = params["k"]
    max_time = params["max_time"]
    completion_limit = params["completion_limit"] / 100.0

    Dmax = max(D1, D2, D_AB)
    dt_diff = stable_dt_diffusion(Dmax, dx, dy)
    dt_react = 0.2 / max(k, 1e-12)
    dt = min(dt_diff, dt_react)

    max_steps = int(np.ceil(max_time / dt))

    def mass(C):
        return C.sum() * dx * dy

    mA0, mB0, mAB0 = mass(A), mass(B), mass(AB)
    max_possible_AB = min(mA0, mB0)
    target_AB = completion_limit * max_possible_AB

    if max_possible_AB <= 0:
        raise ValueError("Both starting materials must be present")

    snapshots = {}
    target_progress_values = np.linspace(0.0, completion_limit, params["snapshots"])
    next_snapshot = 0

    def completion_fraction():
        return mass(AB) / max_possible_AB

    def save_snapshots_if_needed(step, progress):
        nonlocal next_snapshot
        while (
            next_snapshot < len(target_progress_values)
            and progress >= target_progress_values[next_snapshot] - 1e-12
        ):
            snapshots[step] = (A.copy(), B.copy(), AB.copy())
            next_snapshot += 1

    progress = completion_fraction()
    save_snapshots_if_needed(0, progress)

    final_step = 0
    reached_limit = False

    for n in range(1, max_steps + 1):
        apply_no_flux(A)
        apply_no_flux(B)
        apply_no_flux(AB)

        LA = laplacian(A, dx, dy)
        LB = laplacian(B, dx, dy)
        LAB = laplacian(AB, dx, dy)

        R = k * A * B

        A = A + dt * (D1 * LA - R)
        B = B + dt * (D2 * LB - R)
        AB = AB + dt * (D_AB * LAB + R)

        A = np.maximum(A, 0.0)
        B = np.maximum(B, 0.0)
        AB = np.maximum(AB, 0.0)

        progress = completion_fraction()
        save_snapshots_if_needed(n, progress)

        final_step = n
        if mass(AB) >= target_AB:
            reached_limit = True
            snapshots[n] = (A.copy(), B.copy(), AB.copy())
            break

    if final_step not in snapshots:
        snapshots[final_step] = (A.copy(), B.copy(), AB.copy())

    mA1, mB1, mAB1 = mass(A), mass(B), mass(AB)
    final_time = final_step * dt
    final_completion = 100.0 * mAB1 / max_possible_AB

    info.update(
        {
            "dt": dt,
            "Nt": final_step,
            "dt_diff": dt_diff,
            "dt_react": dt_react,
            "mA0": mA0,
            "mB0": mB0,
            "mAB0": mAB0,
            "mA1": mA1,
            "mB1": mB1,
            "mAB1": mAB1,
            "max_possible_AB": max_possible_AB,
            "target_AB": target_AB,
            "final_time": final_time,
            "completion_percent": final_completion,
            "reached_limit": reached_limit,
        }
    )

    return snapshots, info


def plot_results(params, snapshots, info):
    steps_sorted = sorted(snapshots.keys())
    ncols = len(steps_sorted)
    a = params["a"]
    dt = info["dt"]

    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 10), constrained_layout=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    vmin_A, vmax_A = 0.0, 1.0
    vmin_B, vmax_B = 0.0, 1.0
    vmin_AB, vmax_AB = 0.0, 1.0

    for j, step in enumerate(steps_sorted):
        t = step * dt
        a_snap, b_snap, ab_snap = snapshots[step]
        completion = 100.0 * ab_snap.sum() * info["dx"] * info["dy"] / info["max_possible_AB"]

        axA = axes[0, j]
        imA = axA.imshow(
            a_snap, origin="lower", extent=[0, a, 0, a],
            vmin=vmin_A, vmax=vmax_A, aspect="equal"
        )
        axA.set_title(f"A, t={t:.3g}, done={completion:.1f}%")
        axA.set_xlabel("x")
        axA.set_ylabel("y")

        axB = axes[1, j]
        imB = axB.imshow(
            b_snap, origin="lower", extent=[0, a, 0, a],
            vmin=vmin_B, vmax=vmax_B, aspect="equal"
        )
        axB.set_title(f"B, t={t:.3g}, done={completion:.1f}%")
        axB.set_xlabel("x")
        axB.set_ylabel("y")

        axAB = axes[2, j]
        imAB = axAB.imshow(
            ab_snap, origin="lower", extent=[0, a, 0, a],
            vmin=vmin_AB, vmax=vmax_AB, aspect="equal"
        )
        axAB.set_title(f"AB, t={t:.3g}, done={completion:.1f}%")
        axAB.set_xlabel("x")
        axAB.set_ylabel("y")

    cbarA = fig.colorbar(imA, ax=axes[0, :], shrink=0.9, pad=0.02)
    cbarA.set_label("Concentration A")

    cbarB = fig.colorbar(imB, ax=axes[1, :], shrink=0.9, pad=0.02)
    cbarB.set_label("Concentration B")

    cbarAB = fig.colorbar(imAB, ax=axes[2, :], shrink=0.9, pad=0.02)
    cbarAB.set_label("Concentration AB")

    layout_name = "random" if params["random_layout"] else "ordered"
    shape_name = params["shape"].capitalize()
    stop_text = "reached" if info["reached_limit"] else "not reached"
    fig.suptitle(
        f"{shape_name}, {layout_name}: target {params['completion_limit']}% {stop_text}; "
        f"final t={info['final_time']:.4g}, final done={info['completion_percent']:.2f}%",
        fontsize=12,
    )

    plt.show()


class SimulationAppFinal:
    def __init__(self, root):
        self.root = root
        self.root.title("Reaction completion app")
        self.root.geometry("560x720")
        self.root.minsize(520, 620)
        self.root.resizable(True, True)

        self.shape_var = tk.StringVar(value="Square")
        self.random_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="")

        self.entries = {}
        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        footer = ttk.Frame(main)
        footer.pack(side="bottom", fill="x", pady=(12, 0))

        content = ttk.Frame(main)
        content.pack(side="top", fill="both", expand=True)

        title = ttk.Label(content, text="Reaction Completion Simulation", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        choices = ttk.LabelFrame(content, text="Inputs", padding=12)
        choices.pack(fill="x")

        ttk.Label(choices, text="Shape").grid(row=0, column=0, sticky="w", pady=4)
        shape_box = ttk.Combobox(
            choices,
            textvariable=self.shape_var,
            values=list(SHAPE_OPTIONS.keys()),
            state="readonly",
            width=18,
        )
        shape_box.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(choices, text="Placement").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(
            choices,
            text="Random placement",
            variable=self.random_var,
        ).grid(row=1, column=1, sticky="w", pady=4)

        choices.columnconfigure(1, weight=1)

        params = ttk.LabelFrame(content, text="Parameters", padding=12)
        params.pack(fill="x", pady=(12, 0))

        defaults = [
            ("pieces_per_side", "Pieces per side", "8"),
            ("Nx", "Grid points x", "151"),
            ("Ny", "Grid points y", "151"),
            ("completion_limit", "Done limit (%)", "98"),
            ("max_time", "Safety max time", "50"),
            ("snapshots", "Snapshots", "5"),
            ("D1", "Diffusion A", "5e-4"),
            ("D2", "Diffusion B", "5e-3"),
            ("D_AB", "Diffusion AB", "1e-3"),
            ("k", "Reaction rate", "5.0"),
            ("monte_carlo_trials", "Monte Carlo trials", "20000"),
        ]

        for row, (key, label, default) in enumerate(defaults):
            ttk.Label(params, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(params, width=20)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.entries[key] = entry

        params.columnconfigure(1, weight=1)

        random_frame = ttk.LabelFrame(content, text="Random options", padding=12)
        random_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(random_frame, text="Seed").grid(row=0, column=0, sticky="w", pady=3)
        seed_entry = ttk.Entry(random_frame, textvariable=self.seed_var, width=20)
        seed_entry.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(random_frame, text="Leave empty for a new layout each run").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        random_frame.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Choose inputs, then generate results.")
        ttk.Label(footer, textvariable=self.status_var, wraplength=500).pack(
            anchor="w", pady=(0, 8)
        )

        button = ttk.Button(footer, text="Generate results", command=self.run_simulation)
        button.pack(fill="x", ipady=4)

    def read_params(self):
        seed_text = self.seed_var.get().strip()
        random_seed = None if seed_text == "" else int(seed_text)

        params = {
            "shape": SHAPE_OPTIONS[self.shape_var.get()],
            "random_layout": self.random_var.get(),
            "random_seed": random_seed,
            "a": 1.0,
            "pieces_per_side": int(self.entries["pieces_per_side"].get()),
            "Nx": int(self.entries["Nx"].get()),
            "Ny": int(self.entries["Ny"].get()),
            "completion_limit": float(self.entries["completion_limit"].get()),
            "max_time": float(self.entries["max_time"].get()),
            "snapshots": int(self.entries["snapshots"].get()),
            "D1": float(self.entries["D1"].get()),
            "D2": float(self.entries["D2"].get()),
            "D_AB": float(self.entries["D_AB"].get()),
            "k": float(self.entries["k"].get()),
            "monte_carlo_trials": int(self.entries["monte_carlo_trials"].get()),
        }

        if params["Nx"] < 3 or params["Ny"] < 3:
            raise ValueError("Grid points must be at least 3 in both directions")
        if not (0.0 < params["completion_limit"] < 100.0):
            raise ValueError("Done limit must be greater than 0 and less than 100")
        if params["max_time"] <= 0:
            raise ValueError("Safety max time must be positive")
        if params["snapshots"] < 1:
            raise ValueError("Snapshots must be at least 1")
        if min(params["D1"], params["D2"], params["D_AB"]) <= 0:
            raise ValueError("Diffusion coefficients must be positive")
        if params["k"] <= 0:
            raise ValueError("Reaction rate must be positive")

        return params

    def run_simulation(self):
        try:
            params = self.read_params()
            self.status_var.set("Generating results until reaction limit is reached...")
            self.root.update_idletasks()

            snapshots, info = simulate_until_done(params)

            reached_text = "reached" if info["reached_limit"] else "not reached"
            status = (
                f"Done limit {reached_text}. Final time: {info['final_time']:.4g}. "
                f"Completion: {info['completion_percent']:.2f}%."
            )
            self.status_var.set(status)

            print(f"dx={info['dx']:.4g}, dy={info['dy']:.4g}, dt={info['dt']:.4g}, steps={info['Nt']}")
            print(f"dt bounds: diffusion<{info['dt_diff']:.4g}, reaction<{info['dt_react']:.4g}")
            print(f"Visible pieces: {info['visible_tiles']}")
            print(f"A grid points: {info['a_points']}")
            print(f"B grid points: {info['b_points']}")
            print(f"Grid-point difference: {abs(info['a_points'] - info['b_points'])}")
            print(f"Done limit: {params['completion_limit']:.4g}%")
            print(f"Final time: {info['final_time']:.6g}")
            print(f"Final completion: {info['completion_percent']:.6g}%")
            print(f"Limit reached: {info['reached_limit']}")
            print(f"Mass A:  start={info['mA0']:.6g}, end={info['mA1']:.6g}, delta={info['mA1']-info['mA0']:+.3g}")
            print(f"Mass B:  start={info['mB0']:.6g}, end={info['mB1']:.6g}, delta={info['mB1']-info['mB0']:+.3g}")
            print(f"Mass AB: start={info['mAB0']:.6g}, end={info['mAB1']:.6g}, delta={info['mAB1']-info['mAB0']:+.3g}")
            print(
                f"Total:   start={(info['mA0']+info['mB0']+info['mAB0']):.6g}, "
                f"end={(info['mA1']+info['mB1']+info['mAB1']):.6g}, "
                f"delta={(info['mA1']+info['mB1']+info['mAB1'])-(info['mA0']+info['mB0']+info['mAB0']):+.3g}"
            )

            plot_results(params, snapshots, info)

        except Exception as exc:
            self.status_var.set("Could not generate results.")
            traceback.print_exc()
            messagebox.showerror("Error", str(exc))


def main():
    root = tk.Tk()
    SimulationAppFinal(root)
    root.mainloop()


if __name__ == "__main__":
    main()
