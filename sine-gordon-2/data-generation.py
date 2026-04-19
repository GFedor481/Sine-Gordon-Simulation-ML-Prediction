#!/usr/bin/env python3
"""
Optimized sine-Gordon dataset generator.

Generates simulations of a ring of N coupled pendulums governed by the
sine-Gordon equation, detects energy localization (breather formation),
and saves the first 10 time steps to HDF5.

Optimizations over the notebook version:
  - multiprocessing.Pool for true CPU parallelism (no GIL, no lock)
  - Custom fixed-step RK4 integrator (avoids odeint/LSODA overhead)
  - Pre-allocated arrays and minimized temporaries in the ODE RHS
  - Batched HDF5 writes with chunked datasets
  - Optional resume from a partially-generated file
"""

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
N_PENDULUMS = 100
SAVE_STEPS = 10  # number of initial time steps to keep

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("generate_dataset.log"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ODE right-hand side  (pure numpy, zero unnecessary allocations)
# ---------------------------------------------------------------------------
def _rhs(state, out, g, h, N):
    """Compute dx/dt in-place into *out*."""
    x1 = state[:N]
    x2 = state[N:]

    mean_x1 = x1.sum() / N

    out[:N] = x2
    # Coupling via periodic neighbours: x1[(i-1)%N] + x1[(i+1)%N] - 2*x1[i]
    # Using pre-rolled index arrays is faster than np.roll which copies.
    out[N:] = (
        -np.sin(x1)
        + g * (x1[np.arange(-1, N - 1)] + x1[np.arange(1, N + 1) % N] - 2.0 * x1)
        + h * (mean_x1 - x1)
    )


# ---------------------------------------------------------------------------
# Fixed-step RK4 integrator
# ---------------------------------------------------------------------------
def _rk4_integrate(state0, t_array, g, h, N):
    """Integrate using classical RK4 with fixed step size.

    Returns the full trajectory as a (len(t_array), 2*N) array.
    """
    n_steps = len(t_array)
    dim = 2 * N
    trajectory = np.empty((n_steps, dim))
    trajectory[0] = state0

    # Pre-allocate scratch arrays (reused every step)
    k1 = np.empty(dim)
    k2 = np.empty(dim)
    k3 = np.empty(dim)
    k4 = np.empty(dim)
    tmp = np.empty(dim)

    for i in range(n_steps - 1):
        dt = t_array[i + 1] - t_array[i]
        y = trajectory[i]

        _rhs(y, k1, g, h, N)

        np.add(y, 0.5 * dt * k1, out=tmp)
        _rhs(tmp, k2, g, h, N)

        np.add(y, 0.5 * dt * k2, out=tmp)
        _rhs(tmp, k3, g, h, N)

        np.add(y, dt * k3, out=tmp)
        _rhs(tmp, k4, g, h, N)

        trajectory[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return trajectory


# ---------------------------------------------------------------------------
# Localization detection  (vectorised)
# ---------------------------------------------------------------------------
def _find_localization(energy, threshold):
    """Return (pendulum_index, time_index) of first localization, or (-1, -1)."""
    max_per_step = energy.max(axis=1)  # shape (T,)
    hits = np.where(max_per_step > threshold)[0]
    if len(hits) == 0:
        return -1, -1
    t_idx = int(hits[0])
    p_idx = int(energy[t_idx].argmax())
    return p_idx, t_idx


# ---------------------------------------------------------------------------
# Single simulation  (top-level function so it is pickle-able)
# ---------------------------------------------------------------------------
def run_simulation(args):
    """Run one simulation and return trimmed results as a dict."""
    sim_idx, N, t_array, g, h, A_low, A_high, noise_level, threshold_fraction = args

    rng = np.random.default_rng(sim_idx)
    A = float(rng.uniform(A_low, A_high))
    noise = rng.random(N) - 0.5

    x0 = A * np.ones(N) + noise_level * noise
    v0 = np.zeros(N)
    state0 = np.concatenate([x0, v0])

    # Integrate
    traj = _rk4_integrate(state0, t_array, g, h, N)

    theta = traj[:, :N]
    dtheta = traj[:, N:]

    # Energy computation  (vectorised over all time steps)
    theta_next = np.roll(theta, -1, axis=1)
    theta_mean = theta.mean(axis=1, keepdims=True)
    energy = (
        0.5 * dtheta ** 2
        + (1.0 - np.cos(theta))
        + g * (theta - theta_next) ** 2
        + h * (theta - theta_mean) ** 2
    )

    # Localization on full trajectory
    total_energy_t0 = energy[0].sum()
    threshold = threshold_fraction * total_energy_t0
    loc_pendulum, loc_time = _find_localization(energy, threshold)

    # Trim to first SAVE_STEPS
    return {
        "sim_idx": sim_idx,
        "A": A,
        "noise_level": noise_level,
        "g": g,
        "h": h,
        "theta": theta[:SAVE_STEPS].astype(np.float32),
        "dtheta": dtheta[:SAVE_STEPS].astype(np.float32),
        "energy": energy[:SAVE_STEPS].astype(np.float32),
        "localized_pendulum": loc_pendulum,
        "localization_time": loc_time,
    }


# ---------------------------------------------------------------------------
# Batched HDF5 writer
# ---------------------------------------------------------------------------
def _write_batch(hf, results):
    """Write a list of simulation result dicts into the open HDF5 file."""
    for r in results:
        if r is None:
            continue
        key = f"simulation_{r['sim_idx']}"
        grp = hf.create_group(key)

        ic = grp.create_group("initial_conditions")
        ic.attrs["A"] = r["A"]
        ic.attrs["noise_level"] = r["noise_level"]
        ic.attrs["g"] = r["g"]
        ic.attrs["h"] = r["h"]

        grp.create_dataset("theta", data=r["theta"], compression="gzip", compression_opts=1)
        grp.create_dataset("dtheta", data=r["dtheta"], compression="gzip", compression_opts=1)
        grp.create_dataset("energy", data=r["energy"], compression="gzip", compression_opts=1)

        grp.attrs["localized_pendulum"] = r["localized_pendulum"]
        grp.attrs["localization_time"] = r["localization_time"]

    hf.flush()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def generate_dataset(
    num_simulations: int,
    tf: float,
    dt: float,
    g: float,
    h: float,
    start_sim: int = 0,
    A_range: tuple = (1.65, 1.85),
    noise_level: float = 0.025,
    threshold_fraction: float = 0.1,
    batch_size: int = 500,
    workers: int = 0,
    filename: str = "sine_gordon_dataset.h5",
):
    A_low, A_high = float(A_range[0]), float(A_range[1])
    t_array = np.arange(0, tf, dt)
    N = N_PENDULUMS

    if workers <= 0:
        workers = max(1, cpu_count() - 1)

    log.info(
        "Generating %d simulations (start=%d) | tf=%.1f dt=%.1f | %d workers | file=%s",
        num_simulations, start_sim, tf, dt, workers, filename,
    )

    mode = "a" if start_sim > 0 else "w"
    with h5py.File(filename, mode) as hf:
        if start_sim == 0:
            hf.create_dataset("time_steps", data=t_array[:SAVE_STEPS].astype(np.float32))

        total_done = 0
        t0 = time.time()

        for batch_start in range(start_sim, start_sim + num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, start_sim + num_simulations)
            task_args = [
                (idx, N, t_array, g, h, A_low, A_high, noise_level, threshold_fraction)
                for idx in range(batch_start, batch_end)
            ]

            with Pool(processes=workers) as pool:
                results = pool.map(run_simulation, task_args)

            _write_batch(hf, results)

            total_done += len(results)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            log.info(
                "Batch %d–%d done | %d/%d total | %.1f sims/s | elapsed %.0fs",
                batch_start, batch_end - 1, total_done, num_simulations, rate, elapsed,
            )

    log.info("Dataset saved to %s", filename)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate sine-Gordon dataset")
    p.add_argument("-n", "--num-simulations", type=int, default=100_000)
    p.add_argument("--start-sim", type=int, default=0)
    p.add_argument("--tf", type=float, default=150.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--g", type=float, default=0.75)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--A-low", type=float, default=1.65)
    p.add_argument("--A-high", type=float, default=1.85)
    p.add_argument("--noise-level", type=float, default=0.025)
    p.add_argument("--threshold-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    p.add_argument("-o", "--output", type=str, default="sine_gordon_dataset.h5")
    args = p.parse_args()

    generate_dataset(
        num_simulations=args.num_simulations,
        tf=args.tf,
        dt=args.dt,
        g=args.g,
        h=args.h,
        start_sim=args.start_sim,
        A_range=(args.A_low, args.A_high),
        noise_level=args.noise_level,
        threshold_fraction=args.threshold_fraction,
        batch_size=args.batch_size,
        workers=args.workers,
        filename=args.output,
    )


if __name__ == "__main__":
    main()
