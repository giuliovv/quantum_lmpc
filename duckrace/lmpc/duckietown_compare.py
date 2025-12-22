from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DuckietownCompareConfig:
    frame_rate: int = 10
    traj_samples: int = 500
    traj_last_value_distance: float = 1.035
    border_distance: float = 0.16

    # MPC first loop
    N_MPC: int = 10
    delay_seconds: float = 0.15

    # LMPC
    N: int = 3
    K: int = 6
    i_j: int = 4
    i_j_all: bool = False
    more: int = 20
    n_iterations: int = 5


def _build_env(frame_rate: int):
    try:
        from gym_duckietown.simulator import Simulator
    except Exception as e:  # pragma: no cover
        if "NoSuchDisplayException" in repr(e):
            raise RuntimeError(
                "Duckietown simulator requires an X display. Run in a desktop session or under Xvfb "
                "(e.g. `xvfb-run -s '-screen 0 1280x720x24' python3 -m duckrace.lmpc.duckietown_compare ...`)."
            ) from e
        raise RuntimeError(
            "Duckietown simulator not available. Install `duckietown-gym-daffy` "
            "and its dependencies."
        ) from e

    env = Simulator(
        "ETH_large_loop",
        full_transparency=True,
        domain_rand=False,
        user_tile_start=[3, 1],
        seed=42,
        max_steps=float("inf"),
        robot_speed=1.0,
        frame_rate=int(frame_rate),
    )
    return env


def run_compare(
    *,
    cfg: DuckietownCompareConfig,
    quantum: bool,
    enable_augment: bool,
    quantum_backend: str,
    quantum_horizon: int,
    quantum_samples_per_start: int,
    quantum_start_states: int,
    quantum_min_terminal_v: float,
    quantum_min_advance_steps: int,
):
    import utils
    from casadi import DM, Function, vertcat
    from scipy import spatial

    from duckrace.lmpc import LMPCKit, LMPCRunConfig, compare_lmpc_baseline_vs_quantum
    from duckrace.quantum.lmpc_augment import QuantumLMPCAugmenterConfig
    from duckrace.quantum.sampler import QuantumSamplerConfig, reset_quantum_timing, get_quantum_timing

    env = _build_env(cfg.frame_rate)

    # Trajectory + borders
    env.reset()
    env.unwrapped.start_pose = [[0.11699990272521976, 0, 0.41029359288296474], np.pi / 2]
    dt = 1.0 / env.frame_rate

    traj = utils.get_trajectory(
        env, samples=cfg.traj_samples, method="distance", last_value_distance=cfg.traj_last_value_distance
    )

    env.reset()
    env.unwrapped.start_pose = [[0.3, 0, 0.41029359288296474], np.pi / 2]
    env.reset()
    env.unwrapped.start_pose = [[0.3, 0, 0.41029359288296474], np.pi / 2]

    pose = utils.get_position(env)
    distance, index = spatial.KDTree(traj).query([pose.x, pose.y])
    start_index0 = int(index)

    # `utils.get_border` returns N-1 points (it iterates `traj[:-1]`), which makes
    # borders look discontinuous and breaks indexing against the full trajectory.
    # Close the loop by appending the first point so we get N border samples.
    traj_closed = np.vstack([traj, traj[0]])
    inside, outside = utils.get_border(traj_closed, distance=cfg.border_distance)

    angles = np.zeros(traj.shape[0], dtype=float)
    angles[:-1] = np.arctan2(traj[1:, 1] - traj[:-1, 1], traj[1:, 0] - traj[:-1, 0])
    angles[-1] = np.arctan2(traj[0, 1] - traj[-1, 1], traj[0, 0] - traj[-1, 0])
    traj3 = np.concatenate((traj, angles.reshape(-1, 1)), axis=1)

    # First loop: standard MPC (M.casadi) to seed safe set
    Mpc = Function.load("M.casadi")
    F = utils.model_F(dt=dt)

    N_MPC = int(cfg.N_MPC)
    delay = int(round(cfg.delay_seconds / dt))
    u_delay = DM(np.zeros((2, delay)))

    index_dm = DM(int(index))
    traj_dm = DM(traj3)

    r = traj_dm[index_dm : index_dm + N_MPC + 1, :2].T
    tr = traj_dm[index_dm : index_dm + N_MPC + 1, 2].T

    x = np.array([pose.x, pose.y, pose.theta, 0.0, 0.0], dtype=float)
    X_log = np.empty((5, 0))
    U_log = np.empty((2, 0))

    kdtree = spatial.KDTree(traj3[:, :2])
    finish_t = int(10 * env.frame_rate)
    finish_y = float(pose.y)
    finish_x = float(pose.x)
    # First loop stop condition: require 1 wrap around the centerline index after at least finish_t.
    t = 0
    laps = 0
    traj_len = int(traj3.shape[0])
    wrap_thresh = max(1, traj_len // 2)
    last_idx = int(index)
    start_unwrapped = int(index)

    while True:
        u = Mpc(x, r, tr, u_delay, 1e3, 5e-4, 1, 1e-3)
        U_log = np.column_stack((U_log, np.asarray(u).reshape(2)))
        X_log = np.column_stack((X_log, np.asarray(x).reshape(5)))
        u_delay = np.column_stack((u_delay, u))[:, -delay:]
        x = np.asarray(F(x, u)).reshape(-1)

        _, index = kdtree.query(np.array([x[0], x[1]]).reshape(-1))
        if index + N_MPC + 1 < traj_dm.shape[0]:
            r = traj_dm[index : index + N_MPC + 1, :2].T
            tr = traj_dm[index : index + N_MPC + 1, 2].T
        else:
            r = vertcat(traj_dm[index:, :2], traj_dm[: index + N_MPC + 1 - traj_dm.shape[0], :2]).T
            tr = vertcat(traj_dm[index:, 2], traj_dm[: index + N_MPC + 1 - traj_dm.shape[0], 2]).T

        # Update progress along centerline index (wrap-aware).
        if int(index) < int(last_idx) - wrap_thresh:
            laps += 1
        elif int(index) > int(last_idx) + wrap_thresh:
            laps -= 1
        last_idx = int(index)
        unwrapped = int(index) + laps * traj_len

        if (t >= finish_t) and ((unwrapped - start_unwrapped) >= traj_len):
            break
        t += 1

        if (t / env.frame_rate) > 60:
            break

    # Compare LMPC iterations - load or generate LMPC for current K, i_j, N
    from pathlib import Path
    lmpc_file = f"LMPC_K{cfg.K}_ij{cfg.i_j}_N{cfg.N}.casadi"
    if cfg.K == 8 and cfg.i_j == 4 and cfg.N == 2:
        lmpc_file = "LMPC.casadi"  # default file

    if not Path(lmpc_file).exists():
        print(f"Generating {lmpc_file} for K={cfg.K}, i_j={cfg.i_j}, N={cfg.N}...")
        from scripts.generate_lmpc import generate_lmpc
        generate_lmpc(K=cfg.K, i_j=cfg.i_j, N=cfg.N, output=lmpc_file)

    M_lmpc = Function.load(lmpc_file)

    run_cfg = LMPCRunConfig(
        N=int(cfg.N),
        K=int(cfg.K),
        i_j=int(cfg.i_j),
        frame_rate=int(env.frame_rate),
        more=int(cfg.more),
        dt=float(dt),
        finish_y=float(finish_y),
        finish_x_min=float(finish_x - 0.1),
        finish_t_steps=int(finish_t),
        max_seconds=60.0,
        i_j_all=bool(cfg.i_j_all),
        start_traj_index=int(start_index0),
        traj_len=int(traj3.shape[0]),
    )

    kit = LMPCKit(
        M_lmpc=M_lmpc,
        F_model=F,
        traj_xytheta=traj3,
        inside_xy=np.asarray(inside, dtype=float),
        outside_xy=np.asarray(outside, dtype=float),
        x_init=np.asarray(x, dtype=float),  # Use final state after first loop (with velocity)
        idx_init=int(index),
        X_log_first_loop=np.asarray(X_log, dtype=float),
    )

    if not quantum:
        from duckrace.lmpc.runner import run_lmpc_iterations

        baseline = run_lmpc_iterations(
            M_lmpc=kit.M_lmpc,
            F_model=kit.F_model,
            traj_xytheta=kit.traj_xytheta,
            inside_xy=kit.inside_xy,
            outside_xy=kit.outside_xy,
            x_init=kit.x_init,
            idx_init=kit.idx_init,
            X_log_first_loop=kit.X_log_first_loop,
            config=run_cfg,
            n_iterations=int(cfg.n_iterations),
        )
        return baseline, None, traj3, np.asarray(inside, dtype=float), np.asarray(outside, dtype=float), None

    # Reset quantum timing before quantum run
    reset_quantum_timing()

    qcfg = QuantumLMPCAugmenterConfig(
        horizon=int(quantum_horizon),
        n_start_states=int(quantum_start_states),
        n_samples_per_start=int(quantum_samples_per_start),
        sampler=QuantumSamplerConfig(backend=quantum_backend, n_iterations=1, seed=0),
        min_terminal_v=float(quantum_min_terminal_v),
        min_advance_steps=int(quantum_min_advance_steps),
    )

    baseline, quantum_res = compare_lmpc_baseline_vs_quantum(
        kit=kit,
        run_cfg=run_cfg,
        n_iterations=int(cfg.n_iterations),
        quantum_cfg=qcfg,
        enable_augment=bool(enable_augment),
    )

    # Capture quantum timing
    quantum_timing = get_quantum_timing()

    return baseline, quantum_res, traj3, np.asarray(inside, dtype=float), np.asarray(outside, dtype=float), quantum_timing


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckrace.lmpc.duckietown_compare",
        description="Compare baseline LMPC vs quantum-augmented LMPC on Duckietown (ETH_large_loop).",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--K", type=int, default=6, help="Number of neighbors per lap (requires matching LMPC.casadi)")
    parser.add_argument("--i-j", type=int, default=4, help="Number of past laps in safe set (requires matching LMPC.casadi)")
    parser.add_argument("--N", type=int, default=3, help="LMPC horizon (requires matching LMPC.casadi)")
    parser.add_argument("--quantum", action="store_true", default=False)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable the quantum safe-set augmentation step (keeps baseline behavior).",
    )
    parser.add_argument("--quantum-backend", choices=["statevector", "ibm_runtime"], default="statevector")
    parser.add_argument("--ibm", action="store_true", default=False, help="Shorthand for --quantum-backend ibm_runtime (run on real IBM Q hardware)")
    parser.add_argument("--quantum-horizon", type=int, default=4)
    parser.add_argument("--quantum-start-states", type=int, default=8)
    parser.add_argument("--quantum-samples-per-start", type=int, default=4)
    parser.add_argument("--quantum-min-terminal-v", type=float, default=0.45)
    parser.add_argument("--quantum-min-advance-steps", type=int, default=1)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--plot-out", type=str, default="assets/lmpc_compare.png")
    parser.add_argument("--feasibility-tol", type=float, default=1.05)
    parser.add_argument("--diagnostics", action="store_true", default=False)
    parser.add_argument("--timing", action="store_true", default=False, help="Output detailed timing comparison")
    args = parser.parse_args(argv)

    # --ibm is shorthand for --quantum-backend ibm_runtime
    quantum_backend = str(args.quantum_backend)
    if args.ibm:
        quantum_backend = "ibm_runtime"

    K = int(args.K)
    i_j = int(getattr(args, "i_j"))
    N = int(args.N)
    cfg = DuckietownCompareConfig(n_iterations=int(args.iterations), K=K, i_j=i_j, N=N)
    baseline, quantum_res, traj3, inside_xy, outside_xy, quantum_timing = run_compare(
        cfg=cfg,
        quantum=bool(args.quantum),
        enable_augment=not bool(args.no_augment),
        quantum_backend=quantum_backend,
        quantum_horizon=int(args.quantum_horizon),
        quantum_samples_per_start=int(args.quantum_samples_per_start),
        quantum_start_states=int(args.quantum_start_states),
        quantum_min_terminal_v=float(args.quantum_min_terminal_v),
        quantum_min_advance_steps=int(args.quantum_min_advance_steps),
    )

    print("baseline best lap (s):", baseline.best_lap_seconds)
    if quantum_res is not None:
        print("quantum   best lap (s):", quantum_res.best_lap_seconds)

    if bool(args.diagnostics):
        b_i = baseline.best_loop_index
        print("baseline best loop steps:", baseline.plain_loops[b_i].shape[1])
        b_v = baseline.plain_loops[b_i][3, :]
        print("baseline v (min/mean/max):", float(np.min(b_v)), float(np.mean(b_v)), float(np.max(b_v)))
        if quantum_res is not None:
            q_i = quantum_res.best_loop_index
            print("quantum best loop steps:", quantum_res.plain_loops[q_i].shape[1])
            q_v = quantum_res.plain_loops[q_i][3, :]
            print("quantum v (min/mean/max):", float(np.min(q_v)), float(np.mean(q_v)), float(np.max(q_v)))
            if quantum_res.augment_extra_points:
                print("quantum augment extra points per iter:", quantum_res.augment_extra_points)
                print("quantum augment extra v mean per iter:", quantum_res.augment_extra_v_mean)
                print("quantum augment extra J mean per iter:", quantum_res.augment_extra_J_mean)

    if bool(args.timing):
        print("\n=== TIMING COMPARISON ===")
        # Baseline timing
        baseline_total = sum(baseline.iteration_times)
        baseline_casadi = sum(baseline.casadi_times)
        print(f"Baseline (classical):")
        print(f"  Total iteration time:  {baseline_total:.3f} s")
        print(f"  CasADi solver time:    {baseline_casadi:.3f} s")
        print(f"  Per-iteration mean:    {baseline_total / max(1, len(baseline.iteration_times)):.3f} s")

        if quantum_res is not None:
            quantum_total = sum(quantum_res.iteration_times)
            quantum_casadi = sum(quantum_res.casadi_times)
            print(f"\nQuantum ({quantum_backend}):")
            print(f"  Total iteration time:  {quantum_total:.3f} s")
            print(f"  CasADi solver time:    {quantum_casadi:.3f} s")
            print(f"  Per-iteration mean:    {quantum_total / max(1, len(quantum_res.iteration_times)):.3f} s")

            if quantum_timing:
                print(f"\n  Quantum sampler stats ({quantum_timing.get('backend', 'unknown')}):")
                print(f"    Total sampler time:  {quantum_timing.get('total_seconds', 0):.3f} s")
                print(f"    Number of calls:     {quantum_timing.get('n_calls', 0)}")
                if quantum_timing.get('n_calls', 0) > 0:
                    print(f"    Mean per call:       {quantum_timing.get('mean_seconds', 0):.4f} s")

            # Comparison
            overhead = quantum_total - baseline_total
            print(f"\n  Quantum overhead:      {overhead:+.3f} s ({100 * overhead / max(0.001, baseline_total):+.1f}%)")

    if bool(args.plot):
        from pathlib import Path

        from duckrace.lmpc.plotting import TrackGeometry, plot_laps, track_violation_report
        from duckrace.lmpc.compare import plot_lap_times

        # Best loops (5,T)
        b_loop = baseline.plain_loops[baseline.best_loop_index]
        b_xy = b_loop[:2, :].T

        q_xy = None
        if quantum_res is not None:
            q_loop = quantum_res.plain_loops[quantum_res.best_loop_index]
            q_xy = q_loop[:2, :].T

        geom = TrackGeometry(traj_xytheta=np.asarray(traj3, dtype=float), inside_xy=inside_xy, outside_xy=outside_xy)

        out_path = Path(str(args.plot_out))
        plot_laps(
            out_path=out_path,
            geom=geom,
            baseline_xy=b_xy,
            quantum_xy=q_xy,
            title="LMPC lap paths (best)",
        )
        print("wrote plot:", str(out_path))

        lap_times_out = out_path.with_name(f"{out_path.stem}_lap_times{out_path.suffix}")
        plot_lap_times(out_path=lap_times_out, baseline=baseline, quantum=quantum_res)
        print("wrote plot:", str(lap_times_out))

        b_rep = track_violation_report(loop_xy=b_xy, geom=geom, feasibility_tol=float(args.feasibility_tol))
        print("baseline corridor report:", b_rep)

        if quantum_res is not None:
            q_rep = track_violation_report(loop_xy=q_xy, geom=geom, feasibility_tol=float(args.feasibility_tol))
            print("quantum corridor report:", q_rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
