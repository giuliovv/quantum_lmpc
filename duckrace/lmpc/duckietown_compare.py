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
    N: int = 2
    K: int = 8
    i_j: int = 4
    i_j_all: bool = False
    more: int = 20
    n_iterations: int = 5


def _build_env(frame_rate: int):
    try:
        from gym_duckietown.simulator import Simulator
    except Exception as e:  # pragma: no cover
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
    quantum_backend: str,
    quantum_horizon: int,
    quantum_samples_per_start: int,
    quantum_start_states: int,
):
    import utils
    from casadi import DM, Function, vertcat
    from scipy import spatial

    from duckrace.lmpc import LMPCKit, LMPCRunConfig, compare_lmpc_baseline_vs_quantum
    from duckrace.quantum.lmpc_augment import QuantumLMPCAugmenterConfig
    from duckrace.quantum.sampler import QuantumSamplerConfig

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

    # Compare LMPC iterations
    M_lmpc = Function.load("LMPC.casadi")

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
        return baseline, None, traj3, np.asarray(inside, dtype=float), np.asarray(outside, dtype=float)

    qcfg = QuantumLMPCAugmenterConfig(
        horizon=int(quantum_horizon),
        n_start_states=int(quantum_start_states),
        n_samples_per_start=int(quantum_samples_per_start),
        sampler=QuantumSamplerConfig(backend=quantum_backend, n_iterations=1, seed=0),
    )

    baseline, quantum_res = compare_lmpc_baseline_vs_quantum(
        kit=kit,
        run_cfg=run_cfg,
        n_iterations=int(cfg.n_iterations),
        quantum_cfg=qcfg,
    )
    return baseline, quantum_res, traj3, np.asarray(inside, dtype=float), np.asarray(outside, dtype=float)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckrace.lmpc.duckietown_compare",
        description="Compare baseline LMPC vs quantum-augmented LMPC on Duckietown (ETH_large_loop).",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--quantum", action="store_true", default=False)
    parser.add_argument("--quantum-backend", choices=["statevector", "ibm_runtime"], default="statevector")
    parser.add_argument("--quantum-horizon", type=int, default=4)
    parser.add_argument("--quantum-start-states", type=int, default=8)
    parser.add_argument("--quantum-samples-per-start", type=int, default=4)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--plot-out", type=str, default="assets/lmpc_compare.png")
    parser.add_argument("--feasibility-tol", type=float, default=1.05)
    parser.add_argument("--diagnostics", action="store_true", default=False)
    args = parser.parse_args(argv)

    cfg = DuckietownCompareConfig(n_iterations=int(args.iterations))
    baseline, quantum_res, traj3, inside_xy, outside_xy = run_compare(
        cfg=cfg,
        quantum=bool(args.quantum),
        quantum_backend=str(args.quantum_backend),
        quantum_horizon=int(args.quantum_horizon),
        quantum_samples_per_start=int(args.quantum_samples_per_start),
        quantum_start_states=int(args.quantum_start_states),
    )

    print("baseline best lap (s):", baseline.best_lap_seconds)
    if quantum_res is not None:
        print("quantum   best lap (s):", quantum_res.best_lap_seconds)

    if bool(args.diagnostics):
        b_i = baseline.best_loop_index
        print("baseline best loop steps:", baseline.plain_loops[b_i].shape[1])
        if quantum_res is not None:
            q_i = quantum_res.best_loop_index
            print("quantum best loop steps:", quantum_res.plain_loops[q_i].shape[1])

    if bool(args.plot):
        from pathlib import Path

        from duckrace.lmpc.plotting import TrackGeometry, plot_laps, track_violation_report

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

        b_rep = track_violation_report(loop_xy=b_xy, geom=geom, feasibility_tol=float(args.feasibility_tol))
        print("baseline corridor report:", b_rep)

        if quantum_res is not None:
            q_rep = track_violation_report(loop_xy=q_xy, geom=geom, feasibility_tol=float(args.feasibility_tol))
            print("quantum corridor report:", q_rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
