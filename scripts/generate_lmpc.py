#!/usr/bin/env python3
"""
Generate LMPC.casadi with configurable K and i_j parameters.

Usage:
    python scripts/generate_lmpc.py --K 8 --i-j 4  # default
    python scripts/generate_lmpc.py --K 4 --i-j 2  # smaller (8 neighbors)
    python scripts/generate_lmpc.py --K 2 --i-j 2  # very small (4 neighbors)
"""
import argparse
from pathlib import Path


def generate_lmpc(K: int, i_j: int, N: int = 2, CM: int = 5, output: str = "LMPC.casadi"):
    """Generate LMPC.casadi with specified parameters."""
    from casadi import Opti, floor, if_else, logic_and, pi, sum1, vec

    print(f"Generating LMPC with K={K}, i_j={i_j}, N={N}, CM={CM}")
    print(f"Total neighbors: K*i_j = {K*i_j}")

    opti = Opti()

    x = opti.variable(5, N + 1)  # States (x(0),...,x(N+1))
    u = opti.variable(2, N)  # Inputs (u(0),...,u(N))
    l = opti.variable(K * i_j)  # Lambda of convex hull
    l_margin = opti.variable(CM, N)  # Lambda for convex combination of margin
    x0 = opti.parameter(5, 1)  # Initial state
    D = opti.parameter(5, K * i_j)  # Nearest neighbors
    J = opti.parameter(1, K * i_j)  # Cost-to-go values
    t_to_N = opti.parameter(1, N - 1)  # Time steps
    margins = opti.parameter(2, CM)  # Track margins

    def mod(n, base):
        return n - floor(n / base) * base

    # Objective: minimize cost-to-go
    obj = J @ l

    opti.minimize(obj)

    # Import model_F for dynamics
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import utils
    F = utils.model_F(dt=0.1)  # dt=0.1 for frame_rate=10

    # Constraints
    # 1) System dynamics over prediction horizon
    for k_ in range(0, N):
        opti.subject_to(x[:, k_ + 1] == F(x[:, k_], u[:, k_]))
        opti.subject_to(x[:2, k_ + 1] == margins @ l_margin[:, k_])
    opti.subject_to(sum1(l_margin) == 1)
    opti.subject_to(vec(l_margin) >= 0)

    # 2) Input constraints
    opti.subject_to(vec(u) <= 1)
    opti.subject_to(vec(u) >= 0)

    # 3) Initial state constraint
    opti.subject_to(x[:, 0] == x0)

    # 4) Convex hull constraints
    opti.subject_to(l >= 0)
    opti.subject_to(sum1(l) == 1)
    opti.subject_to(D @ l == x[:, N])

    # Solver options
    opts = dict()
    opts["ipopt.print_level"] = 0
    opts["ipopt.acceptable_constr_viol_tol"] = 1e-5
    opts["print_time"] = False
    opts["verbose"] = False
    opti.solver("ipopt", opts)

    # Create callable function
    M = opti.to_function(
        "M",
        [x0, D, J, t_to_N, margins],
        [u[:, 0], l],
        ["x0", "D", "J", "t_to_N", "central_line"],
        ["u_opt", "lambda"],
    )

    # Save
    M.save(output)
    print(f"Saved: {output}")
    return M


def main():
    parser = argparse.ArgumentParser(description="Generate LMPC.casadi with configurable parameters")
    parser.add_argument("--K", type=int, default=8, help="Number of neighbors per lap")
    parser.add_argument("--i-j", type=int, default=4, help="Number of past laps in safe set")
    parser.add_argument("--N", type=int, default=2, help="LMPC horizon")
    parser.add_argument("--CM", type=int, default=5, help="Number of margin points")
    parser.add_argument("--output", type=str, default="LMPC.casadi", help="Output file")
    args = parser.parse_args()

    generate_lmpc(
        K=args.K,
        i_j=getattr(args, "i_j"),
        N=args.N,
        CM=args.CM,
        output=args.output,
    )


if __name__ == "__main__":
    main()
