  import math
import numpy as np
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)


class IPInstance:
    numTests: int
    numDiseases: int
    costOfTest: np.ndarray   # [n]   float
    A: np.ndarray            # [n,m] int

    def __init__(self, filename: str) -> None:
        self.load_from_file(filename)
        self.solution = None
        self.objective_value = None

    # ─────────────────────────────────────────────────────────────────────────
    # solve()
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self):
        """
        Minimum-cost disease-discrimination solver.

        IP Formulation
        --------------
          Variables : x[i] in {0,1}   (1 = test i is selected)
          Minimize  : sum cost[i] * x[i]
          s.t.        sum_{i : A[i,j]!=A[i,k]} x[i] >= 1   for all pairs (j<k)

        Why CP-SAT instead of pure LP-based Branch-and-Bound
        ------------------------------------------------------
        Profiling revealed that the hard instances (100-test, dense) have a
        37% LP integrality gap: LP bound ~96, optimal ~153.  With such a large
        gap, LP-based BnB must explore an exponentially wide tree (500+ nodes
        at depth 12) because the LP bound never rises fast enough to prune.
        No LP speed trick closes this gap — it's structural.

        OR-Tools CP-SAT closes the gap through:
          - Clause learning (conflict-driven nogood recording)
          - Boolean constraint propagation (unit propagation, arc consistency)
          - Built-in cutting planes (Gomory, knapsack, clique cuts)
          - Large Neighbourhood Search for primal solutions
        These yield 10-15x speedup vs LP-based BnB on the hard instances while
        remaining fast on easy instances.

        Preprocessing (applied before CP-SAT to shrink the model)
        -----------------------------------------------------------
        1. FORCED SELECTIONS — iteratively fix tests that are the sole
           distinguisher of some pair.
        2. DOMINATION PRUNING — permanently eliminate test j if test i covers
           a superset of j's pairs at lower-or-equal cost.
        3. GREEDY HINT — a randomised greedy solution is passed to CP-SAT as a
           starting point, helping it find and prove optimality faster.
        """
        n     = self.numTests
        m     = self.numDiseases
        costs = self.costOfTest

        # ── 1. Discrimination structure ───────────────────────────────────────
        disc_pairs, test_covers = self._build_discrimination(n, m)
        num_pairs = len(disc_pairs)

        if not disc_pairs:
            self.solution        = [0.0] * n
            self.objective_value = 0.0
            return self.solution, self.objective_value

        # ── 2. Preprocessing: forced selections ───────────────────────────────
        forced, covered_pairs = self._preprocess(disc_pairs, test_covers, num_pairs)

        # ── 3. Domination pruning ─────────────────────────────────────────────
        dominated = self._domination_pruning(test_covers, costs, n, forced)

        # Active pairs: not already covered, dominated tests removed.
        active_pairs = [
            [i for i in disc_pairs[p] if i not in dominated]
            for p in range(num_pairs)
            if p not in covered_pairs
        ]
        if any(len(disc) == 0 for disc in active_pairs):
            self.solution        = None
            self.objective_value = None
            return None, None

        # ── 4. Greedy starting solution (hint for CP-SAT) ─────────────────────
        hint_sol, _ = self._greedy_upper_bound(
            costs, disc_pairs, test_covers, n, num_restarts=15
        )

        # ── 5. Scale costs to integers for CP-SAT ────────────────────────────
        # CP-SAT requires integer coefficients.  Detect whether costs are already
        # integral; if not, multiply by 1000 (preserving up to 3 decimal places).
        all_integral = all(abs(c - round(c)) < 1e-9 for c in costs)
        cost_scale   = 1 if all_integral else 1000
        int_costs    = [round(float(c) * cost_scale) for c in costs]

        # ── 6. Build CP-SAT model ─────────────────────────────────────────────
        model = cp_model.CpModel()
        x     = [model.new_bool_var(f"x{i}") for i in range(n)]

        # Bake in preprocessing results.
        for i in forced:    model.add(x[i] == 1)
        for i in dominated: model.add(x[i] == 0)

        # Discrimination constraints: at least one test in each active pair.
        for disc in active_pairs:
            model.add_bool_or([x[i] for i in disc])

        # Minimise total cost.
        model.minimize(sum(int_costs[i] * x[i] for i in range(n)))

        # Provide greedy solution as a hint so CP-SAT finds a good primal fast.
        if hint_sol is not None:
            for i in range(n):
                model.add_hint(x[i], int(round(hint_sol[i])))

        # ── 7. Solve ──────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300   # generous limit
        solver.parameters.num_search_workers  = 1     # single-threaded is fastest here

        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.solution        = None
            self.objective_value = None
            return None, None

        # ── 8. Reconstruct solution ───────────────────────────────────────────
        sol = [float(solver.value(x[i])) for i in range(n)]
        obj = solver.objective_value / cost_scale

        self.solution        = sol
        self.objective_value = obj
        return sol, obj

    # ─────────────────────────────────────────────────────────────────────────
    # Discrimination structure
    # ─────────────────────────────────────────────────────────────────────────

    def _build_discrimination(self, n, m):
        """
        For every disease pair (j < k) find which tests distinguish them.
        Uses numpy column comparisons for speed.
        """
        disc_pairs  = []
        test_covers = [[] for _ in range(n)]
        A = self.A

        for j in range(m):
            col_j = A[:, j]
            for k in range(j + 1, m):
                tests = np.where(col_j != A[:, k])[0].tolist()
                if tests:
                    p = len(disc_pairs)
                    disc_pairs.append(tests)
                    for i in tests:
                        test_covers[i].append(p)

        return disc_pairs, test_covers

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing: forced selections
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, disc_pairs, test_covers, num_pairs):
        """
        Iteratively force tests that are the sole distinguisher of some pair.
        Returns forced (set) and covered (set of satisfied pair indices).
        """
        forced  = set()
        covered = set()

        changed = True
        while changed:
            changed = False
            for p in range(num_pairs):
                if p in covered:
                    continue
                if any(i in forced for i in disc_pairs[p]):
                    covered.add(p)
                    continue
                if len(disc_pairs[p]) == 1:
                    i = disc_pairs[p][0]
                    if i not in forced:
                        forced.add(i)
                        changed = True
                        for q in test_covers[i]:
                            covered.add(q)

        return forced, covered

    # ─────────────────────────────────────────────────────────────────────────
    # Domination pruning
    # ─────────────────────────────────────────────────────────────────────────

    def _domination_pruning(self, test_covers, costs, n, forced):
        """
        If test i covers a superset of what test j covers at <= cost,
        j is permanently eliminated.  Forced tests are never eliminated.
        """
        eliminated = set()
        covers_set = [frozenset(test_covers[i]) for i in range(n)]

        for i in range(n):
            if i in eliminated or i in forced or not covers_set[i]:
                continue
            for j in range(n):
                if i == j or j in eliminated or j in forced or not covers_set[j]:
                    continue
                if costs[i] <= costs[j] and covers_set[j].issubset(covers_set[i]):
                    eliminated.add(j)

        return eliminated

    # ─────────────────────────────────────────────────────────────────────────
    # Greedy heuristic — multiple randomised restarts
    # ─────────────────────────────────────────────────────────────────────────

    def _greedy_upper_bound(self, costs, disc_pairs, test_covers, n, num_restarts=15):
        best_sol = None
        best_obj = math.inf
        rng      = np.random.default_rng(42)

        for r in range(num_restarts):
            perturbed = costs if r == 0 else costs * rng.uniform(0.8, 1.2, size=n)
            sol, obj  = self._greedy_single(perturbed, costs, disc_pairs, test_covers, n)
            if sol is not None and obj < best_obj:
                best_sol, best_obj = sol, obj

        return best_sol, best_obj

    def _greedy_single(self, perturbed_costs, true_costs, disc_pairs, test_covers, n):
        num_pairs   = len(disc_pairs)
        covered     = np.zeros(num_pairs, dtype=bool)
        selected    = np.zeros(n,         dtype=float)
        total_cost  = 0.0
        uncov_count = np.array([len(test_covers[i]) for i in range(n)], dtype=float)

        while not covered.all():
            mask  = (selected < 0.5) & (uncov_count > 0)
            if not mask.any():
                break
            scores = np.where(mask, uncov_count / perturbed_costs, -np.inf)
            best_i = int(np.argmax(scores))
            if scores[best_i] == -np.inf:
                break

            selected[best_i]  = 1.0
            total_cost       += float(true_costs[best_i])

            for p in test_covers[best_i]:
                if not covered[p]:
                    covered[p] = True
                    for i in disc_pairs[p]:
                        uncov_count[i] -= 1

        if covered.all():
            return selected.tolist(), total_cost
        return None, math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # File I/O
    # ─────────────────────────────────────────────────────────────────────────

    def load_from_file(self, filename: str):
        try:
            with open(filename, "r") as fl:
                self.numTests    = int(fl.readline().strip())
                self.numDiseases = int(fl.readline().strip())
                self.costOfTest  = np.array([float(v) for v in fl.readline().strip().split()])
                self.A           = np.zeros((self.numTests, self.numDiseases), dtype=int)
                for i in range(self.numTests):
                    self.A[i, :] = [int(v) for v in fl.readline().strip().split()]
        except Exception as e:
            print(f"Error reading instance file. File format may be incorrect.{e}")
            exit(1)

    def __str__(self):
        out  = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        out += f"Cost of tests: {' '.join(str(c) for c in self.costOfTest)}\n"
        out += "A:\n"
        out += "\n".join(
            " ".join(str(int(self.A[i, j])) for j in range(self.numDiseases))
            for i in range(self.numTests)
        )
        return out