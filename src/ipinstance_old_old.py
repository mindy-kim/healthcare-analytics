import heapq
import math
import numpy as np
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
    costOfTest: np.ndarray   # [n]   float cost of each test
    A: np.ndarray            # [n,m] int   A[i,j] = 1 iff test i positive for disease j

    def __init__(self, filename: str) -> None:
        self.load_from_file(filename)
        self.solution = None
        self.objective_value = None

    # ─────────────────────────────────────────────────────────────────────────
    # solve()
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self):
        """
        Minimum-cost disease-discrimination solver via Branch and Bound.

        IP Formulation
        --------------
          Variables : x[i] ∈ {0,1}   (1 = test i is selected)
          Minimize  : Σ cost[i] · x[i]
          s.t.        Σ_{i : A[i,j]≠A[i,k]} x[i] ≥ 1   ∀ disease pairs (j<k)

        Optimizations
        -------------
        1. PERSISTENT GLOP MODEL — one pywraplp model built once, reused for
           every LP solve with warm-start from prior dual-simplex basis.

        2. CONSTRAINT RELAXATION — when a test is fixed to 1, satisfied pair
           constraints have their lower bound relaxed to 0.

        3. PREPROCESSING (forced selections) — single-option pairs force their
           test to 1 before BnB starts.

        4. DOMINATION PRUNING — if test i covers a superset of what test j covers
           at equal-or-lower cost, j can never appear in an optimal solution and
           is permanently eliminated before BnB.

        5. NUMPY VECTORISATION — pairwise column comparisons and greedy scoring.

        6. GREEDY RESTARTS — multiple randomised greedy runs for a tighter initial
           upper bound, pruning more of the tree from the very first node.

        7. COST-WEIGHTED BRANCHING — branch on the fractional variable that covers
           the most uncovered active pairs per unit cost, not just the most
           fractional variable.

        8. REDUCED-COST FIXING — after each LP solve, variables whose reduced cost
           exceeds the remaining gap are fixed to 0 without branching.

        9. HYBRID NODE SELECTION — depth is used as a tiebreaker so equal-bound
           nodes are explored depth-first (finds incumbents faster, tightens bound
           sooner) while the heap still globally favours the best lower bound.

        10. MIN-HEAP (best-first search) — O(log n) node selection via heapq with
            tie-breaking counter so Python never compares lists.
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
        # Must be computed on the FULL test_covers so dominated tests are found
        # correctly before the active-pair subset is computed.
        dominated = self._domination_pruning(test_covers, costs, n, forced)

        # Active pairs: not already covered by forced tests
        active_pairs = [disc_pairs[p] for p in range(num_pairs) if p not in covered_pairs]

        # Remove dominated tests from active pairs — they cannot appear in BnB.
        # If a pair becomes empty after removal the problem is infeasible (should
        # not happen if domination is computed correctly, but guard anyway).
        active_pairs = [
            [i for i in disc if i not in dominated]
            for disc in active_pairs
        ]
        if any(len(disc) == 0 for disc in active_pairs):
            self.solution        = None
            self.objective_value = None
            return None, None

        # Inverse map: for each test, which active pairs does it cover?
        active_covers = [[] for _ in range(n)]
        for p, disc in enumerate(active_pairs):
            for i in disc:
                active_covers[i].append(p)
        num_active = len(active_pairs)

        # ── 4. Greedy upper bound (multiple restarts) ─────────────────────────
        best_sol, best_obj = self._greedy_upper_bound(
            costs, disc_pairs, test_covers, n, num_restarts=15
        )

        # ── 5. Build ONE persistent GLOP model ────────────────────────────────
        lp = pywraplp.Solver.CreateSolver("GLOP")
        lp.SuppressOutput()

        x = [lp.NumVar(0.0, 1.0, f"x{i}") for i in range(n)]

        # Fix forced tests to 1 and dominated tests to 0 permanently
        for i in forced:
            x[i].SetBounds(1.0, 1.0)
        for i in dominated:
            x[i].SetBounds(0.0, 0.0)

        lp_obj = lp.Objective()
        for i in range(n):
            lp_obj.SetCoefficient(x[i], float(costs[i]))
        lp_obj.SetMinimization()

        ctrs = []
        for disc in active_pairs:
            ct = lp.Constraint(1.0, lp.infinity())
            for i in disc:
                ct.SetCoefficient(x[i], 1.0)
            ctrs.append(ct)

        # satisfied_by[p] = how many currently-fixed-to-1 tests cover active pair p.
        satisfied_by = [0] * num_active

        for i in forced:
            for p in active_covers[i]:
                if satisfied_by[p] == 0:
                    ctrs[p].SetBounds(0.0, lp.infinity())
                satisfied_by[p] += 1

        prev_fixings = []

        def apply_and_solve(new_fixings):
            nonlocal prev_fixings

            # Reset previous node's bounds
            for idx, val in prev_fixings:
                x[idx].SetBounds(0.0, 1.0)
                if val == 1:
                    for p in active_covers[idx]:
                        satisfied_by[p] -= 1
                        if satisfied_by[p] == 0:
                            ctrs[p].SetBounds(1.0, lp.infinity())

            # Apply new node's bounds
            for idx, val in new_fixings:
                fv = float(val)
                x[idx].SetBounds(fv, fv)
                if val == 1:
                    for p in active_covers[idx]:
                        if satisfied_by[p] == 0:
                            ctrs[p].SetBounds(0.0, lp.infinity())
                        satisfied_by[p] += 1

            prev_fixings = new_fixings

            status = lp.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                lp_val = lp.Objective().Value()
                sol    = [x[i].solution_value() for i in range(n)]
                rc     = [x[i].reduced_cost()   for i in range(n)]
                return lp_val, sol, rc
            return math.inf, None, None

        # ── 6. Root LP ────────────────────────────────────────────────────────
        root_obj_val, root_sol, root_rc = apply_and_solve([])

        if root_sol is None:
            self.solution        = None
            self.objective_value = None
            return None, None

        if root_obj_val >= best_obj - 1e-6:
            self.solution        = best_sol
            self.objective_value = best_obj
            return best_sol, best_obj

        # ── 7. Branch-and-Bound loop ──────────────────────────────────────────
        EPS     = 1e-6
        counter = 0
        # Heap entry: (lp_bound, counter, neg_depth, fixings, lp_sol, lp_rc)
        # neg_depth as tiebreaker → deeper nodes explored first when bounds tie
        # (depth-first dive finds incumbents faster, tightens bound sooner)
        heap  = [(root_obj_val, counter, 0, [], root_sol, root_rc)]
        free  = set(range(n)) - forced - dominated

        while heap:
            lp_bound, _, neg_depth, fixings, lp_sol, lp_rc = heapq.heappop(heap)
            depth = -neg_depth

            if lp_bound >= best_obj - EPS:
                continue

            # ── Reduced-cost fixing ───────────────────────────────────────────
            # Variables whose reduced cost alone exceeds the remaining gap can be
            # fixed to 0 at this node without branching — they can never improve
            # the objective enough to justify being selected.
            gap = best_obj - lp_bound
            rc_fixes = {
                i for i in free - {fv[0] for fv in fixings}
                if lp_rc[i] > gap + EPS
            }

            branch_i = self._select_branch_var(
                lp_sol, fixings, free, active_covers, costs, rc_fixes
            )

            if branch_i == -1:
                # LP solution is integral → valid IP solution
                obj = float(np.dot(costs, lp_sol))
                if obj < best_obj - EPS:
                    best_obj = obj
                    best_sol = lp_sol[:]
                continue

            fixed_to_zero = {i for i, v in fixings if v == 0} | rc_fixes

            for fix_val in (1, 0):
                if fix_val == 0:
                    fixed_to_zero.add(branch_i)
                    if self._quick_infeasible(active_pairs, fixed_to_zero):
                        fixed_to_zero.discard(branch_i)
                        continue
                    fixed_to_zero.discard(branch_i)

                # Incorporate reduced-cost fixes as explicit 0-fixings so the LP
                # model respects them at the child node.
                new_fixings = (
                    fixings
                    + [(i, 0) for i in rc_fixes]
                    + [(branch_i, fix_val)]
                )
                child_obj, child_sol, child_rc = apply_and_solve(new_fixings)

                if child_sol is not None and child_obj < best_obj - EPS:
                    counter += 1
                    heapq.heappush(heap, (
                        child_obj, counter, -(depth + 1),
                        new_fixings, child_sol, child_rc
                    ))

        self.solution        = best_sol
        self.objective_value = best_obj if best_sol is not None else None
        return self.solution, self.objective_value

    # ─────────────────────────────────────────────────────────────────────────
    # Discrimination structure
    # ─────────────────────────────────────────────────────────────────────────

    def _build_discrimination(self, n, m):
        """
        For every disease pair (j < k) find which tests distinguish them.
        Uses numpy column comparisons instead of a Python element loop.
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
        Returns forced (set of test indices fixed to 1) and covered (set of
        pair indices already satisfied).
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
        If test i covers a superset of the pairs covered by test j and
        cost[i] <= cost[j], then j is dominated and can never appear in an
        optimal solution.  Dominated tests are permanently fixed to 0.

        Forced tests are excluded from elimination since they must be 1.
        """
        eliminated  = set()
        covers_set  = [frozenset(test_covers[i]) for i in range(n)]

        for i in range(n):
            if i in eliminated or i in forced or not covers_set[i]:
                continue
            for j in range(n):
                if (
                    i == j
                    or j in eliminated
                    or j in forced
                    or not covers_set[j]
                ):
                    continue
                # j is dominated by i: i covers everything j does at lower/equal cost
                if costs[i] <= costs[j] and covers_set[j].issubset(covers_set[i]):
                    eliminated.add(j)

        return eliminated

    # ─────────────────────────────────────────────────────────────────────────
    # Greedy heuristic — multiple randomised restarts
    # ─────────────────────────────────────────────────────────────────────────

    def _greedy_upper_bound(self, costs, disc_pairs, test_covers, n, num_restarts=15):
        """
        Set-cover greedy with randomised cost perturbation on each restart.
        The best feasible solution across all restarts is returned.

        Using a noisy cost function on each restart diversifies which tests get
        selected first, escaping local optima of the deterministic greedy and
        yielding a tighter initial upper bound for BnB pruning.
        """
        best_sol = None
        best_obj = math.inf

        rng = np.random.default_rng(42)

        for r in range(num_restarts):
            if r == 0:
                perturbed = costs                          # first run: exact costs
            else:
                noise     = rng.uniform(0.8, 1.2, size=n) # ±20 % random perturbation
                perturbed = costs * noise

            sol, obj = self._greedy_single(perturbed, costs, disc_pairs, test_covers, n)
            if sol is not None and obj < best_obj:
                best_sol, best_obj = sol, obj

        return best_sol, best_obj

    def _greedy_single(self, perturbed_costs, true_costs, disc_pairs, test_covers, n):
        """
        One greedy run.  Selection order uses perturbed_costs (for diversity)
        but the returned objective is computed with true_costs.
        """
        num_pairs   = len(disc_pairs)
        covered     = np.zeros(num_pairs, dtype=bool)
        selected    = np.zeros(n,         dtype=float)
        total_cost  = 0.0
        uncov_count = np.array([len(test_covers[i]) for i in range(n)], dtype=float)

        while not covered.all():
            mask   = (selected < 0.5) & (uncov_count > 0)
            if not mask.any():
                break
            scores  = np.where(mask, uncov_count / perturbed_costs, -np.inf)
            best_i  = int(np.argmax(scores))
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
    # BnB helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _select_branch_var(self, lp_sol, fixings, free, active_covers, costs, rc_fixes):
        """
        Cost-weighted coverage branching: among free, non-fixed, non-rc-fixed
        fractional variables, prefer the one with the highest ratio of
        (active pairs covered × fractionality) / cost.

        This biases branching toward cheap tests that resolve many unsatisfied
        constraints — far more informative than pure most-fractional selection
        for set-cover IPs.

        Returns -1 if all candidates are already integral.
        """
        EPS       = 1e-6
        fixed     = {i for i, _ in fixings} | rc_fixes
        candidates = free - fixed

        best_i     = -1
        best_score = -1.0

        for i in candidates:
            v    = lp_sol[i]
            frac = min(v, 1.0 - v)
            if frac <= EPS:
                continue
            # Pairs covered by this test weighted by fractionality and inverse cost
            coverage = len(active_covers[i])
            score    = frac * coverage / max(costs[i], EPS)
            if score > best_score:
                best_score = score
                best_i     = i

        return best_i

    def _quick_infeasible(self, active_pairs, fixed_to_zero):
        """
        Returns True if any active disease pair has ALL distinguishing tests
        in fixed_to_zero — making the LP trivially infeasible.
        """
        for disc in active_pairs:
            if all(i in fixed_to_zero for i in disc):
                return True
        return False

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