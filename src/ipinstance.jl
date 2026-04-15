using JuMP
import HiGHS


#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)

mutable struct IPInstance
    numTests::Int                        # n: number of tests
    numDiseases::Int                     # m: number of diseases
    costOfTest::Vector{Float64}          # cost[i]: cost of performing test i
    A::Matrix{Int}                       # A[i,j] = 1 iff test i is positive for disease j
    solution::Union{Vector{Float64}, Nothing}
    objective_value::Union{Float64, Nothing}

    function IPInstance(filename::String)
        inst = new()
        load_from_file!(inst, filename)
        inst.solution = nothing
        inst.objective_value = nothing
        return inst
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Discrimination Structure
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_discrimination(A, n, m) -> (disc_pairs, test_covers)

Precompute for every disease pair (j < k) which tests can distinguish them,
and for every test i which disease pairs it covers.

- disc_pairs[p] : indices of tests that distinguish the p-th disease pair
- test_covers[i]: indices of disease pairs that test i can distinguish
"""
function build_discrimination(A::Matrix{Int}, n::Int, m::Int)
    disc_pairs  = Vector{Vector{Int}}()
    test_covers = [Vector{Int}() for _ in 1:n]

    for j in 1:m, k in (j+1):m
        tests = [i for i in 1:n if A[i, j] != A[i, k]]
        if !isempty(tests)
            p = length(disc_pairs) + 1
            push!(disc_pairs, tests)
            for i in tests
                push!(test_covers[i], p)
            end
        end
    end

    return disc_pairs, test_covers
end

# ─────────────────────────────────────────────────────────────────────────────
# Greedy Heuristic (Initial Upper Bound)
# ─────────────────────────────────────────────────────────────────────────────

"""
    greedy_upper_bound(costs, disc_pairs, test_covers, n)

Set-cover greedy: iteratively add the test with the best
coverage-per-unit-cost ratio until all disease pairs are distinguished.

Returns (solution_vector, total_cost) or (nothing, Inf) if infeasible.
"""
function greedy_upper_bound(costs::Vector{Float64},
                            disc_pairs::Vector{Vector{Int}},
                            test_covers::Vector{Vector{Int}},
                            n::Int)
    num_pairs   = length(disc_pairs)
    covered     = falses(num_pairs)
    selected    = zeros(Float64, n)
    total_cost  = 0.0

    # uncov_count[i] = number of currently uncovered pairs test i can distinguish
    uncov_count = [length(test_covers[i]) for i in 1:n]

    while !all(covered)
        best_i     = -1
        best_score = -Inf
        for i in 1:n
            (selected[i] > 0.5 || uncov_count[i] == 0) && continue
            score = uncov_count[i] / costs[i]   # coverage per unit cost
            if score > best_score
                best_score = score
                best_i     = i
            end
        end
        best_i == -1 && break   # no improving test found → infeasible

        selected[best_i]  = 1.0
        total_cost       += costs[best_i]

        for p in test_covers[best_i]
            covered[p] && continue
            covered[p] = true
            # decrement uncov_count for every test that covered this pair
            for i in disc_pairs[p]
                uncov_count[i] -= 1
            end
        end
    end

    return all(covered) ? (selected, total_cost) : (nothing, Inf)
end

# ─────────────────────────────────────────────────────────────────────────────
# LP Relaxation
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_lp(costs, disc_pairs, lb, ub) -> (obj, x_vals)

Solve the LP relaxation of the IP at a BnB node.
Variable bounds lb[i] ≤ x[i] ≤ ub[i] encode fixed branching decisions
(lb[i] = ub[i] = 0 → test excluded; lb[i] = ub[i] = 1 → test included).

IP Formulation:
  min  Σ cost[i] · x[i]
  s.t. Σ_{i ∈ disc_pairs[p]} x[i] ≥ 1   ∀ disease pairs p   (discrimination)
       lb[i] ≤ x[i] ≤ ub[i]              ∀ tests i

Returns (Inf, nothing) if infeasible.
"""
function solve_lp(costs::Vector{Float64},
                  disc_pairs::Vector{Vector{Int}},
                  lb::Vector{Float64},
                  ub::Vector{Float64})
    n     = length(costs)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "time_limit", 30.0)

    @variable(model, lb[i] <= x[i=1:n] <= ub[i])
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:n))
    for disc in disc_pairs
        @constraint(model, sum(x[i] for i in disc) >= 1)
    end

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x)
    end
    return Inf, nothing   # infeasible or other failure
end

# ─────────────────────────────────────────────────────────────────────────────
# BnB Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    select_branch_var(lp_sol, lb, ub) -> index or -1

Most-fractional variable selection: pick the free variable whose LP value
is furthest from any integer (closest to 0.5).  Returns -1 if the solution
is already integral (all free variables within EPS of 0 or 1).
"""
function select_branch_var(lp_sol::Vector{Float64},
                           lb::Vector{Float64},
                           ub::Vector{Float64})
    EPS       = 1e-6
    best_i    = -1
    best_frac = EPS
    for i in eachindex(lp_sol)
        ub[i] - lb[i] < 0.5 && continue    # variable is fixed — skip
        v    = lp_sol[i]
        frac = min(v - floor(v), ceil(v) - v)
        if frac > best_frac
            best_frac = frac
            best_i    = i
        end
    end
    return best_i
end

"""
    quick_infeasible(disc_pairs, ub) -> Bool

Fast pre-check: if any disease pair has NO test with ub[i] > 0,
the sub-problem is infeasible and we can skip the LP solve.
"""
function quick_infeasible(disc_pairs::Vector{Vector{Int}}, ub::Vector{Float64})
    for disc in disc_pairs
        any(i -> ub[i] > 0.5, disc) || return true
    end
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# Branch and Bound Solver
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve!(inst::IPInstance) -> (solution, objective_value)

Branch-and-Bound solver for the minimum-cost test-selection IP.

Algorithm outline:
  1. Build discrimination structure (precompute which tests separate each disease pair).
  2. Obtain an initial upper bound via a greedy set-cover heuristic.
  3. Solve the root LP relaxation.
  4. BnB loop (best-first search, most-fractional branching):
       a. Pop the open node with the smallest LP lower bound.
       b. Prune if LP bound ≥ current best (optimality gap closed).
       c. If LP solution is integral → update best integer solution.
       d. Otherwise branch on the most-fractional variable:
            • x[i] = 1 : forces test i to be selected
            • x[i] = 0 : forces test i to be excluded
          Solve LP at each child; add to queue only if LP bound < best.
  5. Return the optimal solution and its cost once the queue is empty.
"""
function solve!(inst::IPInstance)
    n    = inst.numTests
    m    = inst.numDiseases
    costs = inst.costOfTest

    # ── Step 1: Precompute discrimination structure ───────────────────────────
    disc_pairs, test_covers = build_discrimination(inst.A, n, m)
    num_pairs = length(disc_pairs)

    if num_pairs == 0
        # All diseases are identical; zero-cost solution is trivially optimal
        inst.solution        = zeros(Float64, n)
        inst.objective_value = 0.0
        return inst.solution, inst.objective_value
    end

    # ── Step 2: Greedy initial upper bound ────────────────────────────────────
    best_sol, best_obj = greedy_upper_bound(costs, disc_pairs, test_covers, n)

    # ── Step 3: Root LP relaxation ────────────────────────────────────────────
    lb0 = zeros(Float64, n)
    ub0 = ones(Float64, n)

    root_lp_obj, root_lp_sol = solve_lp(costs, disc_pairs, lb0, ub0)

    if root_lp_sol === nothing
        # Problem is infeasible (should not happen on valid instances)
        inst.solution        = nothing
        inst.objective_value = nothing
        return nothing, nothing
    end

    # If the LP lower bound already meets the greedy upper bound, done
    if root_lp_obj >= best_obj - 1e-6
        inst.solution        = best_sol
        inst.objective_value = best_obj
        return best_sol, best_obj
    end

    # ── Step 4: BnB with best-first search ───────────────────────────────────
    # Each node: (lp_bound, lp_sol, lb, ub)
    # We keep nodes in a simple vector and always pop the one with the
    # smallest LP bound (best-first).  This gives tight pruning at the cost
    # of O(|nodes|) scans — acceptable for the instance sizes here.
    nodes = [(root_lp_obj, root_lp_sol, lb0, ub0)]

    nodes_explored = 0
    EPS = 1e-6

    while !isempty(nodes)

        # ── Select best node ──────────────────────────────────────────────────
        best_idx = argmin(map(nd -> nd[1], nodes))
        lp_bound, lp_sol, lb, ub = nodes[best_idx]
        deleteat!(nodes, best_idx)
        nodes_explored += 1

        # ── Prune by bound ────────────────────────────────────────────────────
        lp_bound >= best_obj - EPS && continue

        # ── Check integrality ─────────────────────────────────────────────────
        branch_i = select_branch_var(lp_sol, lb, ub)

        if branch_i == -1
            # LP solution is integral → feasible IP solution
            obj = sum(costs[i] * lp_sol[i] for i in eachindex(costs))
            if obj < best_obj - EPS
                best_obj = obj
                best_sol = copy(lp_sol)
                # Discard nodes that can no longer beat the new best
                filter!(nd -> nd[1] < best_obj - EPS, nodes)
            end
            continue
        end

        # ── Branch on x[branch_i] ─────────────────────────────────────────────
        # Try x = 1 first (tends to produce tighter subtrees for set-cover IPs)
        for (fix_lb, fix_ub) in ((1.0, 1.0), (0.0, 0.0))
            new_lb = copy(lb)
            new_ub = copy(ub)
            new_lb[branch_i] = fix_lb
            new_ub[branch_i] = fix_ub

            # Quick feasibility pre-check before paying for an LP solve
            quick_infeasible(disc_pairs, new_ub) && continue

            child_obj, child_sol = solve_lp(costs, disc_pairs, new_lb, new_ub)
            if child_sol !== nothing && child_obj < best_obj - EPS
                push!(nodes, (child_obj, child_sol, new_lb, new_ub))
            end
        end
    end

    inst.solution        = best_sol
    inst.objective_value = isnothing(best_sol) ? nothing : best_obj
    return inst.solution, inst.objective_value
end

# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

function load_from_file!(inst::IPInstance, filename::String)
    try
        open(filename, "r") do fl
            inst.numTests    = parse(Int, strip(readline(fl)))
            inst.numDiseases = parse(Int, strip(readline(fl)))
            inst.costOfTest  = parse.(Float64, split(strip(readline(fl))))

            inst.A = zeros(Int, inst.numTests, inst.numDiseases)
            for i in 1:inst.numTests
                inst.A[i, :] = parse.(Int, split(strip(readline(fl))))
            end
        end
    catch e
        println("Error reading instance file. File format may be incorrect. $e")
        exit(1)
    end
end

function Base.show(io::IO, inst::IPInstance)
    println(io, "Number of tests: $(inst.numTests)")
    println(io, "Number of diseases: $(inst.numDiseases)")
    println(io, "Cost of tests: $(inst.costOfTest)")
    println(io, "A:")
    for i in 1:inst.numTests
        println(io, inst.A[i, :])
    end
end