#!/usr/bin/env python3

# validate.py
#
# Usage: python3 validate.py <results.log> <input_folder/>
#
# Expects each line of results.log to be a JSON object with fields:
#   "Instance"       -> filename of the .ip file
#   "Result"         -> reported objective value (number or "--")
#   "Solution"       -> "OPT" or "--"
#   "SolutionVector" -> list of 0/1 values (one per test)

import json
import sys
import os

EPS = 1e-4

def load_instance(filepath):
    with open(filepath) as f:
        n = int(f.readline().strip())
        m = int(f.readline().strip())
        costs = list(map(float, f.readline().strip().split()))
        A = []
        for _ in range(n):
            A.append(list(map(int, f.readline().strip().split())))
    return n, m, costs, A

def validate(n, m, costs, A, x, reported_obj):
    # 1. Check vector length
    if len(x) != n:
        return False, f"SolutionVector length {len(x)} ≠ numTests {n}"

    # 2. Check entries are binary
    for i, v in enumerate(x):
        if not (abs(v) < EPS or abs(v - 1) < EPS):
            return False, f"x[{i}] = {v} is not 0 or 1"

    selected = [i for i in range(n) if x[i] > 0.5]

    # 3. Verify reported cost
    actual_cost = sum(costs[i] for i in selected)
    if abs(actual_cost - reported_obj) > EPS * max(1.0, abs(reported_obj)):
        return False, f"Reported cost {reported_obj} ≠ actual cost {actual_cost:.6f}"

    # 4. Check every disease pair is distinguished
    for j in range(m):
        for k in range(j + 1, m):
            if not any(A[i][j] != A[i][k] for i in selected):
                return False, f"Disease pair ({j+1}, {k+1}) is NOT distinguished"

    return True, ""

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 validate.py <results.log> <input_folder/>")
        sys.exit(1)

    log_file = sys.argv[1]
    input_folder = sys.argv[2].rstrip("/")

    with open(log_file) as f:
        lines = f.readlines()

    total = passed = failed = skipped = 0

    print("=" * 70)
    print("  SOLUTION VALIDATOR")
    print("=" * 70)

    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [LINE {lineno}] Could not parse JSON: {line}")
            continue

        name    = entry.get("Instance", "???")
        result  = entry.get("Result", "--")
        sol_flag = entry.get("Solution", "--")
        sol_vec  = entry.get("SolVec", None)

        total += 1
        label = f"  {name:<30}"

        if result == "--" or sol_flag == "--":
            print(label + "SKIPPED  (no solution reported)")
            skipped += 1
            continue

        if sol_vec is None:
            print(label + "SKIPPED  (SolVec missing — did you update main.jl?)")
            skipped += 1
            continue

        ip_path = os.path.join(input_folder, name)
        if not os.path.isfile(ip_path):
            print(label + f"SKIPPED  (file not found: {ip_path})")
            skipped += 1
            continue

        try:
            n, m, costs, A = load_instance(ip_path)
        except Exception as e:
            print(label + f"SKIPPED  (error reading instance: {e})")
            skipped += 1
            continue

        ok, msg = validate(n, m, costs, A, list(map(float, sol_vec)), float(result))
        if ok:
            n_selected = sum(1 for v in sol_vec if v > 0.5)
            print(label + f"PASS     cost={result}  tests_selected={n_selected}/{n}")
            passed += 1
        else:
            print(label + f"FAIL     {msg}")
            failed += 1

    print("=" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped (of {total} total)")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    main()