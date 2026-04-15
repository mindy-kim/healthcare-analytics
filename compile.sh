#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

julia --project=. -e '
using Pkg
Pkg.add(["JuMP", "HiGHS", "JSON"])
Pkg.instantiate()
println("Julia packages installed successfully.")
'