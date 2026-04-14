#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` <input>"
	exit $E_BADARGS
fi
	
input=$1

if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="python3"
fi

# run the solver
$PYTHON src/main.py $input
