#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install requirements into the virtual environment
./venv/bin/pip install -r requirements.txt