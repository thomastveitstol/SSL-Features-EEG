#!/bin/bash

# Clear terminal screen
clear

# Path to the old virtual environment
old_env="/home/thomas/PycharmProjects/my_env/venv"

# Activate the old environment
source "$old_env/bin/activate"

# Install packages
# pip install optuna seaborn matplotlib
# pip install --upgrade scipy
# pip install --upgrade optuna

# Run the plotting
python -m importance_plots