#!/bin/bash

echo "Starting environment setup..."

# Step 1: Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH."
    exit 1
fi

# Step 2: Set Python executable and create venv
PYTHON_CMD=$(command -v python3 || command -v python)
ENV_DIR="./venv"

if [ -d "$ENV_DIR" ]; then
    echo "Virtual environment already exists at $ENV_DIR."
else
    echo "Creating virtual environment in $ENV_DIR..."
    "$PYTHON_CMD" -m venv "$ENV_DIR"
fi

# Step 3: Activate virtual environment (for Git Bash on Windows)
ACTIVATE_PATH="./venv/Scripts/activate"

if [ -f "$ACTIVATE_PATH" ]; then
    echo "Activating virtual environment..."
    source "$ACTIVATE_PATH"
else
    echo "Could not find activation script at $ACTIVATE_PATH"
    exit 1
fi

# Step 4: Upgrade pip and install dependencies
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Optional: install packages from requirements.txt
# if [ -f "ATB-calc/requirements.txt" ]; then
#     echo "Installing packages from ATB-calc/requirements.txt..."
#     pip install -r ATB-calc/requirements.txt
# else
#     echo "Warning: ATB-calc/requirements.txt not found."
# fi

# Step 4: Ensure Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed. Please install Julia first."
    exit 1
fi

# # Step 5: Set up Julia environment from GenX.jl
# echo "Setting up Julia environment from GenX.jl..."
# cd GenX.jl || { echo "GenX.jl directory not found."; exit 1; }

# julia --project=. -e '
# using Pkg;
# Pkg.instantiate();
# Pkg.precompile();
# println("GenX.jl Julia environment is set up.")
# '

# cd ..

echo "Setup complete!"