#!/bin/bash

echo "Starting environment setup..."

# Step 1: Ensure Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH. Please install Python3 first."
    exit 1
fi

# Step 2: Check if virtualenv is installed
if ! python3 -m pip show virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    python3 -m pip install virtualenv
fi

# Step 3: Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m virtualenv venv
else
    echo "Virtual environment already exists. Updating Python version to 3.11..."
    source venv/bin/activate
    python3 -m pip install python==3.11
fi

# Activate the virtual environment
source venv/bin/activate

# Step 4: Install Python dependencies if they are missing
echo "Checking Python dependencies..."
pip_packages=("numpy" "pandas" "matplotlib" "IPython" \
                "os" "sys" "datetime" "ipykernel"
                "yaml" "subprocess" "openpyxl" 
                "click")

for package in "${pip_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        echo "Installing missing Python package: $package"
        pip install "$package"
    else
        echo "Python package '$package' is already installed."
    fi
done

# Step 5: Ensure Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed. Please install Julia first."
    exit 1
fi

# Step 6: Set up Julia environment from public-repo
echo "Setting up Julia environment from GenX.jl..."
cd GenX.jl  # Navigate to the public repository

# Activate and instantiate the Julia project
julia --project=. -e '
using Pkg;
Pkg.instantiate();
Pkg.precompile();
println("GenX.jl Julia environment is set up.")
'

# Return to master-folder
cd ..

echo "Setup complete!"
