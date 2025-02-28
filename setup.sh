#!/bin/bash

echo "Starting environment setup..."

# Step 1: Ensure Conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH. Please install Conda first."
    exit 1
fi

# Step 2: Check if Conda environment exists before creating it
if conda info --envs | grep -q "master-env"; then
    echo "Conda environment 'master-env' already exists. Updating Python version to 3.11..."
    conda init
    conda activate master-env
    conda install python=3.11 -y
else
    echo "Creating Conda environment..."
    conda create --name master-env python=3.11 -y
fi

# Activate the environment
source ~/.bashrc  # Ensure Conda is initialized
conda init
conda activate master-env

# Step 3: Install Python dependencies if they are missing
echo "Checking Python dependencies..."
pip_packages=("numpy" "pandas" "matplotlib" "IPython" \
                "os" 
                "sys" 
                "datetime" 
                "ipykernel"
                "pyyaml" 
                "subprocess"
                "openpyxl" 
                "click"
                "et-xmlfile"
                "exceptiongroup"
                "iniconfig"
                "lxml"
                "packaging"
                "pluggy"
                "pytest"
                "python-dateutil"
                "pytz"
                "six"
                "tomli"
                "tzdata"
                "xlwings"
                )

          
for package in "${pip_packages[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        echo "Installing missing Python package: $package"
        conda install "$package" -y
    else
        echo "Python package '$package' is already installed."
    fi
done

# # Install all packages listed in ATB-calc/requirements.txt
# if [ -f "ATB-calc/requirements.txt" ]; then
#     echo "Installing packages from ATB-calc/requirements.txt..."
#     conda install --file ATB-calc/requirements.txt -y
# else
#     echo "Error: ATB-calc/requirements.txt not found."
#     exit 1
# fi


# Step 4: Ensure Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed. Please install Julia first."
    exit 1
fi

# Step 5: Set up Julia environment from public-repo
echo "Setting up Julia environment from GenX.jl..."
cd GenX.jl  # Navigate to the public repository

# # Temporarily disable SSL_CERT_FILE for Julia
# export JULIA_SSL_CA_ROOTS_PATH=""

# Activate and instantiate the Julia project
julia --project=. -e '
using Pkg;
Pkg.instantiate();
Pkg.precompile();
println("GenX.jl Julia environment is set up.")
'

# # Temporarily disable SSL_CERT_FILE for Julia
# export JULIA_SSL_CA_ROOTS_PATH=""

# Return to master-folder
cd ..

# # Step 6: Install additional Julia packages
# echo "Installing additional Julia packages..."
# julia --project=. -e '
# using Pkg;
# additional_packages = [
#                         "FilePathsBase",
#                         "XLSX", 
#                         "TextWrap",
#                         ];
# for pkg in additional_packages
#     if !haskey(Pkg.dependencies(), pkg)
#         println("Installing Julia package: $pkg")
#         Pkg.add(pkg)
#     else
#         println("Julia package $pkg is already installed.")
#     end
# end
# Pkg.precompile();
# '






echo "Setup complete!"