#!/bin/bash

# Shell script to execute training from the scripts file

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display error messages
function error_exit {
    echo "$1" 1>&2
    exit 1
}

echo "Activating the Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh" || error_exit "Failed to source conda.sh"
conda activate tensorflow || error_exit "Failed to activate the 'tensorflow' environment"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT" || error_exit "Failed to navigate to project root."

# Set the pythonpath to the src directory (where to look for modules)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run the FastAPI app with uvicorn
uvicorn src.main:app --host localhost --port 8000 --reload