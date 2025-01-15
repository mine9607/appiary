#!/bin/bash

# Shell script to execute training from the scripts file

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display error messages
function error_exit {
    echo "$1" 1>&2
    exit 1
}
'''
1. Activate the Conda Environment
    a) conda info --base outputs the base directory (/home/user/miniconda3)
    b) the $(...) syntax substitutes the output of the command in place of the command
    c) the appended /etc/profile.d/conda.sh finds the script responsible for initializing Conda in the current shell session
'''
echo "Activating the Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh" || error_exit "Failed to source conda.sh"
conda activate tensorflow || error_exit "Failed to activate the 'tensorflow' environment"
'''
2. Navigate to the Project Root Directory
    a) ${BASH_SOURCE[0]} refers to the current script file
    b) dirname "${BASH_SOURCE[0]}" - strips the filename from the path and returns the directory containing the script file
    c) cd "$(dirname "${BASH_SOURCE[0]}")" - changes the current working directory to the directory containing the script file
    d) pwd outputs the absolute path of the current working directory and stores in SCRIPT_DIR variable
    e) "$(dirname "$SCRIPT_DIR")" outputs the directory containing the SCRIPT_DIR to the PROJECT_ROOT variable
''' 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || error_exit "Failed to navigate to project root."

# 3. Create Necessary Directories
MODEL_SAVE_DIR="models"
mkdir -p "$MODEL_SAVE_DIR" || error_exit "Failed to create model save directory at $MODEL_SAVE_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" || error_exit "Failed to create log directory at $LOG_DIR"

# 4. Define Log Files
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/train_error_$(date +%Y%m%d_%H%M%S).log"

# 5. Execute the Training Script
echo "Starting model training..."
python scripts/train_model.py > "$TRAIN_LOG" 2> "$ERROR_LOG" || error_exit "Training script failed. Check $ERROR_LOG for details."

echo "Model training completed successfully."
echo "Training logs are saved to $TRAIN_LOG"
echo "Error logs (if any) are saved to $ERROR_LOG"
