#!/bin/bash

# Log file path
LOG_FILE=spyder_setup.log

# Function to log messages
log_message() {
    local message="$1"
    local color="$2"
    case $color in
        red)
            echo -e "$(date +'%Y-%m-%d %H:%M:%S') - \e[91m$message\e[0m" | tee -a "$LOG_FILE"
            ;;
        yellow)
            echo -e "$(date +'%Y-%m-%d %H:%M:%S') - \e[93m$message\e[0m" | tee -a "$LOG_FILE"
            ;;
        green)
            echo -e "$(date +'%Y-%m-%d %H:%M:%S') - \e[92m$message\e[0m" | tee -a "$LOG_FILE"
            ;;
        *)
            echo "$(date +'%Y-%m-%d %H:%M:%S') - $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Check if Conda is installed
if ! command -v conda &>/dev/null; then
    log_message "Anaconda is not installed. Please install Anaconda before running this script." red
    return 1
fi

# Check if Conda environment name is provided as argument
if [ $# -eq 0 ]; then
    log_message "Usage: $0 <conda_environment_name>" red
    return 1
fi

# Conda environment name provided as argument
conda_environment="$1"

# Create log file
touch "$LOG_FILE"

# Activate Conda environment
log_message "Activating Conda environment $conda_environment..."
source activate "$conda_environment" >> "$LOG_FILE" 2>&1

# Step 1: Check if locate command exists
log_message "Step 1: Checking if locate command exists..."
if ! command -v locate &>/dev/null; then
    # Install plocate if locate command doesn't exist
    log_message "locate command not found. Installing plocate..." yellow
    sudo apt-get update >> "$LOG_FILE" 2>&1
    sudo apt-get install -y mlocate >> "$LOG_FILE" 2>&1
    log_message "plocate installed." green
else
    log_message "locate command found." green
fi

# Step 2: Find the location of libsoftokn3.so
log_message "Step 2: Searching for libsoftokn3.so..."
libsoftokn_path=$(locate libsoftokn3.so)
if [ -z "$libsoftokn_path" ]; then
    log_message "Error: libsoftokn3.so not found." red
else
    log_message "libsoftokn3.so found at: $libsoftokn_path" green
fi

# Step 3: Check if necessary dependencies are installed in Conda environment
log_message "Step 3: Checking if necessary dependencies are installed in Conda environment $conda_environment..."
if ! python -c "import PyQt5.QtWebEngineWidgets" &>/dev/null; then
    log_message "Error: Required dependencies not installed in Conda environment $conda_environment. Installing now..." yellow
    conda install -y pyqt >> "$LOG_FILE" 2>&1
    log_message "Required dependencies installed in Conda environment $conda_environment." green
else
    log_message "Required dependencies are installed in Conda environment $conda_environment." green
fi

# Step 4: Verifying and setting environment variables in Conda environment
log_message "Step 4: Verifying environment variables in Conda environment $conda_env_name..."
ld_library_path=$(conda run -n "$conda_env_name" printenv LD_LIBRARY_PATH)
qt_plugin_path=$(conda run -n "$conda_env_name" printenv QT_PLUGIN_PATH)

if [ -z "$ld_library_path" ]; then
    log_message "Error: LD_LIBRARY_PATH not set in Conda environment $conda_env_name. Setting it now..."
    export LD_LIBRARY_PATH=$(dirname "$libsoftokn_path")
    log_message "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH" "red"
else
    log_message "LD_LIBRARY_PATH in Conda environment $conda_env_name: $ld_library_path" "green"
fi

if [ -z "$qt_plugin_path" ]; then
    log_message "Error: QT_PLUGIN_PATH not set in Conda environment $conda_env_name. Setting it now..."
    export QT_PLUGIN_PATH=$(dirname "$libsoftokn_path")/qt/plugins
    log_message "QT_PLUGIN_PATH set to: $QT_PLUGIN_PATH" "red"
else
    log_message "QT_PLUGIN_PATH in Conda environment $conda_env_name: $qt_plugin_path" "green"
fi


# Step 5: Launch Spyder
log_message "Step 5: Launching Spyder..."
spyder &>> "$LOG_FILE" &

log_message "Setup complete. Check $LOG_FILE for details." green

