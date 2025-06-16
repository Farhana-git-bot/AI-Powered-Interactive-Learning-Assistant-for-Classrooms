#!/bin/bash
# Installs project dependencies using a virtual environment and requirements.txt

echo "Creating Python virtual environment (venv) in current directory..."
# Use python3 to be more explicit on systems where 'python' might be Python 2
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment."
    echo "Please ensure Python 3 is installed and accessible as 'python3'."
    exit 1
fi

echo "Virtual environment created."
echo "Installing required packages from requirements.txt into local venv..."

# Path to pip within the virtual environment
VENV_PIP_PATH="./venv/bin/pip" # On Unix-like systems, it's in venv/bin/

# Check if pip exists in the venv
if [ ! -f "$VENV_PIP_PATH" ]; then
    echo "ERROR: pip not found in $VENV_PIP_PATH. Virtual environment setup might have failed."
    exit 1
fi

# Activating the venv is good practice, though explicit pip path makes it less strictly necessary
# for this specific pip command. It sets VIRTUAL_ENV and modifies PATH for subsequent commands
# if the script were to do more.
echo "Activating virtual environment (for this script session)..."
source ./venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment. Ensure ./venv/bin/activate exists."
    # Attempting to install without full activation if direct pip call is the goal,
    # but it's better if activation succeeds.
    # For robustness, we'll still try to use the explicit pip path.
fi


echo "Using pip from: $VENV_PIP_PATH"
"$VENV_PIP_PATH" install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install one or more packages from requirements.txt."
    echo "Please check your internet connection, the contents of requirements.txt,"
    echo "and ensure the virtual environment is clean."
    # If venv was activated, you might want to offer to deactivate it or just exit
    # if [ -n "$VIRTUAL_ENV" ]; then
    #   deactivate
    # fi
    exit 1
fi

echo ""
echo "Installation complete."
echo "The virtual environment 'venv' should now contain the installed packages."
echo "If you ran this script directly (e.g., './install.sh'), the venv is active in this terminal session."
echo "You can now run the application (e.g., using 'python3 sol_assistant.py' or a run.sh script)."
echo "To deactivate the virtual environment later, type: deactivate"
echo ""

# The script, when sourced or run directly, will leave the current terminal session
# with the venv activated if the 'source ./venv/bin/activate' line was successful.
# If run as a subprocess (e.g. from another script without 'source'), the activation
# would only apply to that subprocess.