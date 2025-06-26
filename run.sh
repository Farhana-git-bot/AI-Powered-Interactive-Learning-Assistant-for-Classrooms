
echo "Locating virtual environment..."

VENV_ACTIVATE_SCRIPT="./venv/bin/activate"
PYTHON_SCRIPT_PATH="sol_assistant.py"
VENV_PYTHON_EXE="./venv/bin/python" 

if [ ! -f "$VENV_ACTIVATE_SCRIPT" ]; then
    echo "ERROR: Virtual environment activation script not found at \"$VENV_ACTIVATE_SCRIPT\"."
    echo "Please ensure you have run install.sh to create and populate the venv."
    read -p "Press Enter to exit."
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Python script \"$PYTHON_SCRIPT_PATH\" not found in the current directory."
    read -p "Press Enter to exit."
    exit 1
fi

echo "Activating virtual environment..."
if ! source "$VENV_ACTIVATE_SCRIPT"; then
    echo "ERROR: Failed to activate the virtual environment."
    echo "Attempting to run script using explicit Python executable from venv."
    if [ ! -x "$VENV_PYTHON_EXE" ]; then
        echo "ERROR: Python executable not found or not executable at \"$VENV_PYTHON_EXE\"."
        echo "Virtual environment might be corrupted or incomplete."
        read -p "Press Enter to exit."
        exit 1
    fi
    echo "Running \"$PYTHON_SCRIPT_PATH\" using \"$VENV_PYTHON_EXE\"..."
    "$VENV_PYTHON_EXE" "$PYTHON_SCRIPT_PATH"
    SCRIPT_EXIT_CODE=$?
else
    echo "Virtual environment activated."
    echo "Running \"$PYTHON_SCRIPT_PATH\" using Python from activated venv (usually 'python' or 'python3')..."
    if command -v python3 &> /dev/null && [ -n "$VIRTUAL_ENV" ]; then
        python3 "$PYTHON_SCRIPT_PATH"
    else
        python "$PYTHON_SCRIPT_PATH"
    fi
    SCRIPT_EXIT_CODE=$?
fi

if [ $SCRIPT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "WARNING: \"$PYTHON_SCRIPT_PATH\" appears to have exited with an error (code: $SCRIPT_EXIT_CODE)."
    echo "Please check the output above for details from the script."
else
    echo ""
    echo "\"$PYTHON_SCRIPT_PATH\" finished successfully."
fi

echo ""
echo "If you ran this script from an existing terminal, the virtual environment"
echo "(if activation was successful) was active only for the duration of this script."
echo "To activate it in your current terminal, run: source $VENV_ACTIVATE_SCRIPT"
echo "Then, to deactivate, type: deactivate"
read -p "Press Enter to exit."
exit $SCRIPT_EXIT_CODE