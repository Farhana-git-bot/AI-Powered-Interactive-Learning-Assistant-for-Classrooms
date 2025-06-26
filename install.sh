

echo "Creating Python virtual environment (venv) in current directory..."
    echo "ERROR: Failed to create virtual environment."
    echo "Please ensure Python 3 is installed and accessible via 'python3' or 'python'."
    exit 1
fi

echo "Virtual environment created."
echo "Installing required packages from requirements.txt into local venv..."

VENV_PIP_PATH="./venv/bin/pip"

if [ ! -f "$VENV_PIP_PATH" ]; then
    echo "ERROR: pip not found in $VENV_PIP_PATH. Virtual environment setup might have failed."
    exit 1
fi

echo "Activating virtual environment (for this script session)..."
if ! source ./venv/bin/activate; then
    echo "WARNING: Failed to activate virtual environment. Ensure ./venv/bin/activate exists."
    echo "Will attempt to proceed using explicit pip path."
fi

echo "Using pip from: $($VENV_PIP_PATH --version | cut -d' ' -f1,2) (resolved: $(command -v pip))"
INSTALL_CMD="$VENV_PIP_PATH"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is active. Using 'pip' from PATH."
    INSTALL_CMD="pip"
else
    echo "Virtual environment not active or activation failed. Using explicit path: $VENV_PIP_PATH"
fi


"$INSTALL_CMD" install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install one or more packages from requirements.txt."
    echo "Please check your internet connection, the contents of requirements.txt,"
    echo "and ensure the virtual environment is clean."
    exit 1
fi

echo ""
echo "Installation complete."
echo "The virtual environment 'venv' should now contain the installed packages."
echo "Please launch run.sh to start the assistant"

