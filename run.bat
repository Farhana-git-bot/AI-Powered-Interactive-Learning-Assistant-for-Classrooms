@echo off

echo Locating virtual environment...

REM Define paths
set "VENV_ACTIVATE_SCRIPT=.\venv\Scripts\activate.bat"
set "PYTHON_SCRIPT_PATH=sol_assistant.py"
set "VENV_PYTHON_EXE=.\venv\Scripts\python.exe"

if not exist "%VENV_ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment activation script not found at "%VENV_ACTIVATE_SCRIPT%".
    echo Please ensure you have run install.bat to create and populate the venv.
    pause
    exit /b 1
)

if not exist "%PYTHON_SCRIPT_PATH%" (
    echo ERROR: Python script "%PYTHON_SCRIPT_PATH%" not found in the current directory.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_ACTIVATE_SCRIPT%"
if errorlevel 1 (
    echo ERROR: Failed to activate the virtual environment.
    echo Attempting to run script using explicit Python executable from venv.
    if not exist "%VENV_PYTHON_EXE%" (
        echo ERROR: Python executable not found at "%VENV_PYTHON_EXE%".
        echo Virtual environment might be corrupted or incomplete.
        pause
        exit /b 1
    )
    echo Running "%PYTHON_SCRIPT_PATH%" using "%VENV_PYTHON_EXE%"...
    "%VENV_PYTHON_EXE%" "%PYTHON_SCRIPT_PATH%"
) else (
    echo Virtual environment activated.
    echo Running "%PYTHON_SCRIPT_PATH%" using Python from activated venv...
    python "%PYTHON_SCRIPT_PATH%"
)

if errorlevel 1 (
    echo.
    echo WARNING: "%PYTHON_SCRIPT_PATH%" appears to have exited with an error.
    echo Please check the output above for details from the script.
) else (
    echo.
    echo "%PYTHON_SCRIPT_PATH%" finished.
)

echo.
echo If this window was opened by double-clicking the .bat file, it will close after you press a key.
echo If you ran this .bat from an existing command prompt, the virtual environment is still active.
echo You can type 'deactivate' to exit the virtual environment if needed.
pause