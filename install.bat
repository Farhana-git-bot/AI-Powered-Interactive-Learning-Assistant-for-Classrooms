@echo off
REM Installs project dependencies using a virtual environment and requirements.txt

echo Creating Python virtual environment (venv) in current directory...
py -3 -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Please ensure Python 3 is installed and accessible via 'py -3' or 'python'.
    exit /b 1
)

echo Virtual environment created.
echo Installing required packages from requirements.txt into local venv...

set "VENV_PIP_PATH=.\venv\Scripts\pip.exe"

if not exist "%VENV_PIP_PATH%" (
    echo ERROR: pip not found in %VENV_PIP_PATH%. Virtual environment setup might have failed.
    exit /b 1
)

echo Activating virtual environment (ONLY FOR THIS SCRIPT SESSION)...
call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo WARNING: Failed to activate virtual environment. Ensure .\venv\Scripts\activate.bat exists.
    echo Will attempt to proceed using explicit pip path.
)

echo Using pip from: %VENV_PIP_PATH%
"%VENV_PIP_PATH%" install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install one or more packages from requirements.txt.
    echo Please check your internet connection, the contents of requirements.txt,
    echo and ensure the virtual environment is clean.
    exit /b 1
)

echo.
echo Installation complete.
echo The virtual environment 'venv' should now contain the installed packages.
echo If you ran this script directly (e.g., by double-clicking or from cmd as 'install.bat'),
echo the venv is active in this command prompt session.
echo You can now run the application (e.g., using 'python sol_assistant.py' or a run.bat script).
echo To deactivate the virtual environment later, type: deactivate
echo.

