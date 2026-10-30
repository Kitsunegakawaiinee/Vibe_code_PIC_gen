@echo off
setlocal

echo ========================================================
echo Installing stable-diffusion-cpp-python with Vulkan...
echo ========================================================

:: 1. Initialize Visual Studio 64-bit environment
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo [ERROR] Visual Studio vcvars64.bat not found!
    exit /b 1
)

:: 2. Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] .venv not found in current folder, using active Python.
)

:: 3. Ensure ninja is installed
pip install ninja

:: 4. Use Ninja generator
set "CMAKE_GENERATOR=Ninja"

:: 5. Install stable-diffusion-cpp-python with Vulkan backend
pip install --force-reinstall --no-cache-dir stable-diffusion-cpp-python -C cmake.args="-DSD_VULKAN=ON"

echo ========================================================
echo Installation complete! Testing import...
python -c "import stable_diffusion_cpp; print('SUCCESS: Version', stable_diffusion_cpp.__version__)"
echo ========================================================
pause
