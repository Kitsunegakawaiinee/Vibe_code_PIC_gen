# Stable Diffusion (stable-diffusion-cpp-python) with Vulkan on Windows

Setup guide for running Stable Diffusion locally with `stable-diffusion-cpp-python`, built with Vulkan GPU support.

## Requirements

- Python
- CMake
- A C++ compiler (`cl`, from Visual Studio)
- Vulkan SDK
- Ninja (installed inside the virtual environment)

## Installation

### 1. Install Python

Download and install Python from [python.org](https://www.python.org/downloads/).

### 2. Create a virtual environment (recommended, not required)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install the build tools

1. **CMake**: install it on your computer.
2. **cl (MSVC compiler)**: install it on your computer. The Visual Studio Installer is an easy way to get it.
3. **Vulkan**: install the Vulkan SDK.

### 4. Build `stable-diffusion-cpp-python`

1. Activate your virtual environment (if you use one).

2. Install Ninja in the virtual environment:

   ```powershell
   pip install ninja
   ```

3. Load the Visual Studio 64-bit environment into the current PowerShell session. This is needed when you installed the compiler through the Visual Studio Installer:

   ```powershell
   cmd /c "call `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" && set" | ForEach-Object {
       if ($_ -match '^(.*?)=(.*)$') { Set-Content "env:\$($matches[1])" $matches[2] }
   }
   ```

   > **Note:** `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat` is the path to `vcvars64.bat` on my machine. Change it to match your own Visual Studio version and edition.

4. Set the CMake generator to Ninja:

   ```powershell
   $env:CMAKE_GENERATOR = "Ninja"
   ```

5. Build and install with Vulkan support:

   ```powershell
   pip install --force-reinstall --no-cache-dir stable-diffusion-cpp-python -C cmake.args="-DSD_VULKAN=ON"
   ```

## Usage

1. **Download your model** and place it in the `model/` folder.

   Models I have used:

   ```
   waiIllustriousSDXL_v170.safetensors
   ponyDiffusionV6XL_v6StartWithThisOne.safetensors
   ```

2. **Run it.** Generated images are saved automatically to the `outputs/` folder.

> **Note:** Loading a model into memory can take some time.
