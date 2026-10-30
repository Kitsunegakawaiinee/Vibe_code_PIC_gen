import os
import shutil
import subprocess

def check_requirements():
    print("--- Checking Build Environment Requirements ---\n")
    
    # 1. Check CMake
    cmake_path = shutil.which("cmake")
    if cmake_path:
        print(f"[SUCCESS] CMake found at: {cmake_path}")
    else:
        print("[FAILED]  CMake is missing. Install from cmake.org and add to PATH.")

    # 2. Check Vulkan SDK / Headers
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        print(f"[SUCCESS] VULKAN_SDK environment variable set: {vulkan_sdk}")
    else:
        print("[WARNING] VULKAN_SDK environment variable not found. Ensure LunarG Vulkan SDK is installed.")

    # 3. Check Vulkan Driver Output
    try:
        res = subprocess.run(["vulkaninfo", "--summary"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "GPU" in res.stdout or "deviceName" in res.stdout or res.returncode == 0:
            print("[SUCCESS] Vulkan driver & runtime detected successfully.")
        else:
            print("[WARNING] vulkaninfo ran, but no active Vulkan devices were detected.")
    except FileNotFoundError:
        print("[FAILED]  'vulkaninfo' executable not found in PATH.")

    # 4. Check for MSVC Compiler (cl.exe)
    cl_path = shutil.which("cl")
    if cl_path:
        print(f"[SUCCESS] MSVC Compiler (cl.exe) found at: {cl_path}")
    else:
        print("[INFO]    'cl.exe' not in standard PATH. (This is normal unless using 'Developer Command Prompt for VS 2022').")

    print("\n------------------------------------------------")

if __name__ == "__main__":
    check_requirements()